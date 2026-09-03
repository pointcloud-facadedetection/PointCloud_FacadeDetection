from __future__ import annotations

from dataclasses import dataclass, field
import time
import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class RegistrationConfig:
    """已全局化的PLY点云残余配准参数。"""
    voxel_size: float = 0.05
    max_correspondence_distance: float = 0.25
    max_iteration: int = 40
    pyramid_scales: tuple[float, ...] = (4.0, 2.0, 1.0)
    z_lock: bool = True          # 是否优先保持Z轴对齐（GPS已全局对齐高程）
    max_z_shift: float = 0.15    # Z平移最大允许修正量（米），GPS典型精度10-20cm
    z_damping: float = 0.3       # Z轴阻尼系数：每轮ICP后Z平移向0回归的比例



# 辅助函数：Z轴软约束（阻尼回归，非硬截断）
def _apply_z_damping(T, z_damping=0.3, max_z_shift=0.15):
    """对变换矩阵施加Z轴阻尼回归，而非硬截断。

    GPS已全局对齐扫描仪姿态，残余误差主要在XY平面。
    策略：
    1. 保持XY平面内的旋转和平移（2D刚体变换）
    2. 对Z平移施加阻尼：向0回归 z_damping 比例
    3. 仅当Z平移超过 max_z_shift 时才进行软限制（缩放而非截断）
    4. 消除绕X/Y轴的微小旋转（保持扫描仪水平）

    这种软约束允许GPS高程存在一定残余误差（10-20cm典型值），
    同时避免ICP过度优化Z方向导致立面错位。
    """
    T = np.asarray(T, dtype=np.float64).copy()
    R = T[:3, :3]

    # 提取XY平面内的二维旋转角度（绕全局Z轴）
    theta = np.arctan2(R[1, 0], R[0, 0])
    c, s_val = np.cos(theta), np.sin(theta)

    # 重建纯Z轴旋转矩阵，强制消除俯仰/横滚（扫描仪始终水平）
    R_locked = np.array([
        [c, -s_val, 0],
        [s_val,  c, 0],
        [0,  0, 1]
    ], dtype=np.float64)
    T[:3, :3] = R_locked

    # Z平移阻尼回归：向0靠近 z_damping 比例
    z_current = T[2, 3]
    z_damped = z_current * (1.0 - z_damping)

    # 软限制：仅当超过阈值时额外缩放
    if abs(z_damped) > max_z_shift:
        scale = max_z_shift / abs(z_damped)
        z_damped = z_damped * (0.5 + 0.5 * scale)  # 软着陆，不完全截断

    T[2, 3] = z_damped
    return T


def _xy_overlap_iou(source_xy, target_xy):
    """计算XY投影边界框的IoU重叠率，用于立面180°歧义判别。"""
    def _bbox(pts):
        if len(pts) == 0:
            return (0, 0, 0, 0)
        return (float(pts[:, 0].min()), float(pts[:, 0].max()),
                float(pts[:, 1].min()), float(pts[:, 1].max()))
    sx_min, sx_max, sy_min, sy_max = _bbox(source_xy)
    tx_min, tx_max, ty_min, ty_max = _bbox(target_xy)
    ix_min = max(sx_min, tx_min)
    ix_max = min(sx_max, tx_max)
    iy_min = max(sy_min, ty_min)
    iy_max = min(sy_max, ty_max)
    inter = max(0.0, ix_max - ix_min) * max(0.0, iy_max - iy_min)
    s_area = max(0.0, sx_max - sx_min) * max(0.0, sy_max - sy_min)
    t_area = max(0.0, tx_max - tx_min) * max(0.0, ty_max - ty_min)
    union = s_area + t_area - inter
    return float(inter / union) if union > 1e-8 else 0.0


# 1. 配准质量评估
def registration_metrics(source_points, target_points, transformation,
                          max_correspondence_distance,
                          source_normals=None, target_normals=None,
                          compute_normals=True):
    """在配准域上重新计算质量指标，可选法向量一致性检查。

    性能提示：当点云较大时，调用方应传入下采样后的点云。
    若compute_normals=False则跳过法向量一致性计算以加速。
    """
    source = np.asarray(source_points, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    t = np.asarray(transformation, dtype=np.float64).reshape(4, 4)
    if len(source) == 0 or len(target) == 0:
        return {
            'metric_domain': 'registration_downsample', 'fitness': 0.0,
            'rmse': float('inf'), 'mean_error': float('inf'),
            'p95_error': float('inf'), 'max_error': float('inf'),
            'correspondence_count': 0, 'normal_consistency': 0.0,
        }
    moved = source @ t[:3, :3].T + t[:3, 3]
    target_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
    moved_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(moved))
    errors = np.asarray(moved_pcd.compute_point_cloud_distance(target_pcd),
                        dtype=np.float64)
    errors = errors[errors <= float(max_correspondence_distance)]
    if len(errors) == 0:
        rmse = mean = p95 = maximum = float('inf')
    else:
        rmse = float(np.sqrt(np.mean(errors * errors)))
        mean = float(np.mean(errors))
        p95 = float(np.percentile(errors, 95))
        maximum = float(np.max(errors))

    # 法向量一致性：正确配准后立面法向量应对齐
    normal_consistency = 0.0
    if (compute_normals and source_normals is not None and target_normals is not None
            and len(source_normals) == len(source) and len(target_normals) == len(target)):
        moved_normals = source_normals @ t[:3, :3].T
        tree = o3d.geometry.KDTreeFlann(target_pcd)
        dots = []
        for i, point in enumerate(moved):
            count, idx, _ = tree.search_knn_vector_3d(point, 1)
            if count:
                dp = np.clip(np.dot(moved_normals[i], target_normals[idx[0]]), -1.0, 1.0)
                dots.append(dp)
        normal_consistency = float(np.median(dots)) if dots else 0.0

    return {
        'metric_domain': 'registration_downsample',
        'fitness': float(len(errors) / len(source)),
        'rmse': rmse,
        'mean_error': mean,
        'p95_error': p95,
        'max_error': maximum,
        'correspondence_count': int(len(errors)),
        'normal_consistency': normal_consistency,
    }


# 2. 配准域点云缓存
@dataclass
class RegistrationCloud:
    """从代理点云派生的不可变配准域快照。"""
    points: np.ndarray
    colors: np.ndarray | None
    voxel_size: float
    normals: np.ndarray | None = None  # 预计算法向量，用于立面配准

    def as_open3d(self) -> o3d.geometry.PointCloud:
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.points))
        if self.colors is not None and len(self.colors) == len(self.points):
            cloud.colors = o3d.utility.Vector3dVector(self.colors)
        if self.normals is not None and len(self.normals) == len(self.points):
            cloud.normals = o3d.utility.Vector3dVector(self.normals)
        return cloud


def build_registration_cloud(points, colors=None, voxel_size=0.05,
                              estimate_normals=True) -> RegistrationCloud:
    """创建配准域，可选预计算法向量。"""
    voxel_size = max(float(voxel_size), 1e-4)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1)
    pts = np.ascontiguousarray(pts[valid])
    cols = None
    if colors is not None:
        raw_cols = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
        if len(raw_cols) == len(valid):
            cols = np.ascontiguousarray(raw_cols[valid])
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    if cols is not None:
        cloud.colors = o3d.utility.Vector3dVector(cols)
    down = cloud.voxel_down_sample(voxel_size)
    down_pts = np.asarray(down.points, dtype=np.float64)
    down_cols = np.asarray(down.colors, dtype=np.float64) if down.has_colors() else None

    # 预计算法向量用于立面配准（初始估计和指标计算均使用）
    normals = None
    if estimate_normals and len(down_pts) >= 3:
        radius = max(voxel_size * 3.0, 0.015)
        down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50))
        down.normalize_normals()
        normals = np.asarray(down.normals, dtype=np.float64)

    return RegistrationCloud(
        np.ascontiguousarray(down_pts),
        np.ascontiguousarray(down_cols) if down_cols is not None else None,
        voxel_size,
        normals,
    )


# 3. ICP 配准算法
@dataclass
class ICPResult:
    transformation: np.ndarray
    fitness: float
    inlier_rmse: float
    correspondence_count: int
    levels: list = field(default_factory=list)
    accepted: bool = True
    message: str = ''
    timing_s: dict = field(default_factory=dict)  # 各阶段耗时（秒）
    normal_consistency: float = 0.0  # 配准后法向量一致性
    warnings: list = field(default_factory=list)  # 非致命警告信息


def rigid_transform_from_correspondences(source_points, target_points):
    """最小二乘刚体变换求解（Kabsch算法，无缩放）。"""
    src = np.asarray(source_points, dtype=np.float64).reshape(-1, 3)
    tgt = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    if len(src) != len(tgt) or len(src) < 3:
        raise ValueError('人工对应点至少需要 3 对且数量必须一致')
    if not np.isfinite(src).all() or not np.isfinite(tgt).all():
        raise ValueError('人工对应点包含无效坐标')
    if (len(np.unique(src, axis=0)) < 3 or
            len(np.unique(tgt, axis=0)) < 3):
        raise ValueError('人工对应点包含重复点，无法稳定求解刚体变换')
    if np.linalg.matrix_rank(src - src.mean(axis=0)) < 2:
        raise ValueError('源点近似共线，无法稳定求解刚体变换')
    if np.linalg.matrix_rank(tgt - tgt.mean(axis=0)) < 2:
        raise ValueError('目标点近似共线，无法稳定求解刚体变换')
    cs, ct = src.mean(axis=0), tgt.mean(axis=0)
    u, _, vt = np.linalg.svd((src - cs).T @ (tgt - ct))
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = ct - r @ cs
    out = np.eye(4, dtype=np.float64)
    out[:3, :3], out[:3, 3] = r, t
    return out


def _xy_pca_frame(points):
    """为立面代理点云返回稳定的二维主方向坐标系。"""
    xy = np.asarray(points, dtype=np.float64).reshape(-1, 3)[:, :2]
    center = np.mean(xy, axis=0)
    centered = xy - center
    if len(centered) < 2:
        return center, np.eye(2)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    frame = vt.T
    if np.linalg.det(frame) < 0:
        frame[:, 1] *= -1.0
    return center, frame


def estimate_xy_initial_transform(source_points, target_points,
                                   source_normals=None, target_normals=None,
                                   z_lock=True, max_z_shift=0.15, z_damping=0.3):
    """从XY几何估计GPS残余刚体初值。

    1. Z轴软约束：阻尼回归而非硬截断，允许GPS高程残余误差
    2. XY投影IoU评分：解决立面180°歧义（法向量在立面场景区分度不足）
    3. 法向量一致性增强：结合中位数与正向比例，更稳健
    4. 预建单个目标KDTree（性能）
    5. 分层采样处理大点云（性能）
    """
    t0 = time.perf_counter()
    src = np.asarray(source_points, dtype=np.float64).reshape(-1, 3)
    tgt = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    src = src[np.isfinite(src).all(axis=1)]
    tgt = tgt[np.isfinite(tgt).all(axis=1)]
    if len(src) < 3 or len(tgt) < 3:
        raise ValueError('粗配准至少需要两组各 3 个有效点')

    sc, sf = _xy_pca_frame(src)
    tc, tf = _xy_pca_frame(tgt)

    # 分层采样：根据点云大小动态调整，百万级点云采样更多点
    src_sample = max(50000, min(100000, len(src) // 100))
    tgt_sample = max(50000, min(100000, len(tgt) // 100))
    step_s = max(1, len(src) // src_sample)
    step_t = max(1, len(tgt) // tgt_sample)
    s = src[::step_s]
    t = tgt[::step_t]

    # 预建目标KDTree（仅一次）
    t_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(t))
    t_tree = o3d.geometry.KDTreeFlann(t_pcd)

    # 预计算目标法向量
    t_normals = None
    if target_normals is not None and len(target_normals) == len(tgt):
        t_normals = target_normals[::step_t]
    else:
        t_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=np.std(t[:, :2]) * 0.3 + 0.05, max_nn=30))
        t_pcd.normalize_normals()
        t_normals = np.asarray(t_pcd.normals, dtype=np.float64)

    # 预计算源法向量
    s_normals = None
    if source_normals is not None and len(source_normals) == len(src):
        s_normals = source_normals[::step_s]
    else:
        s_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(s))
        s_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=np.std(s[:, :2]) * 0.3 + 0.05, max_nn=30))
        s_pcd.normalize_normals()
        s_normals = np.asarray(s_pcd.normals, dtype=np.float64)

    # 高度（Z）统计用于平局决胜
    src_z_std = np.std(src[:, 2])
    tgt_z_std = np.std(tgt[:, 2])

    best = None
    candidates = []
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            q = np.diag([sx, sy])
            r2 = tf @ q @ sf.T
            if np.linalg.det(r2) < 0:
                continue

            # 若启用Z轴锁定，仅保留XY平面旋转
            if z_lock:
                theta = np.arctan2(r2[1, 0], r2[0, 0])
                c, s_val = np.cos(theta), np.sin(theta)
                r = np.array([[c, -s_val, 0], [s_val, c, 0], [0, 0, 1]], dtype=np.float64)
            else:
                r = np.eye(3)
                r[:2, :2] = r2

            trans = np.zeros(3, dtype=np.float64)
            trans[:2] = tc - r2 @ sc
            trans[2] = np.median(t[:, 2]) - np.median(s[:, 2])

            moved = s @ r.T + trans

            # 1. 距离评分（最近邻中位数距离）
            distances = []
            for point in moved:
                count, _, d2 = t_tree.search_knn_vector_3d(point, 1)
                if count:
                    distances.append(d2[0])
            dist_score = float(np.median(distances)) if distances else float('inf')

            # 2. XY投影IoU评分（新增：立面场景180°歧义的关键判别依据）
            iou_score = _xy_overlap_iou(moved[:, :2], t[:, :2])

            # 3. 法向量一致性评分（增强：结合中位数与正向比例）
            moved_normals = s_normals @ r.T
            normal_dots = []
            for i, point in enumerate(moved):
                count, idx, _ = t_tree.search_knn_vector_3d(point, 1)
                if count:
                    dp = np.clip(np.dot(moved_normals[i], t_normals[idx[0]]), -1.0, 1.0)
                    normal_dots.append(dp)
            if normal_dots:
                normal_median = float(np.median(normal_dots))
                positive_ratio = sum(1 for d in normal_dots if d > 0.3) / len(normal_dots)
                # 综合法向量评分：中位数占60%，正向比例占40%
                normal_score = 0.6 * normal_median + 0.4 * (positive_ratio * 2.0 - 1.0)
            else:
                normal_score = -1.0

            # 4. 高度分布相似性
            moved_z_std = np.std(moved[:, 2])
            z_similarity = 1.0 / (1.0 + abs(moved_z_std - tgt_z_std) * 10.0)

            # 综合评分公式
            # 距离为主；IoU低时大幅惩罚（投影不重叠说明方向错误）
            # 法向量一致性作为辅助；Z分布作为平局决胜
            iou_penalty = max(0.0, 1.0 - iou_score * 2.0)  # IoU<0.5时开始惩罚

            if normal_score < -0.3:
                composite = dist_score * 6.0 + iou_penalty * 10.0 + (1.0 - normal_score) * 3.0
            elif normal_score < 0.1:
                composite = dist_score * 2.0 + iou_penalty * 5.0 + (0.5 - normal_score) * 1.0
            else:
                composite = dist_score + iou_penalty * 2.0 - normal_score * 0.3

            # Z分布相似性作为乘法因子微调
            composite = composite * (1.5 - z_similarity)

            candidates.append({
                'sx': sx, 'sy': sy,
                'composite': composite,
                'dist_score': dist_score,
                'iou_score': iou_score,
                'normal_score': normal_score,
                'z_similarity': z_similarity,
                'rotation': r.copy(),
                'translation': trans.copy(),
            })

            if best is None or composite < best['composite']:
                best = candidates[-1]

    if best is None:
        raise ValueError('无法从 XY 几何建立粗配准初值')
    out = np.eye(4, dtype=np.float64)
    out[:3, :3], out[:3, 3] = best['rotation'], best['translation']
    # 最终施加Z轴阻尼回归（软约束）
    if z_lock:
        out = _apply_z_damping(out, z_damping=z_damping, max_z_shift=max_z_shift)
    return out


def manual_seeded_icp(source_points, target_points, source_correspondences,
                       target_correspondences, *, voxel_size=0.05,
                       max_iteration=30, max_correspondence_distance=0.12,
                       pyramid_scales=(4.0, 2.0, 1.0),
                       z_lock=True, max_z_shift=0.15, z_damping=0.3):
    """由人工对应点引导的全局坐标ICP。"""
    src_pairs = np.asarray(source_correspondences, dtype=np.float64).reshape(-1, 3)
    tgt_pairs = np.asarray(target_correspondences, dtype=np.float64).reshape(-1, 3)
    init = rigid_transform_from_correspondences(src_pairs, tgt_pairs)
    # 人工选点时同样施加Z轴阻尼
    if z_lock:
        init = _apply_z_damping(init, z_damping=z_damping, max_z_shift=max_z_shift)
    result = point_to_plane_icp(source_points, target_points, init=init,
                                voxel_size=voxel_size, max_iteration=max_iteration,
                                max_correspondence_distance=max_correspondence_distance,
                                pyramid_scales=pyramid_scales,
                                z_lock=z_lock, max_z_shift=max_z_shift, z_damping=z_damping)
    initial_rmse = float(np.sqrt(np.mean(np.sum((src_pairs @ init[:3, :3].T + init[:3, 3] - tgt_pairs) ** 2, axis=1))))
    final_pairs = src_pairs @ result.transformation[:3, :3].T + result.transformation[:3, 3]
    pair_rmse = float(np.sqrt(np.mean(np.sum((final_pairs - tgt_pairs) ** 2, axis=1))))
    if pair_rmse > max(initial_rmse * 1.5, voxel_size * 2.0):
        result.accepted = False
        result.message = 'ICP 后人工对应点残差变大，请检查同名点选择'
    result.levels.insert(0, {'manual_initial_rmse': initial_rmse,
                             'manual_final_rmse': pair_rmse})
    return result


def point_to_plane_icp(source_points, target_points, *, init=None, voxel_size=0.05,
                       max_correspondence_distance=None, max_iteration=30,
                       pyramid_scales=(2.0, 1.0),
                       source_normals=None, target_normals=None,
                       z_lock=True, max_z_shift=0.15, z_damping=0.3):
    """估计同一全局坐标系下两朵点云之间的残余变换Delta T。

    1. Z轴软约束：阻尼回归而非硬截断，允许GPS高程残余误差（10-20cm典型值）
    2. 自适应金字塔：根据点数自动选择更粗的初始层级以加速
    3. 所有层级使用点到面（立面场景平面约束更强）
    4. 指标计算使用下采样点云（性能）
    5. 法向量一致性作为警告（非拒绝条件），立面场景法向量Z分量小
    """
    t_total = time.perf_counter()
    src = np.asarray(source_points, dtype=np.float64).reshape(-1, 3)
    tgt = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    src = src[np.isfinite(src).all(axis=1)]
    tgt = tgt[np.isfinite(tgt).all(axis=1)]
    if len(src) < 3 or len(tgt) < 3:
        raise ValueError('点到面 ICP 至少需要两组各 3 个有效点')

    # 自适应金字塔：大点云使用更粗的初始层级
    min_pts = min(len(src), len(tgt))
    if min_pts > 5000000:
        adaptive_scales = (10.0, 5.0, 2.0, 1.0)
    elif min_pts > 1000000:
        adaptive_scales = (8.0, 4.0, 2.0, 1.0)
    elif min_pts > 500000:
        adaptive_scales = pyramid_scales
    else:
        adaptive_scales = pyramid_scales

    scales = [float(voxel_size) * float(level) for level in adaptive_scales] if voxel_size else [0.05]
    if not scales:
        raise ValueError('ICP 至少需要一个有效金字塔层级')
    scales = [max(x, 1e-3) for x in scales]

    # 自适应迭代次数：粗层级减少迭代
    adaptive_iters = []
    for i, _ in enumerate(scales):
        if i == 0 and len(scales) > 2:
            adaptive_iters.append(max(15, int(max_iteration * 0.5)))
        elif i == len(scales) - 1:
            adaptive_iters.append(int(max_iteration))
        else:
            adaptive_iters.append(max(20, int(max_iteration * 0.75)))

    T = np.eye(4) if init is None else np.asarray(init, dtype=np.float64).reshape(4, 4)
    # 初始变换施加Z轴阻尼
    if z_lock:
        T = _apply_z_damping(T, z_damping=z_damping, max_z_shift=max_z_shift)

    reports = []
    final = None
    level_times = []
    warnings = []

    base_source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src))
    base_target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt))

    finest_source_pts = None
    finest_target_pts = None
    finest_source_normals = None
    finest_target_normals = None

    for i, scale in enumerate(scales):
        t_level = time.perf_counter()
        source = base_source.voxel_down_sample(scale)
        target = base_target.voxel_down_sample(scale)

        # 保存最细层级下采样点云用于高效指标计算
        if i == len(scales) - 1:
            finest_source_pts = np.asarray(source.points, dtype=np.float64)
            finest_target_pts = np.asarray(target.points, dtype=np.float64)

        radius = max(scale * 2.5, 1e-3)
        for cloud in (source, target):
            cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50))
            cloud.normalize_normals()

        if i == len(scales) - 1:
            finest_source_normals = np.asarray(source.normals, dtype=np.float64)
            finest_target_normals = np.asarray(target.normals, dtype=np.float64)

        # 自适应对应距离：细层级收紧
        distance = (min(float(max_correspondence_distance), max(scale * 2.5, 0.015))
                    if max_correspondence_distance is not None
                    else max(scale * 2.5, 0.015))

        # 所有层级使用点到面（立面场景平面约束更强）
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()

        iters = adaptive_iters[i]
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iters)
        final = o3d.pipelines.registration.registration_icp(
            source, target, distance, T, estimation, criteria)
        T = np.asarray(final.transformation)

        # 每层级后施加Z轴阻尼回归（软约束，非硬截断）
        if z_lock:
            T = _apply_z_damping(T, z_damping=z_damping, max_z_shift=max_z_shift)

        dt_level = time.perf_counter() - t_level
        level_times.append(round(dt_level, 3))
        reports.append({
            'voxel': scale,
            'fitness': float(final.fitness),
            'rmse': float(final.inlier_rmse),
            'correspondences': len(final.correspondence_set),
            'duration_s': round(dt_level, 3),
            'distance_threshold': distance,
            'iterations': iters,
        })
        print(f'[REG-ICP] 层级{i+1}/{len(scales)} voxel={scale:.3f}m '
              f'fitness={final.fitness:.4f} rmse={final.inlier_rmse:.4f} '
              f'corr={len(final.correspondence_set)} iter={iters} time={dt_level:.3f}s', flush=True)

    # 使用最细层级下采样点云计算指标（非原始点云）
    metric_distance = (float(max_correspondence_distance)
                       if max_correspondence_distance is not None
                       else max(scales[-1] * 2.5, 0.01))

    # 大点云时跳过法向量一致性计算以加速
    compute_normals = min_pts <= 500000
    checked = registration_metrics(
        finest_source_pts, finest_target_pts, T,
        max_correspondence_distance=metric_distance,
        source_normals=finest_source_normals,
        target_normals=finest_target_normals,
        compute_normals=compute_normals,
    )

    fitness = checked['fitness']
    rmse = checked['rmse']
    correspondence_count = checked['correspondence_count']
    normal_consistency = checked['normal_consistency']

    min_correspondences = max(3, min(50, int(np.ceil(min(len(src), len(tgt)) * 0.005))))

    # 接受条件：拟合度 + 对应点数（法向量一致性改为警告而非拒绝）
    accepted = (fitness >= 0.01 and correspondence_count >= min_correspondences
                and np.isfinite(rmse))
    message = '' if accepted else '有效对应不足或配准质量未达到门限'

    # 法向量一致性：立面场景法向量主要在XY平面，Z分量小，区分度有限
    # 改为警告而非拒绝条件，阈值放宽到-0.5
    if normal_consistency < -0.5:
        warnings.append(f'法向量一致性偏低({normal_consistency:.3f})，建议人工检查')

    # Z轴偏差：改为警告而非拒绝，阈值与max_z_shift一致
    z_shift = float(T[2, 3])
    if abs(z_shift) > max_z_shift:
        warnings.append(f'Z轴平移偏差{z_shift:.3f}m，超出典型GPS高程精度范围')
    elif abs(z_shift) > max_z_shift * 0.5:
        warnings.append(f'Z轴平移偏差{z_shift:.3f}m，GPS高程残余修正量较大')

    reports.append(checked)

    total_s = time.perf_counter() - t_total
    timing = {
        'total_s': round(total_s, 3),
        'level_times_s': level_times,
        'metrics_s': round(total_s - sum(level_times), 3),
    }

    return ICPResult(T, fitness, rmse, correspondence_count, reports, accepted,
                     message, timing, normal_consistency, warnings)


# =============================================================================
# 4. 全局精修（多站点全局优化）
# =============================================================================

def global_refinement(clouds, transforms, voxel_size=0.05,
                       max_correspondence_distance=0.15,
                       max_iteration=20,
                       z_lock=True, max_z_shift=0.15, z_damping=0.3):
    """多站点配准后的全局精修。

    两两配准完成后，所有点云已变换到参考坐标系。
    本函数执行最终全局ICP：每朵点云同时对所有其他站点的合并点云进行精修。
    同时保持Z轴软约束，不破坏GPS高程对齐。

    参数：
        clouds: 已在参考坐标系中的RegistrationCloud列表（已变换）
        transforms: 每朵点云应用的4x4变换列表（参考=单位阵）
        voxel_size: 精修体素大小
        max_correspondence_distance: 对应点阈值
        max_iteration: ICP最大迭代次数
        z_lock: 是否启用Z轴软约束
        max_z_shift: Z平移最大允许修正量
        z_damping: Z轴阻尼系数

    返回：
        refined_transforms: 精修后的4x4变换列表
        log_summary: 全局精修摘要日志
    """
    t0 = time.perf_counter()
    if len(clouds) < 2:
        return transforms, {'message': '单站点无需全局精修'}

    refined = [np.asarray(t, dtype=np.float64).reshape(4, 4) for t in transforms]

    # 迭代全局精修（通常1-2轮即可收敛）
    for round_idx in range(2):
        round_changes = []
        for i, cloud in enumerate(clouds):
            if i == 0:
                continue  # 参考站点固定

            # 从所有其他站点构建目标点云（已在参考坐标系中）
            target_parts = []
            for j, other in enumerate(clouds):
                if j == i:
                    continue
                pts = np.asarray(other.points, dtype=np.float64)
                moved = pts @ refined[j][:3, :3].T + refined[j][:3, 3]
                target_parts.append(moved)

            if not target_parts:
                continue
            target_all = np.vstack(target_parts)

            # 当前站点在参考坐标系中的位置
            source_pts = np.asarray(cloud.points, dtype=np.float64)
            source_moved = source_pts @ refined[i][:3, :3].T + refined[i][:3, 3]

            # 快速ICP精修（Z轴软约束）
            result = point_to_plane_icp(
                source_moved, target_all,
                init=np.eye(4),
                voxel_size=voxel_size,
                max_correspondence_distance=max_correspondence_distance,
                max_iteration=max_iteration,
                pyramid_scales=(1.0,),
                z_lock=z_lock,
                max_z_shift=max_z_shift,
                z_damping=z_damping,
            )

            if result.accepted:
                delta = np.asarray(result.transformation)
                refined[i] = delta @ refined[i]
                # 每次更新后施加Z轴阻尼
                if z_lock:
                    refined[i] = _apply_z_damping(refined[i], z_damping=z_damping, max_z_shift=max_z_shift)
                # 度量变化量
                angle = np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1))
                trans_norm = np.linalg.norm(delta[:3, 3])
                round_changes.append({
                    'station_idx': i,
                    'angle_deg': round(np.degrees(angle), 4),
                    'translation': round(trans_norm, 6),
                    'rmse': round(result.inlier_rmse, 6),
                })

    # 计算全局重叠指标
    all_merged = []
    for i, cloud in enumerate(clouds):
        pts = np.asarray(cloud.points, dtype=np.float64)
        moved = pts @ refined[i][:3, :3].T + refined[i][:3, 3]
        all_merged.append(moved)
    merged = np.vstack(all_merged)

    log_summary = {
        'phase': 'global_refinement',
        'rounds': round_idx + 1,
        'station_changes': round_changes,
        'merged_point_count': len(merged),
    }

    return refined, log_summary