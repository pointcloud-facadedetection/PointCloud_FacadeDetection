from typing import List, Optional, Tuple
import numpy as np
import open3d as o3d
import copy
import cv2
from collections import deque


# ==================== 距离自适应几何工具（原 range_adaptive.py）====================

def estimate_point_ranges(points, scan_origin=None):
    """计算每个点到最近测站的距离。"""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    origins = np.zeros((1, 3), dtype=float) if scan_origin is None else np.asarray(scan_origin, dtype=float).reshape(-1, 3)
    if not len(pts):
        return np.empty(0, dtype=np.float32)
    return np.min(np.stack([np.linalg.norm(pts - o, axis=1) for o in origins], axis=1), axis=1)


def estimate_elevation_angles(points, scan_origin=None):
    """计算每个点相对测站的高度角（elevation），用于高度方向密度补偿。"""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    origins = np.zeros((1, 3), dtype=float) if scan_origin is None else np.asarray(scan_origin, dtype=float).reshape(-1, 3)
    if not len(pts):
        return np.empty(0, dtype=np.float32)
    ranges = np.min(np.stack([np.linalg.norm(pts - o, axis=1) for o in origins], axis=1), axis=1)
    delta = pts[:, None, :] - origins[None, :, :]
    nearest = np.argmin(np.linalg.norm(delta, axis=2), axis=1)
    dz = pts[:, 2] - origins[nearest, 2]
    horizontal = np.sqrt(np.maximum(ranges ** 2 - dz ** 2, 0.0))
    elevation = np.degrees(np.arctan2(dz, horizontal + 1e-9))
    return elevation.astype(np.float32)


def elevation_scale_factor(elevation, low_scale=1.0, high_scale=0.55, threshold_deg=40.0):
    """高度角越大（高处），扫描线越稀疏，体素应缩小。"""
    elevation = np.asarray(elevation, dtype=float)
    t = np.clip((np.abs(elevation) - threshold_deg) / max(90.0 - threshold_deg, 1e-6), 0.0, 1.0)
    return low_scale + (high_scale - low_scale) * t


def adaptive_plane_tolerance(ranges, base_tol, range_coeff=.0012, max_tol=None):
    """根据距离自适应调整平面容差。"""
    tol = float(base_tol) + float(range_coeff) * np.asarray(ranges, dtype=float)
    return np.minimum(tol, max_tol) if max_tol is not None else tol


def fit_plane_weighted(points, weights=None, irls_iters=0):
    """加权/鲁棒 IRLS 平面拟合。"""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) < 3:
        return np.array([0., 0., 1., 0.])
    w = np.ones(len(pts)) if weights is None else np.maximum(np.asarray(weights, dtype=float), 1e-12)
    model = None
    for iteration in range(max(1, int(irls_iters) + 1)):
        center = np.sum(pts * w[:, None], axis=0) / np.sum(w)
        _, _, vh = np.linalg.svd((pts - center) * np.sqrt(w[:, None]), full_matrices=False)
        n = vh[-1] / (np.linalg.norm(vh[-1]) + 1e-12)
        d = -float(n @ center)
        model = np.r_[n, d]
        if iteration >= int(irls_iters):
            break
        residual = np.abs(pts @ n + d)
        scale = 1.4826 * float(np.median(residual)) + 1e-9
        k = 1.345 * scale
        w = (np.ones(len(pts)) if weights is None else np.maximum(np.asarray(weights, dtype=float), 1e-12)) * np.where(residual <= k, 1., k / np.maximum(residual, 1e-12))
    return model


def adaptive_outlier_indices(points, ranges, std_ratio=2.5, n_shells=24):
    """按距离分壳统计滤波，保留非离群点索引。"""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    ranges = np.asarray(ranges, dtype=float).reshape(-1)
    n = len(pts)
    if n < 2:
        return np.arange(n, dtype=np.int32)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    nn = np.asarray(cloud.compute_nearest_neighbor_distance())
    keep = np.ones(n, dtype=bool)
    edges = np.quantile(ranges, np.linspace(0, 1, max(1, int(n_shells)) + 1))
    for i in range(len(edges) - 1):
        mask = (ranges >= edges[i]) & ((ranges <= edges[i + 1]) if i == len(edges)-2 else (ranges < edges[i+1]))
        if mask.sum() < 30:
            continue
        d = nn[mask]
        med = float(np.median(d)); mad = 1.4826 * float(np.median(np.abs(d-med)))
        threshold = med + std_ratio * max(mad, .15 * med, 1e-6)
        keep[np.where(mask)[0][d > threshold]] = False
    return np.flatnonzero(keep).astype(np.int32)


def stratified_downsample(points, colors, ranges, source_ids=None,
                          shells=((8., .10), (16., .08), (28., .06),
                                  (45., .05), (70., .045), (100., .04)),
                          min_range=.5, crop_range=120.,
                          elevations=None, scan_origin=None,
                          min_voxel=0.02, max_voxel=0.20,
                          elevation_low_scale=1.0, elevation_high_scale=0.55,
                          elevation_threshold_deg=40.0):
    """按距离层选体素代表点，并返回代表点对应的源行号。"""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    rng = np.asarray(ranges, dtype=np.float32).reshape(-1)
    if len(pts) != len(rng):
        raise ValueError("points/ranges 长度不一致")
    src = np.arange(len(pts), dtype=np.int32) if source_ids is None else np.asarray(source_ids, dtype=np.int32).reshape(-1)
    if len(src) != len(pts):
        raise ValueError("points/source_ids 长度不一致")
    col = None if colors is None else np.asarray(colors, dtype=np.float32).reshape(-1, 3)

    if elevations is None and scan_origin is not None:
        elevations = estimate_elevation_angles(pts, scan_origin)
    elev = np.asarray(elevations, dtype=np.float32).reshape(-1) if elevations is not None else np.zeros(len(pts), dtype=np.float32)
    if len(elev) != len(pts):
        raise ValueError("points/elevations 长度不一致")

    selected_p, selected_c, selected_s, selected_r = [], [], [], []
    lo = float(min_range)
    bounds = list(shells) + [(float("inf"), None)]
    for hi, base_voxel in bounds:
        mask = (rng >= lo) & (rng < min(float(hi), crop_range + 1e-9)) & (rng <= crop_range)
        ids = np.flatnonzero(mask)
        if len(ids):
            if base_voxel:
                scale = elevation_scale_factor(
                    elev[ids], elevation_low_scale, elevation_high_scale, elevation_threshold_deg)
                per_voxel = np.clip(base_voxel * scale, min_voxel, max_voxel)
                keys = np.floor(pts[ids] / per_voxel[:, None]).astype(np.int64)
                _, first = np.unique(keys, axis=0, return_index=True)
                ids = ids[np.sort(first)]
            selected_p.append(pts[ids]); selected_s.append(src[ids]); selected_r.append(rng[ids])
            if col is not None:
                selected_c.append(col[ids])
        lo = float(hi)
        if lo >= crop_range:
            break
    if not selected_p:
        empty = np.empty((0, 3), np.float32)
        return empty, (empty.copy() if col is not None else None), np.empty(0, np.int32), np.empty(0, np.float32)
    return np.vstack(selected_p), (np.vstack(selected_c) if col is not None else None), np.concatenate(selected_s), np.concatenate(selected_r)


# ==================== 立面检测重构版新增工具 ====================

def deduplicate_clusters(clusters: List[np.ndarray]) -> List[np.ndarray]:
    """
    基于集合包含关系去重：若 cluster A 被 B 严格包含，则丢弃 A。
    用于多尺度 3D 连通域结果融合。
    """
    if not clusters:
        return []
    sets = [set(c.tolist()) for c in clusters]
    n = len(clusters)
    discarded = [False] * n
    for i in range(n):
        if discarded[i]:
            continue
        si = sets[i]
        for j in range(i + 1, n):
            if discarded[j]:
                continue
            sj = sets[j]
            if si.issubset(sj) and len(si) < len(sj) * 0.95:
                discarded[i] = True
                break
            elif sj.issubset(si) and len(sj) < len(si) * 0.95:
                discarded[j] = True
    return [clusters[i] for i in range(n) if not discarded[i]]


def filter_clusters_by_planarity(clusters: List[np.ndarray], points: np.ndarray,
                                 min_ratio: float = 0.15) -> List[np.ndarray]:
    """
    用 PCA 特征值过滤非平面簇（树木/植被）。
    保留 λ₃/λ₁ < min_ratio 的簇。
    """
    valid = []
    for c in clusters:
        pts = points[c]
        if len(pts) < 10:
            continue
        centered = pts - np.mean(pts, axis=0)
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        s = np.sort(s)[::-1]
        if s[0] > 1e-6 and (s[2] / s[0] < min_ratio):
            valid.append(c)
    return valid

def fit_plane_svd(points):
    """最小二乘 SVD 拟合平面"""
    if len(points) < 3:
        return np.array([0, 0, 1, 0], dtype=float)
    center = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    d = -float(np.dot(normal, center))
    return np.array([normal[0], normal[1], normal[2], d], dtype=float)


def _weighted_svd_plane(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """基于加权协方差的SVD进行加权最小二乘平面拟合。
    返回带单位法向量的平面模型 [nx, ny, nz, d]。
    """
    pts = np.asarray(points, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if len(pts) < 3 or len(w) != len(pts):
        return fit_plane_svd(pts)
    w = np.clip(w, 1e-8, None)
    W = w / (np.sum(w) + 1e-12)
    center = np.sum(pts * W[:, None], axis=0)
    X = (pts - center) * np.sqrt(W)[:, None]
    if len(pts) >= 3:
        _, _, vh = np.linalg.svd(X, full_matrices=False)
        normal = vh[-1]
    else:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    d = -float(np.dot(normal, center))
    return np.array([normal[0], normal[1], normal[2], d], dtype=float)


def fit_plane_irls(points: np.ndarray, init_model: np.ndarray | None = None,
                   max_iters: int = 3, huber_delta: float = 0.05) -> np.ndarray:
    """通过IRLS并采用Huber权重进行鲁棒平面拟合。
    - points: Nx3
    - init_model: 可选参数 [nx, ny, nz, d]；若为 None，则使用奇异值分解（SVD）
    - huber_delta: Huber损失的阈值（单位：米）
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return fit_plane_svd(pts)
    model = np.asarray(init_model, dtype=float) if init_model is not None else fit_plane_svd(pts)
    for _ in range(int(max_iters)):
        n = model[:3]
        n = n / (np.linalg.norm(n) + 1e-12)
        d = float(model[3])
        r = pts @ n + d  # signed distances
        abs_r = np.abs(r)
        # Huber weights
        w = np.ones_like(r)
        mask = abs_r > huber_delta
        if np.any(mask):
            w[mask] = huber_delta / (abs_r[mask] + 1e-12)
        model = _weighted_svd_plane(pts, w)
    return model

def statistical_outlier_mask(points: np.ndarray, k: int = 20, std_ratio: float = 2.0):
    """统计滤波掩码：返回非离群点的布尔掩码"""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < k + 1:
        return np.ones(len(points), dtype=bool)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    try:
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=k, std_ratio=std_ratio)
        mask = np.zeros(len(points), dtype=bool)
        mask[ind] = True
        return mask
    except Exception:
        return np.ones(len(points), dtype=bool)


def split_by_depth_histogram(points, plane_model, max_plane_dist, min_points=80):
    """沿法向深度方向检测多峰结构并切分。
    
    建筑立面常有阳台/雨棚凸出（深度突变），本函数通过有符号距离直方图
    检测谷值，将主墙面与凸出/凹进结构分离。
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < min_points * 2:
        return [np.ones(len(points), dtype=bool)]

    normal = np.asarray(plane_model[:3], dtype=float)
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    d = float(plane_model[3])
    signed = points @ normal + d

    # 直方图
    n_bins = max(12, int(np.sqrt(len(points)) / 2))
    counts, edges = np.histogram(signed, bins=n_bins)

    # 找显著谷值：局部最小，且两侧峰值远高于谷值
    valleys = []
    for i in range(2, len(counts) - 2):
        if counts[i] < counts[i-1] and counts[i] < counts[i+1]:
            left_peak = max(counts[:i])
            right_peak = max(counts[i+1:])
            if left_peak > counts[i] * 2.5 and right_peak > counts[i] * 2.5:
                # 验证两侧中心距是否显著
                left_mask = signed <= edges[i+1]
                right_mask = signed > edges[i+1]
                if np.sum(left_mask) >= min_points and np.sum(right_mask) >= min_points:
                    left_c = np.median(signed[left_mask])
                    right_c = np.median(signed[right_mask])
                    if abs(right_c - left_c) > max_plane_dist * 1.2:
                        valleys.append(edges[i+1])

    if not valleys:
        return [np.ones(len(points), dtype=bool)]

    # 按最深谷值切分（简化：只切一次，避免过度分割）
    split_val = valleys[0]
    m1 = signed <= split_val
    m2 = signed > split_val
    masks = []
    if np.sum(m1) >= min_points:
        masks.append(m1)
    if np.sum(m2) >= min_points:
        masks.append(m2)
    return masks if masks else [np.ones(len(points), dtype=bool)]

def cluster_normals_direction(normals, angle_threshold_deg=5.0):
    """基于球面网格哈希的法向量方向预聚类。
    
    将法向量映射到球面经纬度网格，按方向一致性聚类。
    返回按点数降序排列的布尔掩码列表。
    时间复杂度 O(n)，远优于随机RANSAC采样。
    """
    normals = np.asarray(normals, dtype=np.float64)
    # 法向是无向量：统一半球，避免扫描法向朝向相反造成簇碎裂。
    signs = np.where(normals[:, 2:3] < 0, -1.0, 1.0)
    tie = (np.abs(normals[:, 2:3]) < 1e-8) & (normals[:, 0:1] < 0)
    signs[tie] = -1.0
    normals = normals * signs
    norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    normals = normals / norms
    n = len(normals)
    if n == 0:
        return []

    # 球面坐标量化
    n_bins = max(3, int(180.0 / max(angle_threshold_deg, 1.0)))
    x, y, z = normals[:, 0], normals[:, 1], normals[:, 2]
    theta = np.arctan2(y, x)                # [-pi, pi]
    phi = np.arccos(np.clip(z, -1.0, 1.0))  # [0, pi]

    theta_bin = np.floor((theta + np.pi) / (2 * np.pi) * n_bins).astype(np.int32)
    phi_bin = np.floor(phi / np.pi * n_bins).astype(np.int32)
    theta_bin = np.clip(theta_bin, 0, n_bins - 1)
    phi_bin = np.clip(phi_bin, 0, n_bins - 1)

    grid_map = {}
    for i in range(n):
        key = (int(theta_bin[i]), int(phi_bin[i]))
        grid_map.setdefault(key, []).append(i)

    # 按网格点数降序，优先大方向（主立面）
    sorted_grids = sorted(grid_map.items(), key=lambda x: len(x[1]), reverse=True)
    clusters = []
    used = np.zeros(n, dtype=bool)
    cos_thr = np.cos(np.deg2rad(angle_threshold_deg))

    for key, indices in sorted_grids:
        indices = np.asarray(indices, dtype=int)
        available = indices[~used[indices]]
        if len(available) == 0:
            continue

        seed_idx = available[0]
        seed_normal = normals[seed_idx]

        # 在 3x3 邻域网格内搜索（球面网格邻居）
        t, p = key
        neighbor_keys = [(t + dt, p + dp)
                         for dt in (-1, 0, 1) for dp in (-1, 0, 1)]
        neighbor_indices = []
        for nk in neighbor_keys:
            if nk in grid_map:
                neighbor_indices.extend(grid_map[nk])
        neighbor_indices = np.asarray(neighbor_indices, dtype=int)
        neighbor_indices = neighbor_indices[~used[neighbor_indices]]
        if len(neighbor_indices) == 0:
            continue

        sim = normals[neighbor_indices] @ seed_normal
        matched = neighbor_indices[sim >= cos_thr]
        if len(matched) > 0:
            mask = np.zeros(n, dtype=bool)
            mask[matched] = True
            clusters.append(mask)
            used[matched] = True

    return clusters


def classify_plane(normal):
    """按法向与重力方向关系分类"""
    normal = np.asarray(normal, dtype=float)
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    nz = abs(float(normal[2]))
    verticality = float(np.degrees(np.arcsin(np.clip(nz, 0.0, 1.0))))
    horizontality = float(np.degrees(np.arccos(np.clip(nz, 0.0, 1.0))))

    if nz <= 0.25:
        return 'vertical_facade', '立面', verticality, horizontality
    if nz >= 0.85:
        return 'horizontal', '水平面', verticality, horizontality
    return 'inclined', '倾斜面', verticality, horizontality


def plane_axes(normal, facade_type=None):
    """为平面构造稳定的局部坐标轴"""
    normal = np.asarray(normal, dtype=float)
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    z_axis = np.array([0.0, 0.0, 1.0])

    if facade_type == 'vertical_facade' or abs(normal[2]) < 0.45:
        u = np.cross(z_axis, normal)
        if np.linalg.norm(u) < 1e-8:
            u = np.array([1.0, 0.0, 0.0])
        u = u / (np.linalg.norm(u) + 1e-12)
        v = z_axis
    else:
        ref = z_axis if abs(np.dot(normal, z_axis)) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(normal, ref)
        u = u / (np.linalg.norm(u) + 1e-12)
        v = np.cross(normal, u)
        v = v / (np.linalg.norm(v) + 1e-12)
    return u, v


def estimate_plane_area(points, normal, center, facade_type=None):
    """在平面局部坐标内估计面积"""
    if len(points) < 3:
        return 0.0, None
    u, v = plane_axes(normal, facade_type)
    local = np.column_stack([np.dot(points - center, u), np.dot(points - center, v)])
    min_u, min_v = np.min(local, axis=0)
    max_u, max_v = np.max(local, axis=0)
    area = 0.0
    try:
        lifted = np.column_stack([local, np.zeros(len(local))])
        hull_pcd = o3d.geometry.PointCloud()
        hull_pcd.points = o3d.utility.Vector3dVector(lifted)
        hull, _ = hull_pcd.compute_convex_hull()
        area = float(hull.get_surface_area() * 0.5)
    except Exception:
        area = float(max((max_u - min_u) * (max_v - min_v), 0.0))

    bbox_2d = {
        'u_min': float(min_u), 'u_max': float(max_u),
        'v_min': float(min_v), 'v_max': float(max_v),
        'u_axis': [float(x) for x in u],
        'v_axis': [float(x) for x in v]
    }
    return area, bbox_2d


def plane_distance(points, plane_model):
    """计算点到平面的距离"""
    model = np.asarray(plane_model, dtype=float)
    normal = model[:3]
    norm = np.linalg.norm(normal) + 1e-12
    return np.abs(points @ normal + model[3]) / norm


def signed_plane_distance(points, plane_model):
    """计算点到平面的有符号距离"""
    model = np.asarray(plane_model, dtype=float)
    normal = model[:3]
    norm = np.linalg.norm(normal) + 1e-12
    return (points @ model[:3] + model[3]) / norm


def vertical_deviation_from_normal(normal, measure_height=2.0):
    """计算垂直度相关指标"""
    normal = np.asarray(normal, dtype=float)
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    nz = abs(normal[2])

    verticality_deg = float(np.degrees(np.arcsin(np.clip(nz, 0.0, 1.0))))
    if verticality_deg < 89.9:
        deviation_m = float(measure_height * np.tan(np.deg2rad(verticality_deg)))
    else:
        deviation_m = float('inf')

    return verticality_deg, deviation_m, deviation_m * 1000.0


def is_uniform_color(colors, eps=1e-4):
    if len(colors) == 0:
        return True
    r, g, b = colors[0]
    for i in range(1, len(colors)):
        if abs(colors[i][0] - r) > eps or abs(colors[i][1] - g) > eps or abs(colors[i][2] - b) > eps:
            return False
    return True


def ensure_normals(pcd, voxel_size=0.05, inplace=False):
    """ 估计并规范化点云法向量 """
    if len(pcd.points) == 0:
        return pcd
    # 已有与点数一致的有限法向时直接复用，避免 Windows 下重复 KD-tree
    # 估计；法向不足或非法时才走原有估计流程。
    if pcd.has_normals() and len(pcd.normals) == len(pcd.points):
        existing = np.asarray(pcd.normals)
        if existing.ndim == 2 and existing.shape == (len(pcd.points), 3) and np.all(np.isfinite(existing)):
            norms = np.linalg.norm(existing, axis=1)
            if np.all(norms > 1e-8):
                pcd_work = pcd if inplace else copy.deepcopy(pcd)
                pcd_work.normals = o3d.utility.Vector3dVector(existing / norms[:, None])
                return pcd_work
    # 避免无条件 deepcopy
    pcd_work = pcd if inplace else copy.deepcopy(pcd)
    radius = max(float(voxel_size) * 4.0, 0.2)
    pcd_work.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50)
    )
    try:
        pcd_work.orient_normals_consistent_tangent_plane(30)
    except Exception:
        pcd_work.orient_normals_towards_camera_location([0, 0, 0])
    return pcd_work


def project_to_uv(points, plane_model, facade_type=None):
    """将3D点投影到平面局部UV坐标系，与 backend 实现保持一致。"""
    normal = np.asarray(plane_model[:3], dtype=float)
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    center = np.mean(points, axis=0)
    u_axis, v_axis = plane_axes(normal, facade_type)

    centered = points - center
    u_coords = centered @ u_axis
    v_coords = centered @ v_axis

    return np.column_stack([u_coords, v_coords]), u_axis, v_axis, center


def connected_components_2d_grid(uv_points, grid_size=None, min_cells=3,
                                 adaptive_ratio=2.5, sample_ratio=0.1,
                                 connectivity: int = 8,
                                 close_radius_cells: int = 0):
    """
    基于UV网格的连通域分析（4/8邻域）。
    
    【关键参数】close_radius_cells: 在BFS搜索时扩展邻域半径，
    允许跨越小缝隙（如门窗洞口），保持立面完整性。
    """
    if len(uv_points) == 0:
        return []

    if grid_size is None:
        n_sample = max(20, int(len(uv_points) * sample_ratio))
        if len(uv_points) > n_sample:
            indices = np.random.choice(len(uv_points), n_sample, replace=False)
            sample_pts = uv_points[indices]
        else:
            sample_pts = uv_points

        sample_pcd = o3d.geometry.PointCloud()
        sample_pcd.points = o3d.utility.Vector3dVector(
            np.column_stack([sample_pts, np.zeros(len(sample_pts))])
        )
        tree = o3d.geometry.KDTreeFlann(sample_pcd)

        avg_nn_dist = 0.0
        valid_count = 0
        for i in range(len(sample_pts)):
            k, _idx, dists = tree.search_knn_vector_3d(sample_pcd.points[i], 2)
            if k >= 2:
                avg_nn_dist += np.sqrt(dists[1])
                valid_count += 1
        avg_nn_dist = avg_nn_dist / valid_count if valid_count > 0 else 0.1
        grid_size = max(avg_nn_dist * adaptive_ratio, 0.02)

    u_min, v_min = np.min(uv_points, axis=0)
    u_max, v_max = np.max(uv_points, axis=0)
    if u_max - u_min < 1e-6 and v_max - v_min < 1e-6:
        return [np.ones(len(uv_points), dtype=bool)]

    grid_u = np.floor((uv_points[:, 0] - u_min) / grid_size).astype(np.int32)
    grid_v = np.floor((uv_points[:, 1] - v_min) / grid_size).astype(np.int32)

    grid_to_indices = {}
    for i, (gu, gv) in enumerate(zip(grid_u, grid_v)):
        grid_to_indices.setdefault((gu, gv), []).append(i)

    if not grid_to_indices:
        return []

    visited = set()
    components = []
    
    # 【关键】根据 close_radius_cells 生成邻域偏移
    rng = range(-max(1, close_radius_cells), max(1, close_radius_cells) + 1)
    if connectivity == 8:
        offsets = [(du, dv) for du in rng for dv in rng if not (du == 0 and dv == 0)]
    else:
        offsets = [(du, 0) for du in rng if du != 0] + [(0, dv) for dv in rng if dv != 0]

    for key in grid_to_indices:
        if key in visited:
            continue

        queue = deque([key])
        visited.add(key)
        cell_cluster = [key]

        while queue:
            cu, cv = queue.popleft()
            for du, dv in offsets:
                nb = (cu + du, cv + dv)
                if nb in grid_to_indices and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
                    cell_cluster.append(nb)

        if len(cell_cluster) < min_cells:
            continue

        point_mask = np.zeros(len(uv_points), dtype=bool)
        for cell in cell_cluster:
            for idx in grid_to_indices[cell]:
                point_mask[idx] = True
        components.append(point_mask)

    return components

def connected_components_3d_grid(points, grid_size, min_points=100):
    """基于3D体素网格的26邻域连通域分析。
    
    只有空间上体素相邻的点才会被归为同一组件，
    从根本上避免"法向量相同但空间不相邻"的平行面被误合并。
    """
    if len(points) == 0:
        return []
    points = np.asarray(points, dtype=np.float64)
    min_coords = np.min(points, axis=0)
    gs = max(grid_size, 1e-6)
    grid_coords = np.floor((points - min_coords) / gs).astype(np.int32)

    grid_map = {}
    for i, gc in enumerate(grid_coords):
        key = tuple(gc)
        grid_map.setdefault(key, []).append(i)

    if not grid_map:
        return []

    visited = set()
    components = []
    # 26邻域
    neighbors = [(dx, dy, dz)
                 for dx in (-1, 0, 1)
                 for dy in (-1, 0, 1)
                 for dz in (-1, 0, 1)
                 if not (dx == 0 and dy == 0 and dz == 0)]

    for key in grid_map:
        if key in visited:
            continue

        queue = deque([key])
        visited.add(key)
        cell_cluster = [key]

        while queue:
            ck = queue.popleft()
            for dn in neighbors:
                nb = (ck[0] + dn[0], ck[1] + dn[1], ck[2] + dn[2])
                if nb in grid_map and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
                    cell_cluster.append(nb)

        indices = []
        for ck in cell_cluster:
            indices.extend(grid_map[ck])

        if len(indices) >= min_points:
            mask = np.zeros(len(points), dtype=bool)
            mask[indices] = True
            components.append(mask)

    return components

def plane_basis_from_normal(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-12)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    return u, v


def project_to_plane(points: np.ndarray, origin: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    rel = points - origin.reshape(1, 3)
    return np.column_stack([rel @ u, rel @ v])


def estimate_uv_bbox_area(uv: np.ndarray) -> float:
    """UV投影bbox面积，用于快速排除碎片"""
    if len(uv) < 3:
        return 0.0
    umin, vmin = np.min(uv, axis=0)
    umax, vmax = np.max(uv, axis=0)
    return max((umax - umin) * (vmax - vmin), 0.0)


def uv_bbox_distance(f1: dict, f2: dict) -> float:
    """两个立面在UV投影上的bbox距离，法向必须近似平行"""
    n1 = np.asarray(f1.get('normal') or f1['plane_model'][:3])
    n2 = np.asarray(f2.get('normal') or f2['plane_model'][:3])
    if abs(float(np.dot(n1, n2))) < 0.996:
        return float('inf')
    b1 = f1.get('bbox_2d')
    b2 = f2.get('bbox_2d')
    if b1 is None or b2 is None:
        return float('inf')
    du = max(b1['u_min'] - b2['u_max'], b2['u_min'] - b1['u_max'], 0.0)
    dv = max(b1['v_min'] - b2['v_max'], b2['v_min'] - b1['v_max'], 0.0)
    return np.sqrt(du**2 + dv**2)