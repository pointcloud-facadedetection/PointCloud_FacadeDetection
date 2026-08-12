import numpy as np
import open3d as o3d

def to_zup(p_yup):
    """前端 Y-up (x,y,z) -> 后端 Z-up (x, -z, y)"""
    return np.array([p_yup[0], -p_yup[2], p_yup[1]])

def pcd_to_json(pcd):
    """将Open3D点云转换为JSON格式（Z-up坐标系）"""
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else np.ones_like(points) * 0.7
    normals = np.asarray(pcd.normals, dtype=np.float32) if pcd.has_normals() else np.zeros_like(points)
    return {
        'positions': points.flatten().tolist(),
        'colors': colors.flatten().tolist(),
        'normals': normals.flatten().tolist()
    }

def fit_plane_svd(points):
    """最小二乘SVD拟合平面"""
    if len(points) < 3:
        return np.array([0, 0, 1, 0], dtype=float)
    center = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    d = -float(np.dot(normal, center))
    return np.array([normal[0], normal[1], normal[2], d], dtype=float)


def estimate_point_ranges(points, scan_origin=None):
    """
    点到扫描仪的距离。单站扫描默认扫描仪位于坐标原点。

    scan_origin 可为单个测站 (3,) 或多测站 (K,3)（配准合并点云）；
    多测站时取每点到最近测站的距离。
    """
    pts = np.asarray(points, dtype=float)
    origin = np.zeros(3) if scan_origin is None else np.asarray(scan_origin, dtype=float)
    if origin.ndim == 2:
        dists = np.stack([np.linalg.norm(pts - o, axis=1) for o in origin], axis=1)
        return np.min(dists, axis=1)
    return np.linalg.norm(pts - origin, axis=1)


def resolve_scan_origin(points, scan_origin=None, cell=1.0, tripod_height=1.7):
    """
    确定扫描仪位置。显式传入优先；否则默认坐标原点（扫描仪坐标系数据）。

    若最近点距原点 >10m，说明点云在世界/项目坐标系（如 SCENE 配准导出），
    此时用最密体素启发式估计测站位置：单站扫描中扫描仪脚下的地面环密度
    最高（密度 ∝ 1/r²）。估计误差在米级，对 τ(r) 距离缩放已足够；
    生产环境建议从 E57/FLS 元数据传入真实测站坐标。
    """
    if scan_origin is not None:
        return np.asarray(scan_origin, dtype=float)
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return np.zeros(3)
    origin = np.zeros(3)
    if float(np.min(np.linalg.norm(pts, axis=1))) <= 10.0:
        return origin

    cells = np.floor(pts / float(cell)).astype(np.int64)
    _, inverse, counts = np.unique(cells, axis=0, return_inverse=True,
                                   return_counts=True)
    densest = int(np.argmax(counts))
    patch = pts[inverse == densest]
    est = np.median(patch, axis=0)
    est[2] = float(np.percentile(patch[:, 2], 5.0)) + float(tripod_height)
    return est


def range_noise_sigma(ranges, sigma0=0.0015, sigma_slope=0.00025):
    """TLS 测距噪声模型 σ(r) = σ0 + k·r（单位米）。"""
    return sigma0 + sigma_slope * np.asarray(ranges, dtype=float)


def adaptive_plane_tolerance(ranges, base_tol, range_coeff=0.0012, max_tol=None):
    """距离自适应平面距离容差 τ(r) = base + k·r，远处点噪声大需放宽。"""
    tol = base_tol + range_coeff * np.asarray(ranges, dtype=float)
    if max_tol is not None:
        tol = np.minimum(tol, max_tol)
    return tol


def range_stratified_downsample(pcd, scan_origin=None,
                                shells=((15.0, 0.08), (30.0, 0.06),
                                        (50.0, 0.045), (75.0, 0.035))):
    """
    按距离分层下采样：近处过采样区用大体素狠采，远处稀疏区少采/不采。

    均匀体素下采样会把近处削到目标密度、远处本来就稀疏一个不动，导致
    总点数仍被近场主导——算法里所有"按总点数比例"的阈值
    （min_cluster_points、min_remaining 等）都被近场撑大，远处目标
    永远过不了线。分层采样把各距离段密度拉平后这些阈值才公平。

    shells: ((上限米, 体素米), ...) 按距离升序；超出最后一档的点保留原始密度。
    """
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return pcd
    ranges = estimate_point_ranges(pts, resolve_scan_origin(pts, scan_origin))
    parts = []
    lo = 0.0
    for hi, vox in shells:
        m = (ranges >= lo) & (ranges < hi)
        if m.any():
            sub = pcd.select_by_index(np.where(m)[0].tolist())
            parts.append(sub.voxel_down_sample(float(vox)) if vox else sub)
        lo = hi
    m = ranges >= lo
    if m.any():
        parts.append(pcd.select_by_index(np.where(m)[0].tolist()))
    if not parts:
        return pcd
    out = parts[0]
    for p in parts[1:]:
        out += p
    return out


def adaptive_outlier_mask(pcd, scan_origin=None, std_ratio=2.5, n_shells=24):
    """
    距离分壳离群点掩码：按到扫描仪距离分壳，壳内按最近邻距离做稳健统计判定。

    固定半径滤波/全局统计滤波隐含"密度均匀"假设，会把远处稀疏但正常的
    表面点成片误删；分壳后每个点只与同距离段的点比较，避免系统性误删。
    返回保留掩码 (N,) bool。
    """
    pts = np.asarray(pcd.points)
    n = len(pts)
    if n < 2:
        return np.ones(n, dtype=bool)
    nn_dists = np.asarray(pcd.compute_nearest_neighbor_distance())
    ranges = estimate_point_ranges(pts, resolve_scan_origin(pts, scan_origin))
    keep = np.ones(n, dtype=bool)
    n_shells = max(1, int(n_shells))
    edges = np.quantile(ranges, np.linspace(0.0, 1.0, n_shells + 1))
    for si in range(n_shells):
        lo, hi = edges[si], edges[si + 1]
        if si == n_shells - 1:
            mask = (ranges >= lo) & (ranges <= hi)
        else:
            mask = (ranges >= lo) & (ranges < hi)
        if int(np.sum(mask)) < 30:
            continue
        d = nn_dists[mask]
        med = float(np.median(d))
        mad = 1.4826 * float(np.median(np.abs(d - med)))
        # MAD 下限取中位数的15%：规则化点云（体素下采样后近似点阵）上 MAD
        # 趋近0，否则阈值过紧会误删正常点
        thr = med + std_ratio * max(mad, 0.15 * med, 1e-6)
        bad = np.where(mask)[0][d > thr]
        keep[bad] = False
    return keep


def fit_plane_weighted(points, weights=None, irls_iters=0):
    """
    加权SVD平面拟合，可选 IRLS(Huber) 迭代抗离群。

    weights: 先验权重（如 1/σ(r)²，近处高权重主导拟合，远处噪声点不拽偏平面）
    irls_iters: >0 时在先验权重基础上叠加 Huber 权重迭代，抑制残差离群点
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return np.array([0, 0, 1, 0], dtype=float)

    base_w = np.ones(len(pts)) if weights is None else np.clip(
        np.asarray(weights, dtype=float), 1e-12, None)
    w = base_w
    model = None
    for it in range(max(1, int(irls_iters) + 1)):
        w_sum = float(np.sum(w))
        center = (pts * w[:, None]).sum(axis=0) / w_sum
        shifted = (pts - center) * np.sqrt(w)[:, None]
        _, _, vh = np.linalg.svd(shifted, full_matrices=False)
        normal = vh[-1]
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        d = -float(np.dot(normal, center))
        model = np.array([normal[0], normal[1], normal[2], d], dtype=float)
        if it >= irls_iters:
            break
        residuals = np.abs(pts @ normal + d)
        scale = 1.4826 * float(np.median(residuals)) + 1e-9
        k = 1.345 * scale
        huber = np.where(residuals <= k, 1.0, k / np.maximum(residuals, 1e-12))
        w = base_w * huber
    return model

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

def ensure_normals(pcd, voxel_size=0.05, inplace=False, force=False, orient=False):
    """估计点云法向量；默认复用已有法向，避免大点云重复耗时计算。"""
    if len(pcd.points) == 0:
        return pcd
    pcd_work = pcd if inplace else o3d.geometry.PointCloud(pcd)

    has_valid_normals = (
        pcd_work.has_normals() and
        len(pcd_work.normals) == len(pcd_work.points)
    )
    if has_valid_normals and not force:
        return pcd_work

    # 大半径 + 邻居数封顶：近处等效 kNN(30)，远处稀疏区自动扩大搜索范围，
    # 避免固定小半径在远处凑不齐邻居导致法向噪声过大
    radius = max(float(voxel_size) * 4.0, 2.0)
    pcd_work.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    if orient:
        try:
            pcd_work.orient_normals_consistent_tangent_plane(20)
        except Exception:
            pcd_work.orient_normals_towards_camera_location([0, 0, 0])
    return pcd_work

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
    """
    计算垂直度相关指标
    法向量 n=(A,B,C), Z轴 e_z=(0,0,1)
    cosθ = C/√(A²+B²+C²)
    Δ_wall = 2·tan(π/2 - θ)
    """
    normal = np.asarray(normal, dtype=float)
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    nz = abs(normal[2])

    verticality_deg = float(np.degrees(np.arcsin(np.clip(nz, 0.0, 1.0))))

    if verticality_deg < 89.9:
        deviation_m = float(measure_height * np.tan(np.deg2rad(verticality_deg)))
    else:
        deviation_m = float('inf')

    return verticality_deg, deviation_m, deviation_m * 1000.0

def project_to_uv(points, plane_model, facade_type=None):
    """
    将3D点投影到平面的局部UV坐标系。

    返回:
        uv: (N, 2) 数组，u为水平轴，v为竖直轴（立面时v沿Z轴）
        u_axis, v_axis: 局部坐标轴
        center: 平面中心（用于反投影）
    """
    normal = np.asarray(plane_model[:3], dtype=float)
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    center = np.mean(points, axis=0)

    u, v = plane_axes(normal, facade_type)

    # 投影：偏移到中心后点乘u、v轴
    centered = points - center
    u_coords = centered @ u
    v_coords = centered @ v

    return np.column_stack([u_coords, v_coords]), u, v, center

def connected_components_2d_grid(uv_points, grid_size, min_cells=3,
                                 point_bridge_m=None, max_bridge_cells=3):
    """
    基于网格的2D连通域分析。

    参数:
        uv_points: (N, 2) 投影坐标
        grid_size: 网格单元大小（如 voxel_size * 2）
        min_cells: 最小连通域占用的网格数（过滤噪声，只统计真实占用格）
        point_bridge_m: 可选 (N,) 每点桥接半径（米）。远处点云稀疏，点间距
            超过网格尺寸时占用格互不相邻会碎片化；给稀疏点一个与其局部点间距
            成比例的桥接半径，可将同一立面的远端连回来
        max_bridge_cells: 单点桥接半径上限（格数），防止过度桥接吞并邻近平面

    返回:
        list of np.array(bool_mask)，每个连通域一个掩码
    """
    if len(uv_points) == 0:
        return []

    # 1. 坐标归一化到网格索引
    u_min, v_min = np.min(uv_points, axis=0)
    u_max, v_max = np.max(uv_points, axis=0)

    # 避免单点或极小范围导致的除零
    if u_max - u_min < 1e-6 and v_max - v_min < 1e-6:
        return [np.ones(len(uv_points), dtype=bool)]

    # 网格索引（向下取整）
    grid_u = np.floor((uv_points[:, 0] - u_min) / grid_size).astype(np.int32)
    grid_v = np.floor((uv_points[:, 1] - v_min) / grid_size).astype(np.int32)

    # 2. 构建稀疏占用图（用字典记录每个网格中的点索引）
    grid_to_indices = {}
    for i, (gu, gv) in enumerate(zip(grid_u, grid_v)):
        key = (gu, gv)
        if key not in grid_to_indices:
            grid_to_indices[key] = []
        grid_to_indices[key].append(i)

    if not grid_to_indices:
        return []

    # 2.5 稀疏桥接格：按格取该格内点的最大桥接半径，向外扩张为"虚格"。
    # 虚格参与连通遍历但不含点、不计入 min_cells
    bridge_cells = set()
    if point_bridge_m is not None:
        reach_cells = np.floor(
            np.asarray(point_bridge_m, dtype=float) / float(grid_size)
        ).astype(np.int32)
        reach_cells = np.clip(reach_cells, 0, int(max_bridge_cells))
        cell_reach = {}
        for i in np.where(reach_cells > 0)[0]:
            key = (grid_u[i], grid_v[i])
            prev = cell_reach.get(key, 0)
            if reach_cells[i] > prev:
                cell_reach[key] = int(reach_cells[i])
        for (gu, gv), w in cell_reach.items():
            for du in range(-w, w + 1):
                for dv in range(-w, w + 1):
                    bridge_cells.add((gu + du, gv + dv))
        bridge_cells -= set(grid_to_indices.keys())

    # 3. 四连通BFS/DFS找连通域（真实格 + 虚格）
    visited = set()
    components = []

    for key in grid_to_indices:
        if key in visited:
            continue

        # BFS
        queue = [key]
        visited.add(key)
        cell_cluster = [key]

        while queue:
            cu, cv = queue.pop(0)
            # 四邻域
            for du, dv in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nb = (cu + du, cv + dv)
                if nb in visited:
                    continue
                if nb in grid_to_indices:
                    visited.add(nb)
                    queue.append(nb)
                    cell_cluster.append(nb)
                elif nb in bridge_cells:
                    visited.add(nb)
                    queue.append(nb)

        # 过滤小连通域（cell_cluster 只收集了真实占用格）
        if len(cell_cluster) < min_cells:
            continue

        # 收集该连通域包含的所有原始点
        point_mask = np.zeros(len(uv_points), dtype=bool)
        for cell in cell_cluster:
            for idx in grid_to_indices[cell]:
                point_mask[idx] = True

        components.append(point_mask)

    return components

def compute_registration_error(src_pcd, tgt_pcd, transformation, threshold=1.0):
    """计算配准后的RMSE和重叠率"""
    import copy
    src_transformed = copy.deepcopy(src_pcd)
    src_transformed.transform(transformation)
    dists = src_transformed.compute_point_cloud_distance(tgt_pcd)
    dists = np.asarray(dists)
    inlier_dists = dists[dists < threshold]
    rmse = np.sqrt(np.mean(inlier_dists ** 2)) if len(inlier_dists) > 0 else float('inf')
    overlap_ratio = len(inlier_dists) / len(dists) if len(dists) > 0 else 0
    return {
        'rmse': float(rmse),
        'overlap_ratio': float(overlap_ratio),
        'inlier_count': int(len(inlier_dists)),
        'total_count': int(len(dists))
    }

def is_uniform_color(colors, eps=1e-4):
    if len(colors) == 0:
        return True
    r, g, b = colors[0]
    for i in range(1, len(colors)):
        if abs(colors[i][0] - r) > eps or abs(colors[i][1] - g) > eps or abs(colors[i][2] - b) > eps:
            return False
    return True