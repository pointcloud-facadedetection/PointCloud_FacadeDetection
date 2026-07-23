import numpy as np
import open3d as o3d
import copy


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
                                 adaptive_ratio=2.5, sample_ratio=0.1):
    """基于UV网格的四连通域分析，用于拆分同平面但空间不连续的立面。"""
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
    for key in grid_to_indices:
        if key in visited:
            continue

        queue = [key]
        visited.add(key)
        cell_cluster = [key]

        while queue:
            cu, cv = queue.pop(0)
            for du, dv in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
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