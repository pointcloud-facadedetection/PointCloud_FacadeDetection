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

    radius = max(float(voxel_size) * 4.0, 0.2)
    pcd_work.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=40)
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