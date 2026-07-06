import os
import time
import numpy as np
import open3d as o3d

from ..core.cache import get_cache
from ..config import Config


def process_ply(filepath, voxel_size):
    """加载PLY文件，下采样"""
    print(f"[INFO] 加载点云: {filepath}")
    start = time.time()

    pcd = o3d.io.read_point_cloud(filepath)
    original_count = len(pcd.points)

    if original_count == 0:
        raise ValueError("点云为空")

    print(f"[INFO] 原始点数: {original_count:,}")

    if voxel_size > 0:
        display_pcd = pcd.voxel_down_sample(voxel_size)
        print(f"[INFO] 下采样: 体素={voxel_size}m, {original_count:,} -> {len(display_pcd.points):,}")
    else:
        display_pcd = pcd
        print(f"[INFO] 无下采样: {original_count:,} 点")

    load_time = time.time() - start
    print(f"[INFO] 加载耗时: {load_time:.2f}s")

    if not display_pcd.has_colors():
        display_pcd.paint_uniform_color([0.7, 0.7, 0.7])

    return display_pcd


def save_cloud(uuid_name, filename):
    """保存当前点云"""
    cache = get_cache()
    pcd = cache.get_display(uuid_name)
    if pcd is None:
        raise ValueError('点云未找到')

    save_path = os.path.join(Config.UPLOAD_FOLDER, filename)
    o3d.io.write_point_cloud(save_path, pcd)

    return save_path, len(pcd.points)


def denoise_cloud(uuid_name, voxel_size, method='radius', **kwargs):
    """去噪处理"""
    cache = get_cache()
    pcd = cache.get_display(uuid_name)
    if pcd is None:
        raise ValueError('点云未找到')

    radius = float(kwargs.get('radius', voxel_size * 2))
    min_neighbors = int(kwargs.get('min_neighbors', 10))
    nb_neighbors = int(kwargs.get('nb_neighbors', 20))
    std_ratio = float(kwargs.get('std_ratio', 2.0))

    if method == 'radius':
        clean_pcd, _ = pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
    else:
        clean_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    if len(clean_pcd.points) < 0.1 * len(pcd.points):
        clean_pcd = pcd

    cache.set_display(uuid_name, clean_pcd)

    return clean_pcd


def compute_normals(uuid_name, voxel_size):
    """计算法向量"""
    cache = get_cache()
    pcd = cache.get_display(uuid_name)
    if pcd is None:
        raise ValueError('点云未找到')

    pcd_work = o3d.geometry.PointCloud(pcd)
    if voxel_size > 0:
        pcd_work = pcd_work.voxel_down_sample(voxel_size)

    radius = max(float(voxel_size) * 4.0, 0.2)
    pcd_work.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50)
    )
    try:
        pcd_work.orient_normals_consistent_tangent_plane(30)
    except Exception:
        pcd_work.orient_normals_towards_camera_location([0, 0, 0])

    cache.set_display(uuid_name, pcd_work)
    return pcd_work


def get_bounding_box(uuid_name):
    """获取边界框"""
    cache = get_cache()
    pcd = cache.get_display(uuid_name)
    if pcd is None:
        raise ValueError('点云未找到')

    min_bound = pcd.get_min_bound()
    max_bound = pcd.get_max_bound()
    center = (min_bound + max_bound) / 2
    size = max_bound - min_bound

    return {
        'center': center.tolist(),
        'size': size.tolist(),
        'min': min_bound.tolist(),
        'max': max_bound.tolist()
    }

