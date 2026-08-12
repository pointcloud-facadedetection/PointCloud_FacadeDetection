import os
import re
import time
import numpy as np
import open3d as o3d

from ..core.cache import get_cache
from ..core.geometry_utils import adaptive_outlier_mask
from ..config import Config


def _voxel_tag(voxel_size):
    """将体素尺寸格式化为稳定文件名片段。"""
    return f"{float(voxel_size):.6g}".replace('.', 'p')


def _safe_filename(filename):
    """生成可用于缓存路径的安全文件名。"""
    base = os.path.basename(filename or 'cloud.ply')
    base = re.sub(r'[^\w.\-]+', '_', base, flags=re.UNICODE)
    return base or 'cloud.ply'


def get_downsample_cache_path(original_filename, voxel_size, source_size):
    """按原始文件名 + 文件大小 + 体素尺寸生成缓存路径。"""
    name = _safe_filename(original_filename)
    stem, _ = os.path.splitext(name)
    tag = _voxel_tag(voxel_size)
    return os.path.join(
        Config.CACHE_FOLDER,
        f"{stem}_s{int(source_size)}_v{tag}.ply"
    )


def process_ply(filepath, voxel_size, original_filename=None):
    """加载PLY文件并下采样；同文件同体素优先读取落盘缓存。"""
    print(f"[INFO] 加载点云: {filepath}")
    start = time.time()

    cache_name = original_filename or os.path.basename(filepath)
    source_size = os.path.getsize(filepath)
    cache_path = None

    if float(voxel_size) > 0:
        os.makedirs(Config.CACHE_FOLDER, exist_ok=True)
        cache_path = get_downsample_cache_path(cache_name, voxel_size, source_size)
        if os.path.isfile(cache_path):
            print(f"[INFO] 命中下采样缓存: {cache_path}")
            display_pcd = o3d.io.read_point_cloud(cache_path)
            if len(display_pcd.points) == 0:
                print("[WARN] 缓存点云为空，忽略并重新处理")
            else:
                if not display_pcd.has_colors():
                    display_pcd.paint_uniform_color([0.7, 0.7, 0.7])
                print(
                    f"[INFO] 缓存加载: {len(display_pcd.points):,} 点, "
                    f"耗时 {time.time() - start:.2f}s"
                )
                return display_pcd

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

    if cache_path is not None:
        ok = o3d.io.write_point_cloud(cache_path, display_pcd)
        if ok:
            print(f"[INFO] 已写入下采样缓存: {cache_path}")
        else:
            print(f"[WARN] 写入下采样缓存失败: {cache_path}")

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
        # 注意：固定半径滤波对单站扫描的远处稀疏区会成片误删，远场建议用 adaptive
        clean_pcd, _ = pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
    elif method == 'adaptive':
        # 距离分壳自适应滤波：远处稀疏但正常的表面点不会被当成离群点
        keep = adaptive_outlier_mask(
            pcd,
            scan_origin=kwargs.get('scan_origin'),
            std_ratio=float(kwargs.get('std_ratio', 2.5)),
        )
        clean_pcd = pcd.select_by_index(np.where(keep)[0].tolist())
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

    # 大半径 + 邻居数封顶：远处稀疏区自动扩大搜索范围（与 geometry_utils.ensure_normals 一致）
    radius = max(float(voxel_size) * 4.0, 2.0)
    pcd_work.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
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
