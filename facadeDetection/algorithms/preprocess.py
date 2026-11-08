import numpy as np
import open3d as o3d
from typing import Optional, Tuple


def finite_sparse_keep_indices(points: np.ndarray, voxel_size: float = 0.05,
                               radius: Optional[float] = None,
                               min_neighbors: Optional[int] = None,
                               enabled: bool = True) -> np.ndarray:
    """Return a deterministic keep mask for finite points and isolated noise.

    The returned mask is in the *input* coordinate system.  This is deliberately
    separate from :func:`denoise`: facade detection exposes input-local indices,
    so callers can slice every synchronized feature array with the same mask.
    A conservative default is used because sparse facade samples must not be
    removed merely for being farther from the scanner.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be (N,3)")
    finite = np.all(np.isfinite(pts), axis=1)
    if not enabled or finite.sum() < 12:
        return finite

    ids = np.flatnonzero(finite)
    clean = pts[ids]
    r = float(radius if radius is not None else max(float(voxel_size) * 2.5, 0.08))
    k = int(min_neighbors if min_neighbors is not None else 2)
    if r <= 0 or k < 1:
        raise ValueError("radius must be > 0 and min_neighbors must be >= 1")

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(clean))
    # Open3D 的 nb_points 计数包含查询点自身。
    radius_eps = max(np.finfo(np.float32).eps * max(abs(r), 1.0) * 8.0,
                     1e-9)
    query_radius = r + radius_eps
    if k == 1:
        # Open3D counts the query point itself, so every finite sample satisfies
        # the one-point rule.
        kept_local = np.arange(len(clean), dtype=np.int64)
    elif k == 2:
        nearest = np.asarray(
            cloud.compute_nearest_neighbor_distance(), dtype=np.float64)
        kept_local = np.flatnonzero(nearest <= query_radius)
    else:
        _, kept_local = cloud.remove_radius_outlier(
            nb_points=k, radius=query_radius)
    keep = np.zeros(len(pts), dtype=bool)
    if len(kept_local):
        keep[ids[np.asarray(kept_local, dtype=np.int64)]] = True
    if np.count_nonzero(keep) < 3:
        return finite
    return keep

def voxel_downsample(points: np.ndarray,
                     colors: Optional[np.ndarray] = None,
                     voxel_size: float = 0.05) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        empty_pts = pts.reshape(0, 3)
        if colors is not None:
            empty_col = np.asarray(colors, dtype=np.float64).reshape(0, 3)
            return empty_pts, empty_col
        return empty_pts, None

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be (N,3)")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        col = np.asarray(colors, dtype=np.float64)
        if col.shape != pts.shape:
            raise ValueError("colors must have same shape as points")
        pcd.colors = o3d.utility.Vector3dVector(col)

    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    pts_ds = np.asarray(pcd_down.points, dtype=np.float64)
    if colors is not None:
        cols_ds = np.asarray(pcd_down.colors, dtype=np.float64)
        return pts_ds, cols_ds
    return pts_ds, None

def denoise(pcd: o3d.geometry.PointCloud,
            voxel_size: float,
            method: str = 'radius',
            **kwargs) -> o3d.geometry.PointCloud:
    """
    :param kwargs: 自定义滤波参数
        radius: 搜索半径，默认 voxel_size * 2
        min_neighbors: 半径内最少点数阈值(半径滤波)，默认10
        nb_neighbors: 统计滤波邻域点数，默认20
        std_ratio: 统计滤波标准差倍数，默认2.0（adaptive 默认2.5）
        scan_origin: 扫描仪位置(adaptive)，默认坐标原点
        n_shells: 距离分壳数(adaptive)，默认24
    :return: 去噪后点云对象

    注意：radius/statistical 隐含"密度均匀"假设，对单站扫描的远处稀疏区
    会成片误删正常表面点；单站数据建议用 adaptive（距离分壳自适应）。
    """
    if not isinstance(pcd, o3d.geometry.PointCloud):
        raise TypeError("pcd must be open3d.geometry.PointCloud instance")
    original_num = len(pcd.points)
    if original_num == 0:
        return pcd

    # 解析滤波参数
    radius = float(kwargs.get("radius", voxel_size * 2))
    min_neighbors = int(kwargs.get("min_neighbors", 10))
    nb_neighbors = int(kwargs.get("nb_neighbors", 20))
    std_ratio = float(kwargs.get("std_ratio", 2.0))

    clean_pcd = None
    if method == "radius":
        # 半径离群点去除
        clean_pcd, _ = pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
    elif method == "statistical":
        # 统计离群点去除
        clean_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    elif method == "adaptive":
        # 距离分壳自适应滤波：远处稀疏但正常的表面点不会被当成离群点
        from algorithms.geometry import adaptive_outlier_indices, estimate_point_ranges
        pts = np.asarray(pcd.points, dtype=np.float64)
        ranges = kwargs.get('ranges')
        if ranges is None and kwargs.get('scan_origin') is not None:
            ranges = estimate_point_ranges(pts, kwargs.get('scan_origin'))
        if ranges is None:
            ranges = np.zeros(len(pts), dtype=np.float64)
        keep = adaptive_outlier_indices(
            pts,
            np.asarray(ranges, dtype=np.float64).reshape(-1),
            std_ratio=float(kwargs.get("std_ratio", 2.5)),
            n_shells=int(kwargs.get("n_shells", 8)),
        )
        clean_pcd = pcd.select_by_index(np.where(keep)[0].tolist())
    else:
        raise ValueError(f"unsupported denoise method: {method}")

    return clean_pcd