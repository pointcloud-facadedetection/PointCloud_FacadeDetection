import numpy as np
import open3d as o3d
from typing import Optional, Tuple
def voxel_downsample(points: np.ndarray,
                     colors: Optional[np.ndarray] = None,
                     voxel_size: float = 0.05) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Use Open3D native interface to implement voxel grid downsampling.
    - points: (N,3) float array
    - colors: optional (N,3) float array in range [0,1]
    - voxel_size: positive float

    Returns (points_ds, colors_ds)
    """
    pts = np.asarray(points, dtype=np.float64)
    # 空数组兜底
    if pts.size == 0:
        empty_pts = pts.reshape(0, 3)
        if colors is not None:
            empty_col = np.asarray(colors, dtype=np.float64).reshape(0, 3)
            return empty_pts, empty_col
        return empty_pts, None

    # 入参合法性校验
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be (N,3)")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")

    # 构建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # 挂载颜色信息
    if colors is not None:
        col = np.asarray(colors, dtype=np.float64)
        if col.shape != pts.shape:
            raise ValueError("colors must have same shape as points")
        pcd.colors = o3d.utility.Vector3dVector(col)

    # Open3D原生体素下采样
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

    # 转回numpy数组输出
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
    # 校验点云合法性
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
            n_shells=int(kwargs.get("n_shells", 24)),
        )
        clean_pcd = pcd.select_by_index(np.where(keep)[0].tolist())
    else:
        raise ValueError(f"unsupported denoise method: {method}, only support [radius, statistical, adaptive]")

    return clean_pcd
