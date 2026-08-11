from __future__ import annotations

from typing import Optional, Dict

import numpy as np
import open3d as o3d

from algorithms.preprocess import denoise


class PointCloudService:
    """
    点云相关业务逻辑 Service：
    - 视口点云数据的读取/更新
    - 算法调用封装
    """

    def __init__(self, viewport=None, render_service=None):
        self.viewport = viewport
        self.render_service = render_service

    def set_dependencies(self, viewport=None, render_service=None):
        if viewport is not None:
            self.viewport = viewport
        if render_service is not None:
            self.render_service = render_service

    def _pick_active_cloud_name(self) -> Optional[str]:
        vp = self.viewport
        if vp is None:
            return None
        # 优先活动点云
        name = getattr(vp, "_active_name", None)
        if name:
            return name
        # 退化选择最后一个
        if hasattr(vp, "get_cloud_names"):
            names = vp.get_cloud_names()
            if names:
                return names[-1]
        return None

    def denoise(self, method: str = "radius", voxel_size: float = 0.05, **kwargs) -> Optional[Dict]:
        """
        对当前活动点云执行去噪，并回写视口。
        :param method: 'radius' 或 'statistical'
        :param voxel_size: 影响半径默认值等
        :param kwargs: radius/min_neighbors/nb_neighbors/std_ratio
        :return: 统计信息字典（原始点数/新点数/云名称/方法），或 None（失败）
        """
        vp = self.viewport
        if vp is None:
            print("PointCloudService: viewport 未注入，无法去噪", flush=True)
            return None

        name = self._pick_active_cloud_name()
        if not name:
            print("PointCloudService: 未找到可去噪的点云", flush=True)
            return None

        data = vp.get_cloud_data(name) if hasattr(vp, "get_cloud_data") else None
        if not data or "pos" not in data:
            print(f"PointCloudService: 点云数据不可用: {name}", flush=True)
            return None

        pts = np.asarray(data["pos"], dtype=np.float64)
        cols = None
        try:
            if "color" in data and data["color"] is not None:
                cols = np.asarray(data["color"], dtype=np.float64)
        except Exception:
            cols = None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        if cols is not None and len(cols) == len(pts):
            pcd.colors = o3d.utility.Vector3dVector(cols)

        clean = denoise(pcd, voxel_size=voxel_size, method=str(method), **kwargs)

        new_pts = np.asarray(clean.points, dtype=np.float64)
        new_cols = None
        try:
            if clean.has_colors():
                new_cols = np.asarray(clean.colors, dtype=np.float64)
        except Exception:
            new_cols = None

        if hasattr(vp, "update_cloud_points"):
            vp.update_cloud_points(name, new_pts, new_cols)
        elif hasattr(vp, "add_cloud"):
            # 退化方案：替换添加（不重置视图）
            vp.add_cloud(name, new_pts, new_cols)

        stats = {
            "name": name,
            "method": method,
            "voxel_size": float(voxel_size),
            "points_before": int(len(pts)),
            "points_after": int(len(new_pts)),
        }
        print(
            f"PointCloudService: 去噪完成: {name}, 点数 {stats['points_before']} -> {stats['points_after']} ({method})",
            flush=True,
        )
        return stats
