from __future__ import annotations

from typing import Optional, Dict
import logging
import uuid

import numpy as np
import open3d as o3d

from algorithms.preprocess import denoise
from .pointcloud_index import RawPointStore, PointCloudDataset, VoxelCascadeIndex, DecisionSet

log = logging.getLogger("facadeDetection.pointcloud")


class PointCloudService:
    """
    点云相关业务逻辑 Service：
    - 视口点云数据的读取/更新
    - 算法调用封装
    """

    def __init__(self, viewport=None, render_service=None):
        self.viewport = viewport
        self.render_service = render_service
        self.datasets: Dict[str, PointCloudDataset] = {}
        self.source_assets: Dict[str, dict] = {}
        self.decisions: Dict[str, DecisionSet] = {}
        self._decision_versions: Dict[str, int] = {}

    def register_dataset(self, dataset_id: str, points, colors=None, metadata=None) -> PointCloudDataset:
        raw = RawPointStore.from_arrays(points, colors)
        meta = metadata or {}
        # 使用细粒度索引，仅承担 proxy->voxel->raw 映射，不再下采样。
        index_size = 1e-4 if meta.get('pipeline_version', '').startswith('range-adaptive') else None
        dataset = PointCloudDataset(dataset_id, raw,
                                    VoxelCascadeIndex.build(raw, voxel_size=index_size), meta)
        self.datasets[dataset_id] = dataset
        #  数据集替换会生成一个新的代理/体素 ID 空间。
        self.decisions.pop(dataset_id, None)
        self._decision_versions.pop(dataset_id, None)
        return dataset

    def register_source_asset(self, source_id: str, points, colors=None, metadata=None):
        """保存 FLS 原始点，供质量域在 processed index 展开后回到 source Raw。"""
        self.source_assets[source_id] = {
            "points": np.ascontiguousarray(np.asarray(points, dtype=np.float32).reshape(-1, 3)),
            "colors": None if colors is None else np.ascontiguousarray(np.asarray(colors, dtype=np.float32).reshape(-1, 3)),
            "metadata": metadata or {},
        }

    def get_source_asset(self, source_id):
        return self.source_assets.get(source_id)

    def get_metadata(self, dataset_id):
        dataset = self.datasets.get(dataset_id)
        return dataset.metadata if dataset else None

    def get_dataset(self, dataset_id: str) -> Optional[PointCloudDataset]:
        """返回用于 facade/quality 服务的注册数据."""
        return self.datasets.get(dataset_id)

    def map_proxy_decision(self, dataset_id: str, proxy_ids, source: str = "unknown",
                           expand_raw: bool = True) -> np.ndarray | None:
        dataset = self.datasets[dataset_id]
        voxel_ids = dataset.index.voxel_ids_for_proxy(proxy_ids)
        version = self._decision_versions.get(dataset_id, 0) + 1
        self._decision_versions[dataset_id] = version
        decision = DecisionSet(dataset_id, voxel_ids, source,
                               operation_id=uuid.uuid4().hex, version=version)
        self.decisions[dataset_id] = decision
        log.info("[PCFD] decision.done dataset=%s source=%s proxy=%d voxel=%d raw=deferred version=%d",
                 dataset_id, source, len(np.asarray(proxy_ids).reshape(-1)), len(voxel_ids), version)
        return dataset.index.voxel_to_raw_ids(voxel_ids) if expand_raw else None

    def raw_ids_for_aabb(self, dataset_id: str, min_bound, max_bound) -> np.ndarray:
        dataset = self.datasets[dataset_id]
        return dataset.index.query_aabb(min_bound, max_bound, dataset.raw.points, exact=True)

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

    def denoise(self, method: str = "adaptive", voxel_size: float = 0.05,
                update_viewport: bool = True, **kwargs) -> Optional[Dict]:
        """
        对当前活动点云执行去噪，并回写视口。
        :param method: 'adaptive'（默认） / 'radius' / 'statistical'
        :param voxel_size: 影响半径默认值等
        :param kwargs: radius/min_neighbors/nb_neighbors/std_ratio/scan_origin/n_shells
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

        dataset_id = data.get("dataset_id")

        pts = np.asarray(data["pos"], dtype=np.float32)
        cols = None
        try:
            if "color" in data and data["color"] is not None:
                cols = np.asarray(data["color"], dtype=np.float32)
        except Exception:
            cols = None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        if cols is not None and len(cols) == len(pts):
            pcd.colors = o3d.utility.Vector3dVector(cols)

        # 自适应去噪需要扫描仪位置；优先从 dataset metadata 读取
        dataset = self.datasets.get(dataset_id) if dataset_id else None
        if method == "adaptive" and dataset is not None:
            scan_origins = kwargs.get("scan_origin") or (dataset.metadata or {}).get("scan_origins")
            if scan_origins is not None:
                kwargs["scan_origin"] = np.asarray(scan_origins, dtype=np.float64)

        clean = denoise(pcd, voxel_size=voxel_size, method=str(method), **kwargs)

        new_pts = np.asarray(clean.points, dtype=np.float32)
        new_cols = None
        try:
            if clean.has_colors():
                new_cols = np.asarray(clean.colors, dtype=np.float32)
        except Exception:
            new_cols = None

        raw_ids = None
        dataset = self.datasets.get(dataset_id) if dataset_id else None
        if dataset is not None and len(new_pts):
            proxy_ids = np.asarray(data.get("proxy_ids", []), dtype=np.int32)
            if len(proxy_ids) != len(pts):
                proxy_ids = np.arange(len(pts), dtype=np.int32)
            keep_proxy = dataset.index.proxy_ids_for_points(new_pts)
            # 过滤无效映射，避免 -1 导致后续索引错位
            keep_proxy = keep_proxy[(keep_proxy >= 0) & (keep_proxy < len(proxy_ids))]
            proxy_ids = proxy_ids[keep_proxy] if len(keep_proxy) else np.empty(0, dtype=np.int32)
            self.map_proxy_decision(dataset_id, proxy_ids, "denoise", expand_raw=False)
            raw_ids = None

        if not update_viewport:
            stats = {
                "name": name, "method": method, "voxel_size": float(voxel_size),
                "points_before": int(len(pts)), "points_after": int(len(new_pts)),
                "dataset_id": dataset_id, "raw_ids": raw_ids,
                "raw_count": dataset.index.raw_count_for_voxels(self.decisions[dataset_id].voxel_ids) if dataset_id in self.decisions else 0,
                "proxy_points": new_pts, "proxy_colors": new_cols,
            }
            return stats
        if hasattr(vp, "queue_update_cloud_points"):
            vp.queue_update_cloud_points(name, new_pts, new_cols)
        elif hasattr(vp, "update_cloud_points"):
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
            "dataset_id": dataset_id,
            "raw_ids": raw_ids,
        }
        print(
            f"PointCloudService: 去噪完成: {name}, 点数 {stats['points_before']} -> {stats['points_after']} ({method}), "
            f"raw子集点数={0 if raw_ids is None else len(raw_ids)}",
            flush=True,
        )
        return stats
