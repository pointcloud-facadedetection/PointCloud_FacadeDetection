"""立面检测业务编排服务。

仅负责检测流程调度、参数组装与结果持久化；不直接操作 UI 控件，也不包含
底层检测算法实现。
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import open3d as o3d

from algorithms.facade.adaptive_facade_detection import detect_facades_adaptive
from config.settings import Config
from services.dal.results_repo import ResultsRepo
from services.facade.facade_index_service import FacadeIndexService
from utils.logging_utils import trace

log = logging.getLogger("facadeDetection.facade")


class FacadeDetectionService:
    """外立面检测业务编排服务。"""

    def __init__(self, viewport, pointcloud_service=None, index_service: Optional[FacadeIndexService] = None):
        self._viewport = viewport
        self._pointcloud_service = pointcloud_service
        self._index_service = index_service

    def set_index_service(self, index_service: FacadeIndexService):
        self._index_service = index_service

    def _get_dataset(self, cloud_name: str):
        """获取点云数据集。"""
        data = self._viewport.get_cloud_data(cloud_name)
        if data is None:
            return None, None
        dataset_id = data.get('dataset_id')
        service = self._pointcloud_service
        dataset = service.get_dataset(dataset_id) if service and dataset_id else None
        return data, dataset

    def detect(self, cloud_name: str, project_uuid: Optional[str] = None) -> list[dict]:
        """全局立面检测。"""
        started = time.perf_counter()
        trace("facade.detect.begin", cloud=cloud_name)

        pcd = self._viewport.get_cloud_data(cloud_name)
        if pcd is None:
            return []

        data, dataset = self._get_dataset(cloud_name)
        if dataset is None:
            trace("facade.detect.warn", cloud=cloud_name, reason="no_dataset")

        geo = o3d.geometry.PointCloud()
        geo.points = o3d.utility.Vector3dVector(pcd["pos"].astype(float))
        if pcd.get("color") is not None and len(pcd["color"]) == len(pcd["pos"]):
            geo.colors = o3d.utility.Vector3dVector(pcd["color"].astype(float))

        vsize = float(getattr(Config, 'DEFAULT_VOXEL_SIZE', 0.05))
        min_area = float(getattr(Config, 'MIN_FACADE_AREA', 10.0))

        detect_kwargs = (dataset.metadata if dataset is not None else {}) or {}
        result = detect_facades_adaptive(
            geo, voxel_size=vsize, min_facade_area=min_area,
            metadata=detect_kwargs,
            scan_origin=detect_kwargs.get('scan_origins'),
        )

        facades = result.get('facades', []) if isinstance(result, dict) else (result or [])

        if self._index_service is not None:
            proxy_ids = self._index_service.proxy_ids_for_positions(
                cloud_name, np.asarray(pcd['pos']), np.arange(len(pcd['pos'])))
            self._index_service.normalize_facade_indices(facades, proxy_ids)
            self._index_service.populate_voxel_ids(facades, cloud_name)

        trace("facade.detect.done", cloud=cloud_name, facades=len(facades),
              seconds=f"{time.perf_counter()-started:.2f}")

        if project_uuid:
            self._persist_results(project_uuid, facades)
        return facades

    def detect_on_roi(self, cloud_name: str,
                      roi_indices: list[int] | None = None,
                      roi_bounds: tuple | None = None,
                      roi_box: dict | None = None,
                      project_uuid: Optional[str] = None,
                      min_facade_area: float | None = None,
                      roi_scope: str = 'seed') -> list[dict]:
        """立面检测入口。

        ``roi_scope`` 保持三种明确语义：
        - ``global``：没有 ROI 时使用完整代理点云；
        - ``seed``：ROI 只用于筛选命中的完整立面，算法仍运行于全场景；
        - ``clip``：兼容旧行为，仅在 ROI 内检测。
        """
        started = time.perf_counter()
        if roi_scope not in {'global', 'seed', 'clip'}:
            raise ValueError(f'unsupported roi_scope: {roi_scope}')
        has_roi = roi_box is not None or roi_bounds is not None or (
            roi_indices is not None and len(roi_indices) > 0
        )
        if not has_roi:
            roi_scope = 'global'
        trace("facade.roi.begin", cloud=cloud_name,
              roi_indices=len(roi_indices or []), bounds=bool(roi_bounds),
              box=bool(roi_box), scope=roi_scope)

        pcd = self._viewport.get_cloud_data(cloud_name)
        if pcd is None:
            return []

        data, dataset = self._get_dataset(cloud_name)

        pos = np.asarray(pcd["pos"], dtype=float).reshape(-1, 3)
        col = pcd.get("color")
        if min_facade_area is None:
            min_facade_area = 5.0

        seed_mask = np.ones(len(pos), dtype=bool)
        if roi_box is not None:
            center = np.asarray(roi_box['center'], dtype=float)
            axes = np.asarray(roi_box['axes'], dtype=float).reshape(3, 3)
            half = np.asarray(roi_box['half_extent'], dtype=float)
            local = (pos - center) @ axes.T
            seed_mask = np.all(np.abs(local) <= half + 1e-9, axis=1)
        elif roi_bounds is not None:
            bmin = np.asarray(roi_bounds[0], dtype=float)
            bmax = np.asarray(roi_bounds[1], dtype=float)
            seed_mask = np.all((pos >= bmin) & (pos <= bmax), axis=1)
        elif roi_indices is not None and len(roi_indices):
            seed_mask = np.zeros(len(pos), dtype=bool)
            valid = np.asarray(roi_indices, dtype=int)
            seed_mask[valid[(valid >= 0) & (valid < len(pos))]] = True

        crop_mask = seed_mask if roi_scope == 'clip' else np.ones(len(pos), dtype=bool)
        global_indices = np.flatnonzero(crop_mask).astype(np.int64)
        seed_indices_local = np.flatnonzero(seed_mask).astype(np.int64)
        trace("facade.roi.selected", source_points=len(pos),
              selected=len(global_indices), roi_seed=len(seed_indices_local),
              seconds=f"{time.perf_counter()-started:.2f}")

        if len(global_indices) == 0:
            return []

        roi_pos = pos[global_indices]
        roi_col = np.asarray(col)[global_indices] if col is not None and len(col) == len(pos) else None

        geo = o3d.geometry.PointCloud()
        geo.points = o3d.utility.Vector3dVector(roi_pos)
        if roi_col is not None:
            geo.colors = o3d.utility.Vector3dVector(roi_col.astype(float))

        vsize = float(getattr(Config, 'DEFAULT_VOXEL_SIZE', 0.05))

        detect_kwargs = (dataset.metadata if dataset is not None else {}) or {}
        detect_fn = detect_facades_adaptive
        result = detect_fn(
            geo, voxel_size=vsize,
            min_facade_area=float(getattr(Config, 'MIN_FACADE_AREA', min_facade_area or 10.0)),
            roi_indices=(np.arange(len(roi_pos), dtype=np.int64)
                         if roi_scope == 'clip' else None),
            roi_bounds=None, enable_grow=False,
            metadata=detect_kwargs,
            scan_origin=detect_kwargs.get('scan_origins'),
        )
        facades = result.get('facades', []) if isinstance(result, dict) else (result or [])

        if roi_scope == 'seed' and len(seed_indices_local):
            seed_set = set(seed_indices_local.tolist())
            facades = [f for f in facades if seed_set.intersection(
                np.asarray(f.get('inlier_indices', []), dtype=np.int64).tolist()
            )]
        trace("facade.roi.algorithm_done", proxy_points=len(roi_pos),
              facades=len(facades), scope=roi_scope,
              seconds=f"{time.perf_counter()-started:.2f}")

        if self._index_service is not None:
            roi_proxy_ids = self._index_service.proxy_ids_for_positions(cloud_name, roi_pos, global_indices)
            self._index_service.normalize_facade_indices(facades, roi_proxy_ids)
            self._index_service.populate_voxel_ids(facades, cloud_name)

        for facade in facades:
            trace("facade.roi.mapping", facade=facade.get('id'),
                  proxy=len(facade.get('proxy_indices', [])),
                  voxels=facade.get('voxel_count', 0))

        if project_uuid:
            self._persist_results(project_uuid, facades)

        trace("facade.roi.done", cloud=cloud_name, facades=len(facades),
              seconds=f"{time.perf_counter()-started:.2f}")
        return facades

    def _persist_results(self, project_uuid: str, facades: list[dict]) -> None:
        try:
            ResultsRepo.save_detected_facades(project_uuid, facades)
        except Exception as e:
            print(f'FacadeDetectionService: 持久化失败: {e}', flush=True)
