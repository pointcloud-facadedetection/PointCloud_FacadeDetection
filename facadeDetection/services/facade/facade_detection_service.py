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

from algorithms.facade.facade_detection import detect_facades_adaptive
from config.settings import Config
from services import proxy_cache
from services.dal.results_repo import ResultsRepo
from services.facade.facade_index_service import FacadeIndexService
from utils.logging_utils import trace

log = logging.getLogger("facadeDetection.facade")


def _estimate_geo_normals(geo, voxel_size):
    """大点云代理法向估计：参数与 ensure_normals 估计分支完全一致
    （radius=max(voxel_size*4, 0.2), max_nn=50），保证首次走估计路径的
    检测结果与优化前逐点一致；估计原值（float64）由调用方缓存复用。"""
    geo.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=max(float(voxel_size) * 4.0, 0.2), max_nn=50))


class FacadeDetectionService:
    """外立面检测业务编排服务"""

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
        if not dataset_id and service is not None and hasattr(service, 'bind_processing_cloud'):
            dataset_id = service.bind_processing_cloud(cloud_name)
        dataset = service.get_dataset(dataset_id) if service and dataset_id else None
        return data, dataset

    def _ensure_proxy_normals(self, dataset, proxy_pts, vsize):
        """确保 dataset.proxy_normals 与完整代理点云一一对应。

        缓存命中（长度一致）直接返回；缺失时在完整 proxy_pts 上按与 detect()
        完全一致的参数估计一次，写回 dataset.proxy_normals 供后续复用。

        return (normals, estimated_large)：
            normals:         Nx3 ndarray，dtype float32（小点云）或 float64（大点云）
            estimated_large: 是否走了大点云估计分支，供持久化 proxy 缓存判断
        """
        cached = getattr(dataset, 'proxy_normals', None)
        if cached is not None and len(cached) == len(proxy_pts):
            return cached, False

        started = time.perf_counter()
        geo = o3d.geometry.PointCloud()
        geo.points = o3d.utility.Vector3dVector(proxy_pts.astype(float))

        estimated_large = False
        if len(proxy_pts) < 500000:
            # 兜底：估计法向（仅对小数据）
            geo.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=max(float(Config.DEFAULT_VOXEL_SIZE) * 4, 0.2),
                    max_nn=30
                )
            )
            normals = np.asarray(geo.normals, dtype=np.float32)
        else:
            _estimate_geo_normals(geo, vsize)
            normals = np.asarray(geo.normals, dtype=np.float64)
            estimated_large = True

        dataset.proxy_normals = normals
        trace("facade.normals.estimated", points=len(proxy_pts),
              large=estimated_large,
              seconds=f"{time.perf_counter()-started:.2f}")
        return normals, estimated_large

    @staticmethod
    def _select_top_facades(facades: list[dict]) -> list[dict]:
        """按最终立面点数稳定选取前 N 个，不改变单个立面的业务索引字段。"""
        limit = int(getattr(Config, 'MAX_FACADE_COUNT', 40) or 0)
        if limit <= 0 or len(facades or []) <= limit:
            return facades or []
        return sorted(
            facades or [],
            key=lambda f: (
                -int(f.get('point_count', 0) or 0),
                -float(f.get('area', 0.0) or 0.0),
                int(f.get('id', 0) or 0),
            ),
        )[:limit]

    def detect(self, cloud_name: str, project_uuid: Optional[str] = None) -> list[dict]:
        """全局立面检测"""
        started = time.perf_counter()
        trace("facade.detect.begin", cloud=cloud_name)

        # 获取数据集（含 VoxelCascadeIndex 和 source_raw_ids）
        data, dataset = self._get_dataset(cloud_name)
        if dataset is None:
            trace("facade.detect.error", cloud=cloud_name, reason="no_dataset")
            return []

        proxy_pts = dataset.proxy_points
        if len(proxy_pts) == 0:
            trace("facade.detect.error", cloud=cloud_name, reason="empty_proxy")
            return []

        vsize = float(getattr(Config, 'DEFAULT_VOXEL_SIZE', 0.05))

        # 统一入口：缓存命中直接复用；缺失时在完整 proxy 上估计一次并写回 dataset
        proxy_normals, estimated_large = self._ensure_proxy_normals(
            dataset, proxy_pts, vsize)

        geo = o3d.geometry.PointCloud()
        geo.points = o3d.utility.Vector3dVector(proxy_pts.astype(float))
        geo.normals = o3d.utility.Vector3dVector(
            np.asarray(proxy_normals, dtype=float))

        min_area = float(getattr(Config, 'MIN_FACADE_AREA', 10.0))

        detect_kwargs = dataset.metadata or {}

        result = detect_facades_adaptive(
            geo,
            dataset=dataset,
            voxel_size=vsize,
            min_facade_area=min_area,
            metadata=detect_kwargs,
            scan_origin=detect_kwargs.get('scan_origins'),
            normals=dataset.proxy_normals,
        )

        facades = result.get('facades', []) if isinstance(result, dict) else (result or [])
        detected_count = len(facades)
        facades = self._select_top_facades(facades)

        # 索引归一化：facade 的 inlier_indices 当前是 proxy 局部索引
        # 需要映射到 proxy_global，再映射到 raw（若需要）
        if self._index_service is not None:
            # proxy 局部 → proxy_global
            proxy_ids = np.arange(len(proxy_pts), dtype=np.int32)
            self._index_service.normalize_facade_indices(facades, proxy_ids)
            self._index_service.populate_voxel_ids(facades, cloud_name)

        # 每个已持久化/渲染的结果都与具体的处理数据集相关联。
        revision = getattr(dataset, 'revision', str(dataset.dataset_id))
        station_id = (data or {}).get('station_id') or (dataset.metadata or {}).get('station_id')
        if project_uuid and station_id is None:
            raise ValueError('当前处理点云未绑定站点，拒绝保存立面结果')
        if estimated_large and project_uuid and station_id is not None:
            # 代理法向对同一资产指纹是确定的，落盘后重开项目直接复用；
            # 指纹缺失或长度校验失败时 save_proxy_normals 仅告警跳过
            fingerprint = (dataset.metadata or {}).get('asset_fingerprint')
            if fingerprint:
                proxy_cache.save_proxy_normals(
                    project_uuid, station_id, tuple(fingerprint),
                    dataset.proxy_normals)
        for facade in facades:
            facade['dataset_id'] = dataset.dataset_id
            facade['dataset_revision'] = revision
            facade['station_id'] = int(station_id) if station_id is not None else None
            facade['dataset_fingerprint'] = (dataset.metadata or {}).get('asset_fingerprint')
            facade['index_space'] = 'proxy_global'
            facade['review_status'] = facade.get('review_status',
                                                 facade.get('preview_status', 'pending'))
            facade['quality_ready'] = facade['review_status'] == 'complete'
            if self._index_service is not None:
                facade['raw_point_count'] = self._index_service.get_raw_count_for_facade(
                    cloud_name, facade)
            else:
                facade['raw_point_count'] = int(facade.get('point_count', 0) or 0)

        if project_uuid:
            ResultsRepo.save_detected_facades(project_uuid, facades)

        trace("facade.detect.done", cloud=cloud_name, facades=len(facades),
              detected=detected_count,
              dropped=max(0, detected_count - len(facades)),
              proxy_points=len(proxy_pts), seconds=f"{time.perf_counter()-started:.2f}")

        # Detection is a working-set operation.  It must not create a
        # historical scene before quality evaluation has succeeded.
        return facades

    def detect_on_roi(self, cloud_name: str,
                      roi_indices: list[int] | None = None,
                      roi_bounds: tuple | None = None,
                      roi_box: dict | None = None,
                      project_uuid: Optional[str] = None,
                      min_facade_area: float | None = None,
                      roi_scope: str = 'seed') -> list[dict]:
        """ROI 立面检测"""
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

        data, dataset = self._get_dataset(cloud_name)
        if dataset is None:
            trace("facade.roi.error", cloud=cloud_name, reason="no_dataset")
            return []

        proxy_pts = dataset.proxy_points
        n_proxy = len(proxy_pts)
        if n_proxy == 0:
            return []

        vsize = float(getattr(Config, 'DEFAULT_VOXEL_SIZE', 0.05))

        proxy_normals, estimated_large = self._ensure_proxy_normals(
            dataset, proxy_pts, vsize)

        seed_mask = np.ones(n_proxy, dtype=bool)
        if roi_box is not None:
            center = np.asarray(roi_box['center'], dtype=float)
            axes = np.asarray(roi_box['axes'], dtype=float).reshape(3, 3)
            half = np.asarray(roi_box['half_extent'], dtype=float)
            local = (proxy_pts - center) @ axes.T
            seed_mask = np.all(np.abs(local) <= half + 1e-9, axis=1)
        elif roi_bounds is not None:
            bmin = np.asarray(roi_bounds[0], dtype=float)
            bmax = np.asarray(roi_bounds[1], dtype=float)
            seed_mask = np.all((proxy_pts >= bmin) & (proxy_pts <= bmax), axis=1)
        elif roi_indices is not None and len(roi_indices):
            seed_mask = np.zeros(n_proxy, dtype=bool)
            valid = np.asarray(roi_indices, dtype=int)
            seed_mask[valid[(valid >= 0) & (valid < n_proxy)]] = True

        # clip 模式：只取 ROI 内点；seed 模式：全点检测，后过滤
        crop_mask = seed_mask if roi_scope == 'clip' else np.ones(n_proxy, dtype=bool)
        global_indices = np.flatnonzero(crop_mask).astype(np.int64)
        seed_indices_local = np.flatnonzero(seed_mask).astype(np.int64)

        trace("facade.roi.selected", proxy_points=n_proxy,
              selected=len(global_indices), roi_seed=len(seed_indices_local),
              seconds=f"{time.perf_counter()-started:.2f}")

        if len(global_indices) == 0:
            return []

        # 构建 ROI 子集 o3d.PointCloud
        roi_pos = proxy_pts[global_indices]
        geo = o3d.geometry.PointCloud()
        geo.points = o3d.utility.Vector3dVector(roi_pos.astype(float))

        # ROI 法向：直接按索引切分完整 proxy 缓存，与全局检测法向逐点一致；
        # detect_facades_adaptive 命中 skip_ensure 分支，完全跳过 KD-tree 估计。
        roi_normals = np.asarray(proxy_normals, dtype=float)[global_indices]
        geo.normals = o3d.utility.Vector3dVector(roi_normals)

        detect_kwargs = dataset.metadata or {}

        result = detect_facades_adaptive(
            geo,
            dataset=dataset,
            voxel_size=vsize,
            min_facade_area=float(getattr(Config, 'MIN_FACADE_AREA', min_facade_area or 10.0)),
            roi_indices=(np.arange(len(roi_pos), dtype=np.int64)
                         if roi_scope == 'clip' else None),
            roi_bounds=None,
            enable_grow=False,
            metadata=detect_kwargs,
            scan_origin=detect_kwargs.get('scan_origins'),
            normals=roi_normals,
        )
        facades = result.get('facades', []) if isinstance(result, dict) else (result or [])

        # seed 模式：只保留与 ROI 相交的 facade
        if roi_scope == 'seed' and len(seed_indices_local):
            seed_set = set(seed_indices_local.tolist())
            facades = [f for f in facades if seed_set.intersection(
                np.asarray(f.get('inlier_indices', []), dtype=np.int64).tolist()
            )]

        detected_count = len(facades)
        facades = self._select_top_facades(facades)

        trace("facade.roi.algorithm_done", proxy_points=len(roi_pos),
              facades=len(facades), detected=detected_count,
              dropped=max(0, detected_count - len(facades)), scope=roi_scope,
              seconds=f"{time.perf_counter()-started:.2f}")

        # 索引归一化
        if self._index_service is not None:
            # ROI 局部 → 全局 proxy 索引
            roi_proxy_ids = global_indices.astype(np.int32)
            self._index_service.normalize_facade_indices(facades, roi_proxy_ids)
            self._index_service.populate_voxel_ids(facades, cloud_name)

        revision = getattr(dataset, 'revision', str(dataset.dataset_id))
        station_id = (data or {}).get('station_id') or (dataset.metadata or {}).get('station_id')
        # ROI 首次触发时若已估计大点云法向且具备持久化条件，则落盘 proxy 缓存，
        # 让项目重开后全局/ROI 检测都直接命中缓存。
        if estimated_large and project_uuid and station_id is not None:
            fingerprint = (dataset.metadata or {}).get('asset_fingerprint')
            if fingerprint:
                proxy_cache.save_proxy_normals(
                    project_uuid, station_id, tuple(fingerprint),
                    dataset.proxy_normals)

        for facade in facades:
            facade['dataset_id'] = dataset.dataset_id
            facade['dataset_revision'] = revision
            facade['station_id'] = int(station_id) if station_id is not None else None
            facade['dataset_fingerprint'] = (dataset.metadata or {}).get('asset_fingerprint')
            facade['index_space'] = 'proxy_global'
            facade['review_status'] = facade.get('review_status',
                                                 facade.get('preview_status', 'pending'))
            facade['quality_ready'] = facade['review_status'] == 'complete'
            if self._index_service is not None:
                facade['raw_point_count'] = self._index_service.get_raw_count_for_facade(
                    cloud_name, facade)
            else:
                facade['raw_point_count'] = int(facade.get('point_count', 0) or 0)

        if project_uuid:
            ResultsRepo.save_detected_facades(project_uuid, facades)

        # for facade in facades:
        #     trace("facade.roi.mapping", facade=facade.get('id'),
        #           proxy=len(facade.get('proxy_indices', [])),
        #           voxels=facade.get('voxel_count', 0))

        # ROI detection is also process data; persistence is committed by the
        # quality-result transaction, not by this method.

        trace("facade.roi.done", cloud=cloud_name, facades=len(facades),
              seconds=f"{time.perf_counter()-started:.2f}")
        return facades

    def _persist_results(self, project_uuid: str, facades: list[dict]) -> None:
        try:
            ResultsRepo.save_detected_facades(project_uuid, facades)
        except Exception as e:
            print(f'FacadeDetectionService: 持久化失败: {e}', flush=True)