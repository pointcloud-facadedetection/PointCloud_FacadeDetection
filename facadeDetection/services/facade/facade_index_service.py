"""立面索引映射服务：负责 proxy / voxel / raw 三级索引转换与质量域构建。

本层只处理数据映射与索引计算，不直接操作 UI 控件，也不包含立面检测或
质量评估算法实现。
"""
from __future__ import annotations

import numpy as np

from utils.array_utils import as_array, valid_ids
from utils.logging_utils import trace
from config.settings import Config


class FacadeIndexService:
    """基于 VoxelCascadeIndex 的立面索引映射服务。"""

    def __init__(self, pointcloud_service=None):
        self._pointcloud_service = pointcloud_service

    def set_pointcloud_service(self, pointcloud_service):
        self._pointcloud_service = pointcloud_service

    def _get_dataset(self, cloud_name: str):
        """通过点云服务获取数据集。"""
        if self._pointcloud_service is None:
            return None
        vp = getattr(self._pointcloud_service, 'viewport', None)
        if vp is not None:
            data = vp.get_cloud_data(cloud_name)
            if data is not None:
                dataset_id = data.get('dataset_id')
                if dataset_id:
                    return self._pointcloud_service.get_dataset(dataset_id)
        return None

    def proxy_ids_for_positions(self, cloud_name: str, positions, fallback_rows=None):
        """坐标 → proxy 行号（通过 VoxelCascadeIndex 二分查找）。"""
        dataset = self._get_dataset(cloud_name)
        if dataset is None:
            if fallback_rows is not None:
                return np.asarray(fallback_rows, dtype=np.int32).reshape(-1)
            return np.arange(len(positions), dtype=np.int32)
        return dataset.index.proxy_ids_for_points(np.asarray(positions, dtype=np.float32))

    def normalize_facade_indices(self, facades: list[dict], row_to_proxy: np.ndarray):
        """将立面索引从局部空间映射到 proxy 全局空间。"""
        row_to_proxy = np.asarray(row_to_proxy, dtype=np.int64).reshape(-1)
        fields = ('inlier_indices', 'support_indices')

        for facade in facades:
            for key in fields:
                local = valid_ids(facade.get(key), len(row_to_proxy), np.int32)
                mapped = row_to_proxy[local]
                facade[key] = mapped[mapped >= 0].astype(np.int32).tolist()

            facade['proxy_indices'] = list(facade.get('inlier_indices', []))
            facade['facade_proxy_indices'] = list(facade['inlier_indices'])
            facade['proxy_count'] = len(facade['facade_proxy_indices'])
            facade['__index_space'] = 'proxy_global'

    def populate_voxel_ids(self, facades: list[dict], cloud_name: str):
        """通过 dataset.index 填充 voxel_ids。"""
        dataset = self._get_dataset(cloud_name)
        if dataset is None:
            return
        index = dataset.index
        for facade in facades:
            proxy_indices = as_array(
                facade.get('proxy_indices', facade.get('inlier_indices', [])),
                dtype=np.int32
            )
            voxel_ids = index.proxy_to_voxel_ids(proxy_indices)
            voxel_ids = np.unique(voxel_ids[voxel_ids >= 0])
            facade['voxel_ids'] = voxel_ids.tolist()
            facade['voxel_count'] = len(voxel_ids)

    def build_quality_domain(self, facade: dict, cloud_name: str) -> tuple[np.ndarray, dict]:
        """
        构建质量域：通过 VoxelCascadeIndex 将代理空间立面映射到原始空间。

        【关键修复】距离分层下采样后，使用 source_raw_ids 直接映射到原始高保真点，
        不再依赖体素坐标反查（避免 1e-4 超细体素导致的 1:1 映射退化）。
        """
        dataset = self._get_dataset(cloud_name)
        if dataset is None:
            return np.empty(0, np.int32), {'error': 'dataset_unavailable'}

        index = dataset.index
        metadata = dataset.metadata or {}

        # === 修复：优先使用 source 原始点进行质量计算 ===
        source_id = metadata.get('source_id')
        pcs = self._pointcloud_service
        source_asset = None
        source_points = None
        source_colors = None

        if pcs is not None and source_id:
            source_asset = pcs.get_source_asset(source_id)

        # 获取 facade 的代理索引
        proxy_indices = as_array(
            facade.get('proxy_indices', facade.get('inlier_indices', [])),
            dtype=np.int32
        )
        if len(proxy_indices) == 0:
            return np.empty(0, np.int32), {'error': 'no_proxy_indices'}

        # === 核心修复：使用 source 映射直接获取原始点索引 ===
        if index.has_source_mapping():
            # ✅ 正确路径：proxy -> source_raw_ids -> 原始高保真点
            raw_indices = index.proxy_to_source_ids(proxy_indices, deduplicate=True)
            source_points = index.get_source_points()

            # 获取原始点颜色（如果有）
            if source_asset is not None and source_asset.get('colors') is not None:
                source_colors = source_asset['colors']

            trace("quality.domain.source_mapping", 
                  proxy=len(proxy_indices), source_raw=len(raw_indices),
                  source_points=len(source_points) if source_points is not None else 0)
        else:
            # 退化路径：没有 source 映射，使用 processed 点
            voxel_ids = facade.get('voxel_ids')
            if voxel_ids is None or len(voxel_ids) == 0:
                voxel_ids = index.proxy_to_voxel_ids(proxy_indices)
                voxel_ids = np.unique(voxel_ids[voxel_ids >= 0])

            voxel_ids = np.asarray(voxel_ids, dtype=np.int32)
            if len(voxel_ids) == 0:
                return np.empty(0, np.int32), {'error': 'no_voxel_ids'}

            raw_count = index.raw_count_for_voxels(voxel_ids)
            trace("quality.domain.raw_count", voxels=len(voxel_ids), raw_count=int(raw_count))

            if raw_count <= 5_000_000:
                raw_indices = index.voxel_to_raw_ids(voxel_ids, deduplicate=True)
            else:
                chunks = []
                for chunk in index.iter_raw_ids(voxel_ids, chunk_size=1_000_000):
                    chunks.append(chunk)
                raw_indices = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int32)
                raw_indices = np.unique(raw_indices)

            source_points = np.asarray(dataset.raw.points, dtype=np.float64)
            if dataset.raw.colors is not None:
                source_colors = np.asarray(dataset.raw.colors, dtype=np.float64)

        if len(raw_indices) == 0:
            return np.empty(0, np.int32), {'error': 'no_raw_indices'}

        # === 使用原始点进行平面距离计算 ===
        pts = source_points[raw_indices] if source_points is not None else np.empty((0, 3))

        plane_model = np.asarray(facade.get('plane_model', []), dtype=np.float64)
        if plane_model.shape[0] != 4:
            return raw_indices, {'warning': 'invalid_plane_model', 'raw_count': len(raw_indices)}

        norm = np.linalg.norm(plane_model[:3])
        if norm < 1e-12:
            return raw_indices, {'warning': 'degenerate_plane', 'raw_count': len(raw_indices)}

        plane_model = plane_model / norm
        n = plane_model[:3]
        d = float(plane_model[3])

        signed_dist = pts @ n + d

        voxel_size = float(getattr(Config, 'DEFAULT_VOXEL_SIZE', 0.05))
        detect_tol = float(getattr(Config, 'DETECT_DIST_TOL_MM', 20.0)) / 1000.0
        voxel_error = voxel_size * np.sqrt(3.0)

        median_dist = float(np.median(signed_dist)) if len(signed_dist) else 0.0
        mad_dist = float(np.median(np.abs(signed_dist - median_dist))) if len(signed_dist) else 0.0
        robust_sigma = 1.4826 * mad_dist

        base_limit = max(
            float(getattr(Config, 'FACADE_QUALITY_DEPTH_MIN_M', 0.02)),
            voxel_error + detect_tol + float(getattr(Config, 'FACADE_QUALITY_DEPTH_MULT', 3.0)) * robust_sigma,
        )

        depth_span = float(np.max(signed_dist) - np.min(signed_dist)) if len(signed_dist) > 0 else 0.0
        adaptive_max = max(
            float(getattr(Config, 'FACADE_QUALITY_DEPTH_MAX_M', 0.50)),
            min(depth_span * 0.5, 2.0)
        )
        distance_limit = min(base_limit, adaptive_max)

        trace("quality.domain.depth",
              base_limit=f"{base_limit:.3f}",
              adaptive_max=f"{adaptive_max:.3f}",
              final_limit=f"{distance_limit:.3f}",
              depth_span=f"{depth_span:.3f}")

        main_mask = np.abs(signed_dist - median_dist) <= distance_limit
        defect_mask = (~main_mask) & (np.abs(signed_dist - median_dist) <= distance_limit * 2.0)
        valid_mask = main_mask | defect_mask
        quality_indices = raw_indices[valid_mask]

        main_count = int(np.sum(main_mask))
        defect_count = int(np.sum(defect_mask))
        filtered_out = int(len(raw_indices) - len(quality_indices))

        stats = {
            'voxel_count': int(facade.get('voxel_count', 0)),
            'proxy_count': int(len(proxy_indices)),
            'raw_before_filter': int(len(raw_indices)),
            'raw_after_filter': int(len(quality_indices)),
            'main_depth_count': main_count,
            'defect_count': defect_count,
            'filtered_out': filtered_out,
            'distance_limit_mm': float(distance_limit * 1000.0),
            'adaptive_max_mm': float(adaptive_max * 1000.0),
            'depth_span_mm': float(depth_span * 1000.0),
            'signed_dist_median_mm': float(median_dist * 1000.0),
            'signed_dist_mad_mm': float(mad_dist * 1000.0),
            'signed_dist_mean_mm': float(np.mean(signed_dist) * 1000.0) if len(signed_dist) else 0.0,
            'signed_dist_std_mm': float(np.std(signed_dist) * 1000.0) if len(signed_dist) else 0.0,
            'signed_dist_p99_mm': float(np.percentile(np.abs(signed_dist), 99) * 1000.0) if len(signed_dist) else 0.0,
            'source_mapping': index.has_source_mapping(),
        }

        if index.has_source_mapping():
            stats['quality_source'] = 'fls_source_raw'
            stats['source_raw_count'] = int(len(quality_indices))
        else:
            stats['quality_source'] = 'processed_raw'
            stats['processed_raw_count'] = int(len(quality_indices))

        return quality_indices, stats

    def map_roi_to_raw(self, cloud_name: str, roi_proxy_indices: np.ndarray) -> np.ndarray:
        """将代理空间的 ROI 决策结果映射回原始空间。"""
        dataset = self._get_dataset(cloud_name)
        if dataset is None:
            return np.empty(0, dtype=np.int32)

        # === 修复：优先使用 source 映射 ===
        if dataset.index.has_source_mapping():
            return dataset.index.proxy_to_source_ids(roi_proxy_indices, deduplicate=True)
        return dataset.index.proxy_to_raw_ids(roi_proxy_indices, deduplicate=True)

    def map_raw_to_proxy(self, cloud_name: str, raw_indices: np.ndarray) -> np.ndarray:
        """将原始空间索引映射到代理空间。"""
        dataset = self._get_dataset(cloud_name)
        if dataset is None:
            return np.empty(0, dtype=np.int32)
        return dataset.index.raw_to_voxel_ids(raw_indices)

    def get_raw_count_for_facade(self, cloud_name: str, facade: dict) -> int:
        """获取立面对应的原始点数量（不分配内存）。"""
        dataset = self._get_dataset(cloud_name)
        if dataset is None:
            return 0
        proxy_indices = as_array(
            facade.get('proxy_indices', facade.get('inlier_indices', [])),
            dtype=np.int32
        )
        # === 修复：优先使用 source 映射计数 ===
        if dataset.index.has_source_mapping():
            return dataset.index.source_count_for_proxy(proxy_indices)
        return dataset.index.raw_count_for_proxy(proxy_indices)