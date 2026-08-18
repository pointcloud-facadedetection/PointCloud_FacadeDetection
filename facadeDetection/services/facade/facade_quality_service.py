"""立面质量评估业务编排服务。

仅负责质量评估流程调度、参数组装与结果导出；不直接操作 UI 控件，也不包含
底层质量统计算法实现。
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import open3d as o3d

from algorithms.facade.quality import compute_facade_quality
from config.settings import Config
from services.facade.facade_index_service import FacadeIndexService
from utils.logging_utils import trace


class FacadeQualityService:
    """外立面质量评估业务编排服务。"""

    def __init__(self, pointcloud_service=None, index_service: Optional[FacadeIndexService] = None):
        self._pointcloud_service = pointcloud_service
        self._index_service = index_service

    def set_index_service(self, index_service: FacadeIndexService):
        self._index_service = index_service

    def _get_source_points(self, cloud_name: str):
        """返回质量评估使用的点源与原始点数组。"""
        pcs = self._pointcloud_service
        if pcs is None or self._index_service is None:
            return None, None
        dataset = self._index_service._get_dataset(cloud_name)
        if dataset is None:
            return None, None
        metadata = dataset.metadata or {}
        source_id = metadata.get('source_id')
        source_asset = pcs.get_source_asset(source_id) if source_id else None
        if source_asset is not None:
            return source_asset['points'], dataset.raw.points
        return dataset.raw.points, dataset.raw.points

    def compute_quality(self, cloud_name: str, facade: dict,
                        grid_size: float | None = None,
                        flatness_limit: float = 0.004,
                        verticality_limit_mm: float = 4.0,
                        ruler_size: float | None = None,
                        ruler_step: float | None = None,
                        profile=None, results_dir=None) -> Optional[dict]:
        """质量评估计算（基于 VoxelCascadeIndex 体素级联映射）。"""
        started = time.perf_counter()
        facade_id = facade.get('id', 0)
        trace("quality.begin", cloud=cloud_name, facade=facade_id,
              proxy=len(facade.get('proxy_indices', facade.get('inlier_indices', [])) or []))

        try:
            if profile is not None:
                flatness_limit = profile.flatness_limit_mm / 1000.0
                verticality_limit_mm = profile.verticality_limit_mm
                ruler_size = profile.window_size_m
                ruler_step = profile.step_size_m

            if self._index_service is None:
                trace("quality.error", facade=facade_id, reason="index_service_unavailable")
                return None

            quality_indices, domain_stats = self._index_service.build_quality_domain(facade, cloud_name)

            if len(quality_indices) == 0:
                trace("quality.reject", facade=facade_id, reason="empty_quality_domain")
                return None

            n_valid = int(len(quality_indices))
            trace("quality.domain", facade=facade_id,
                  raw_before=domain_stats.get('raw_before_filter', n_valid),
                  raw_after=n_valid,
                  main_depth=domain_stats.get('main_depth_count', n_valid),
                  defects=domain_stats.get('defect_count', 0),
                  voxels=domain_stats.get('voxel_count', 0),
                  limit_mm=f"{domain_stats.get('distance_limit_mm', 0):.1f}",
                  p99_mm=f"{domain_stats.get('signed_dist_p99_mm', 0):.2f}")

            if n_valid < int(getattr(profile, 'min_points', 3)):
                trace("quality.reject", facade=facade_id,
                      reason=f"too_few_points:{n_valid}")
                return None

            point_source, raw_points = self._get_source_points(cloud_name)
            if point_source is None:
                trace("quality.error", facade=facade_id, reason="point_source_unavailable")
                return None
            filtered_pts = np.asarray(point_source)[quality_indices]

            plane_model = np.asarray(facade.get('plane_model', []), dtype=np.float64)
            if plane_model.shape[0] != 4:
                trace("quality.reject", facade=facade_id, reason="missing_plane_model")
                return None

            norm = np.linalg.norm(plane_model[:3])
            if not np.isfinite(norm) or norm < 1e-12:
                trace("quality.reject", facade=facade_id, reason="degenerate_plane")
                return None
            plane_model = plane_model / norm

            facade_ref = {
                'plane_model': [float(x) for x in plane_model],
                'center': [float(x) for x in np.mean(filtered_pts, axis=0)],
            }

            trace("quality.domain_ready", facade=facade_id,
                  quality_points=n_valid, seconds=f"{time.perf_counter()-started:.2f}")

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(filtered_pts.astype(float))

            gsize = float((profile.interval_size_m if profile is not None
                           else getattr(Config, 'QUALITY_GRID_SIZE', 20.0))
                          if grid_size is None else grid_size)
            rsize = float(getattr(Config, 'RULER_SIZE', 0.05) if ruler_size is None else ruler_size)
            rstep = float(getattr(Config, 'RULER_STEP', 0.05) if ruler_step is None else ruler_step)

            trace("quality.algorithm", facade=facade_id,
                  points=n_valid, grid=gsize, window=rsize, step=rstep)

            algo_started = time.perf_counter()
            result = compute_facade_quality(
                facade_info=facade_ref, pcd=pcd, grid_size=gsize,
                flatness_limit=float(flatness_limit),
                verticality_limit_mm=float(verticality_limit_mm),
                ruler_size=rsize, ruler_step=rstep,
                min_points=int(getattr(profile, 'min_points', 3)),
                measure_height=float(getattr(profile, 'measure_height_m', 2.0)),
            )
            algo_elapsed = time.perf_counter() - algo_started

            n_windows = len(result.get('windows', {}).get('window_id', []))
            n_intervals = len(result.get('intervals', []))
            z_span = float((result.get('overall') or {}).get('global_z_span_m', 0.0))
            trace("quality.algorithm_done", facade=facade_id,
                  windows=n_windows, intervals=n_intervals, z_span=f"{z_span:.1f}m",
                  seconds=f"{algo_elapsed:.2f}")

            result = dict(result)
            result['__global_indices'] = np.asarray(quality_indices, dtype=np.int32)
            result['__index_space'] = 'facade_local_to_raw_global'
            result['quality_domain'] = dict(domain_stats)

            voxel_ids = facade.get('voxel_ids')
            if voxel_ids is not None:
                result['quality_domain']['voxel_ids'] = voxel_ids
                result['quality_domain']['voxel_count'] = len(voxel_ids)
            result['quality_domain']['proxy_count'] = len(facade.get('proxy_indices', []))

            if profile is not None:
                result['profile_snapshot'] = profile.snapshot()

            if results_dir:
                try:
                    from services.result_export_service import ResultExportService
                    source_asset = None
                    if self._pointcloud_service is not None:
                        metadata = (self._index_service._get_dataset(cloud_name).metadata or {}) if self._index_service else {}
                        source_id = metadata.get('source_id')
                        source_asset = self._pointcloud_service.get_source_asset(source_id) if source_id else None
                    if source_asset is not None and source_asset.get('colors') is not None:
                        colors = source_asset['colors'][quality_indices]
                    else:
                        colors = (raw_points.colors[quality_indices]
                                  if hasattr(raw_points, 'colors') and raw_points.colors is not None else None)
                    result['__points_index_space'] = 'facade_local_raw'
                    result['artifacts'] = ResultExportService().export_facade(
                        results_dir, facade_id, filtered_pts, colors, result)
                    trace("quality.export_done", facade=facade_id)
                except Exception as export_error:
                    result['export_error'] = str(export_error)
                    trace("quality.export_error", facade=facade_id, error=str(export_error))

            total_elapsed = time.perf_counter() - started
            trace("quality.finish", facade=facade_id, total_seconds=f"{total_elapsed:.2f}")
            return result

        except Exception as e:
            trace("quality.error", facade=facade.get('id', 0),
                  seconds=f"{time.perf_counter()-started:.2f}", error=repr(e))
            print(f'立面质量评估失败: {e}', flush=True)
            return None
