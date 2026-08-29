"""立面质量评估业务编排服务。

仅负责质量评估流程调度、参数组装与结果返回；不直接操作 UI 控件，也不包含
底层质量统计算法实现，不执行任何文件导出。
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
from algorithms.facade.ruler_quality import RulerQualityParameters, compute_ruler_quality
from algorithms.geometry import classify_plane, plane_axes
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
        points = dataset.processed_raw_points
        colors = dataset.index.get_source_colors()
        return points, colors

    def compute_quality(self, cloud_name: str, facade: dict,
                        grid_size: float | None = None,
                        flatness_limit: float = 0.004,
                        verticality_limit_mm: float = 4.0,
                        ruler_size: float | None = None,
                        ruler_step: float | None = None,
                        profile=None, results_dir=None) -> Optional[dict]:
        """质量评估计算,仅返回计算结果，不执行任何文件导出。"""
        started = time.perf_counter()
        facade_no = int(facade.get('display_no', facade.get('id', 0)))

        trace("quality.begin", cloud=cloud_name, facade_no=facade_no,
              proxy=len(facade.get('proxy_indices', facade.get('inlier_indices', [])) or []))

        try:
            if profile is not None:
                flatness_limit = profile.flatness_limit_mm / 1000.0
                verticality_limit_mm = profile.verticality_limit_mm
                ruler_size = float(getattr(profile, 'measure_height_m', ruler_size or 2.0))
                ruler_step = float(getattr(profile, 'scan_step_m', ruler_step or 0.05))

            if self._index_service is None:
                trace("quality.error", facade_no=facade_no, 
                      reason="index_service_unavailable")
                return None

            dataset = self._index_service._get_dataset(cloud_name)
            if dataset is None:
                return {'ok': False, 'reason': 'dataset_unavailable'}
            result_revision = facade.get('dataset_revision')
            if result_revision and result_revision != dataset.revision:
                trace('quality.reject', facade_no=facade_no,
                      reason='stale_facade_result', result_revision=result_revision,
                      dataset_revision=dataset.revision)
                return {'ok': False, 'reason': 'stale_facade_result',
                        'message': '立面结果属于旧处理版本，请重新执行立面检测。'}

            quality_indices, domain_stats = self._index_service.build_quality_domain(facade, cloud_name)

            if len(quality_indices) == 0:
                trace("quality.reject", facade_no=facade_no, 
                      reason="empty_quality_domain")
                return None

            n_valid = int(len(quality_indices))
            trace("quality.domain", facade_no=facade_no,
                  raw=n_valid,
                  voxels=domain_stats.get('voxel_count', 0),
                  source_mapping=domain_stats.get('source_mapping', False))

            if n_valid < int(getattr(profile, 'min_points', 3)):
                trace("quality.reject", facade_no=facade_no,
                      reason=f"too_few_points:{n_valid}")
                return None

            point_source, point_colors = self._get_source_points(cloud_name)
            if point_source is None:
                trace("quality.error", facade_no=facade_no, 
                      reason="point_source_unavailable")
                return None
            filtered_pts = np.asarray(point_source)[quality_indices]

            # Validate points
            if not np.all(np.isfinite(filtered_pts)):
                n_invalid = int(np.sum(~np.all(np.isfinite(filtered_pts), axis=1)))
                trace("quality.warn", facade_no=facade_no, 
                      reason=f"filtered_pts_contains_nan:{n_invalid}")
                valid_mask = np.all(np.isfinite(filtered_pts), axis=1)
                if not np.any(valid_mask):
                    trace("quality.reject", facade_no=facade_no, 
                          reason="all_filtered_pts_invalid")
                    return None
                filtered_pts = filtered_pts[valid_mask]
                quality_indices = quality_indices[valid_mask]
                n_valid = len(filtered_pts)

            plane_model = np.asarray(facade.get('plane_model', []), dtype=np.float64)
            if plane_model.shape[0] != 4:
                trace("quality.reject", facade_no=facade_no, 
                      reason="missing_plane_model")
                return None

            norm = np.linalg.norm(plane_model[:3])
            if not np.isfinite(norm) or norm < 1e-12:
                trace("quality.reject", facade_no=facade_no, 
                      reason="degenerate_plane")
                return None
            plane_model = plane_model / norm

            # Use stable plane origin
            bbox_min = np.min(filtered_pts, axis=0)
            bbox_max = np.max(filtered_pts, axis=0)
            bbox_center = (bbox_min + bbox_max) / 2.0
            plane_d = float(plane_model[3])
            plane_n = plane_model[:3]
            origin_on_plane = bbox_center - (np.dot(bbox_center, plane_n) + plane_d) * plane_n

            facade_ref = {
                'plane_model': [float(x) for x in plane_model],
                'center': [float(x) for x in origin_on_plane],
            }

            trace("quality.domain_ready", facade_no=facade_no,
                  quality_points=n_valid, seconds=f"{time.perf_counter()-started:.2f}")

            # Compute axes
            facade_type, _, _, _ = classify_plane(plane_model[:3])
            u_axis, v_axis = plane_axes(plane_model[:3], facade_type)

            # Full UV range
            rel = filtered_pts - origin_on_plane
            u_coords = rel @ u_axis
            v_coords = rel @ v_axis
            u_range = float(u_coords.max() - u_coords.min())
            v_range = float(v_coords.max() - v_coords.min())

            trace("quality.uv_range", facade_no=facade_no,
                  u_range=f"{u_range:.3f}", v_range=f"{v_range:.3f}",
                  u_min=f"{u_coords.min():.3f}", u_max=f"{u_coords.max():.3f}",
                  v_min=f"{v_coords.min():.3f}", v_max=f"{v_coords.max():.3f}")

            gsize = float(profile.interval_size_m if profile is not None else 20.0)
            if grid_size is not None:
                gsize = float(grid_size)

            rsize = 2.0 if ruler_size is None else float(ruler_size)
            rstep = 0.05 if ruler_step is None else float(ruler_step)

            trace("quality.algorithm", facade_no=facade_no,
                  points=n_valid, grid=gsize, window_length_m=rsize,
                  ruler_width_m=float(getattr(profile, 'ruler_width_m', .055)), 
                  step_m=rstep)

            algo_started = time.perf_counter()

            params = RulerQualityParameters(
                ruler_length_m=rsize,
                ruler_width_m=float(getattr(profile, 'ruler_width_m', .055)),
                scan_step_m=rstep,
                strip_step_m=float(getattr(profile, 'strip_step_m', rstep)),
                select_band_m=float(getattr(profile, 'select_band_m', .01)),
                hole_band_m=float(getattr(profile, 'hole_band_m', .02)),
                bin_size_m=float(getattr(profile, 'bin_size_m', .04)),
                top_q=float(getattr(profile, 'top_q', 1.0)),
                flatness_limit_mm=float(flatness_limit) * 1000.0,
                verticality_limit_mm=float(verticality_limit_mm),
                verticality_bin_size_m=float(getattr(profile, 'verticality_bin_size_m', .05)),
                min_points=int(getattr(profile, 'min_points', 30)),
                sor_enabled=bool(getattr(profile, 'sor_enabled', True)),
                sor_sigma=float(getattr(profile, 'sor_sigma', 4.0)),
                sor_k=int(getattr(profile, 'sor_k', 8)),
                sor_method=str(getattr(profile, 'sor_method', 'local')),
                sor_w_weight=float(getattr(profile, 'sor_w_weight', 50.0)),
                max_hole_ratio=float(getattr(profile, 'max_hole_ratio', .20)),
                verticality_enabled=bool(getattr(profile, 'verticality_enabled', True)),
                plane_model=tuple(float(x) for x in plane_model),
                interval_size_m=gsize,
                parallel_mode=str(getattr(profile, 'parallel_mode', 'process')),
                n_jobs=int(getattr(profile, 'n_jobs', 4)),
                retain_covered_source_ids=bool(getattr(profile, 'retain_covered_source_ids', False)))

            result = compute_ruler_quality(
                filtered_pts, quality_indices, plane_model,
                np.asarray(facade_ref['center'], dtype=float), 
                u_axis, v_axis, params)

            result = dict(result) if isinstance(result, dict) else {'ok': False, 'reason': 'invalid_result'}

            result['interval_size_m'] = gsize

            # FIX: Ensure overall contains all required fields with correct names
            result.setdefault('overall', {})
            result['overall'].update({
                'plane_model': plane_model.tolist(),
                'normal': plane_model[:3].tolist(),
                'center': facade_ref['center'],
            })

            result.setdefault('thresholds', {})
            result['thresholds'].update({
                'flatness_limit_mm': float(params.flatness_limit_mm),
                'verticality_limit_mm': float(params.verticality_limit_mm),
            })

            vert_data = result.get('verticality', {})
            overall = result.get('overall', {})

            overall['verticality_pass'] = vert_data.get('verticality_pass', False)
            overall['verticality_pass_rate'] = vert_data.get('verticality_pass_rate', 0.0)

            if 'verticality_deviation_mm' not in overall:
                overall['verticality_deviation_mm'] = np.nan
            if 'verticality_max_angle_deg' not in overall:
                overall['verticality_max_angle_deg'] = np.nan

            algo_elapsed = time.perf_counter() - algo_started
            n_windows = len(result.get('windows', []))
            n_intervals = len(result.get('intervals', []))
            n_valid_windows = result.get('overall', {}).get('quality_valid_window_count', 0)

            trace("quality.algorithm_done", facade_no=facade_no,
                  windows=n_windows, intervals=n_intervals, valid_windows=n_valid_windows,
                  seconds=f"{algo_elapsed:.2f}")

            result['__global_indices'] = np.asarray(quality_indices, dtype=np.int64)
            result['__index_space'] = 'facade_local_to_raw_global'
            result['quality_domain'] = dict(domain_stats)

            voxel_ids = facade.get('voxel_ids')
            if voxel_ids is not None:
                result['quality_domain']['voxel_ids'] = voxel_ids
                result['quality_domain']['voxel_count'] = len(voxel_ids)
            result['quality_domain']['proxy_count'] = len(facade.get('proxy_indices', []))

            if profile is not None:
                result['profile_snapshot'] = profile.snapshot()

            # Export context
            result['__export_context'] = {
                'results_dir': results_dir,
                'facade_no': facade_no,
                'points': filtered_pts,
                'colors': point_colors[quality_indices] if point_colors is not None else None,
            }

            total_elapsed = time.perf_counter() - started
            trace("quality.finish", facade_no=facade_no, 
                  total_seconds=f"{total_elapsed:.2f}")

            return result

        except Exception as e:
            trace("quality.error", facade_no=facade_no,
                  seconds=f"{time.perf_counter()-started:.2f}", error=repr(e))
            print(f'立面质量评估失败: {e}', flush=True)
            return {
                'ok': False,
                'reason': 'exception',
                'message': f'质量评估异常: {e}',
                'error': repr(e),
                'facade_no': facade_no,
            }