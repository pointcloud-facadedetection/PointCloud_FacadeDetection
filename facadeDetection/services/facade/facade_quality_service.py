"""立面质量评估业务编排服务。

仅负责质量评估流程调度、参数组装与结果返回；不直接操作 UI 控件，也不包含
底层质量统计算法实现，不执行任何文件导出。
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
from algorithms.facade.ruler_quality import RulerQualityParameters, compute_ruler_quality
from algorithms.facade.global_plane_quality import fit_global_plane, compute_global_plane_quality
from algorithms.facade.coverage import CoverageMask
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

    def commit_quality_success(self, project_uuid: str, facade_id: int, quality: dict,
                               *, display_no=None, facade_data=None, color=None,
                               dataset_revision=None, quality_artifact_path=None) -> None:
        """持久化一次成功的质量评估结果（委托 DAL）。"""
        from services.dal.results_repo import ResultsRepo
        ResultsRepo.commit_quality_success(
            project_uuid, facade_id, quality,
            display_no=display_no,
            facade_data=facade_data,
            color=color,
            dataset_revision=dataset_revision,
            quality_artifact_path=quality_artifact_path,
        )

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
                # New-window defaults are authoritative. A caller may still
                # explicitly override the step for legacy profiles.
                if ruler_step is None:
                    ruler_step = None

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

            # 鲁棒全局平面拟合：先验法向量 + Huber IRLS（全量点）
            # 新参数优先从 profile 读取，无则回退 Config 常量，保证老项目兼容
            from config.settings import Config
            global_fit = fit_global_plane(
                filtered_pts,
                reference_plane=plane_model,
                seed=int(getattr(profile, 'global_plane_seed', 42)),
                huber_delta_m=float(getattr(
                    profile, 'global_plane_huber_delta_m',
                    getattr(Config, 'GLOBAL_PLANE_HUBER_DELTA_M', 0.015))),
                max_iterations=int(getattr(
                    profile, 'global_plane_max_iterations',
                    getattr(Config, 'GLOBAL_PLANE_MAX_ITERATIONS', 500))),
                angle_limit_deg=float(getattr(
                    profile, 'global_plane_angle_limit_deg',
                    getattr(Config, 'GLOBAL_PLANE_ANGLE_LIMIT_DEG', 3.0))),
                outlier_sigma=float(getattr(
                    profile, 'global_plane_outlier_sigma',
                    getattr(Config, 'GLOBAL_PLANE_OUTLIER_SIGMA', 3.0))),
                final_gate_sigma=float(getattr(
                    profile, 'global_plane_final_gate_sigma',
                    getattr(Config, 'GLOBAL_PLANE_FINAL_GATE_SIGMA', 2.5))),
                min_inlier_ratio=float(getattr(
                    profile, 'global_plane_min_inlier_ratio',
                    getattr(Config, 'GLOBAL_PLANE_MIN_INLIER_RATIO', 0.30))),
                max_p95_mm=float(getattr(
                    profile, 'global_plane_max_p95_mm',
                    getattr(Config, 'GLOBAL_PLANE_MAX_P95_MM', 100.0))),
                enable_partition_fallback=bool(getattr(
                    profile, 'global_plane_enable_partition_fallback',
                    getattr(Config, 'GLOBAL_PLANE_ENABLE_PARTITION_FALLBACK', True))),
                partition_depth_gap_m=float(getattr(
                    profile, 'global_plane_partition_depth_gap_m',
                    getattr(Config, 'GLOBAL_PLANE_PARTITION_DEPTH_GAP_M', 0.08))),
                partition_min_points_ratio=float(getattr(
                    profile, 'global_plane_partition_min_points_ratio',
                    getattr(Config, 'GLOBAL_PLANE_PARTITION_MIN_POINTS_RATIO', 0.10))),
            )
            if not global_fit.get('fit_accepted', False):
                trace('quality.reject', facade_no=facade_no,
                      reason='global_plane_fit_not_accepted',
                      inlier_ratio=global_fit.get('inlier_ratio'),
                      p95_mm=global_fit.get('p95_abs_residual_mm'))
                return {
                    'ok': False, 'reason': 'global_plane_fit_not_accepted',
                    'message': '全局立面基准面拟合未达到质量门槛，已拒绝输出毫米级检测结果。',
                    'reference_plane': global_fit,
                }
            global_plane = np.asarray(global_fit['plane_model'], dtype=np.float64)
            if float(global_plane[:3] @ plane_model[:3]) < 0:
                global_plane *= -1.0

            # The fitted support, not the raw candidate cloud, defines the
            # facade extent.  Raw candidates may contain balconies, frames and
            # background fragments; they are still measured *inside* this
            # extent but must never inflate the common area denominator.
            support_mask = np.asarray(global_fit.pop('support_mask', []), dtype=bool)
            if support_mask.shape != (len(filtered_pts),) or support_mask.sum() < 3:
                support_mask = np.ones(len(filtered_pts), dtype=bool)
            support_pts = filtered_pts[support_mask]

            # Use robust support quantiles for the frame origin; raw outliers
            # must not move the frame or inflate the effective facade area.
            facade_type_seed, _, _, _ = classify_plane(plane_model[:3])
            seed_u_axis, seed_v_axis = plane_axes(plane_model[:3], facade_type_seed)
            seed_rel = support_pts - np.mean(support_pts, axis=0)
            seed_u = seed_rel @ seed_u_axis
            seed_v = seed_rel @ seed_v_axis
            q_u0, q_u1 = np.quantile(seed_u, [0.0025, 0.9975])
            q_v0, q_v1 = np.quantile(seed_v, [0.0025, 0.9975])
            bbox_center = (np.mean(support_pts, axis=0) +
                           seed_u_axis * ((q_u0 + q_u1) * .5) +
                           seed_v_axis * ((q_v0 + q_v1) * .5))
            # Both algorithms use the robust global reference frame. The
            # detected facade plane is retained only as the orientation prior.
            plane_model = global_plane
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

            # Full UV range. Bounds are computed in the final shared global
            # reference frame, so the same values can be passed to both
            # algorithms without an origin/axis mismatch.
            support_rel = support_pts - origin_on_plane
            support_u = support_rel @ u_axis
            support_v = support_rel @ v_axis
            q_u0, q_u1 = np.quantile(support_u, [0.0025, 0.9975])
            q_v0, q_v1 = np.quantile(support_v, [0.0025, 0.9975])
            # Keep every raw candidate inside the reliable facade envelope;
            # this includes actual local bulges/recesses but removes detached
            # structures that were responsible for the 4.2m/23.4m domain split.
            raw_rel = filtered_pts - origin_on_plane
            raw_u, raw_v = raw_rel @ u_axis, raw_rel @ v_axis
            bounds_mask = ((raw_u >= q_u0) & (raw_u <= q_u1) &
                           (raw_v >= q_v0) & (raw_v <= q_v1))
            if np.count_nonzero(bounds_mask) >= max(3, int(getattr(profile, 'min_points', 3))):
                filtered_pts = filtered_pts[bounds_mask]
                quality_indices = quality_indices[bounds_mask]
                n_valid = len(filtered_pts)
            u_range = float(q_u1 - q_u0)
            v_range = float(q_v1 - q_v0)

            trace("quality.uv_range", facade_no=facade_no,
                  u_range=f"{u_range:.3f}", v_range=f"{v_range:.3f}",
                  u_min=f"{q_u0:.3f}", u_max=f"{q_u1:.3f}",
                  v_min=f"{q_v0:.3f}", v_max=f"{q_v1:.3f}")

            gsize = float(profile.interval_size_m if profile is not None else 20.0)
            if grid_size is not None:
                gsize = float(grid_size)

            rsize = 2.0 if ruler_size is None else float(ruler_size)
            window_width = float(getattr(profile, 'global_window_width_m',
                                         getattr(profile, 'ruler_width_m', .055)))
            # A 2 m ruler is the footprint length, not the scan pitch.  Using
            # the ruler length here produced one row every 2 m and left most
            # of a facade outside the measurement union.  The standard
            # profile's 5 cm pitch is the authoritative default.
            rstep = float(getattr(profile, 'scan_step_m', .05)) if ruler_step is None else float(ruler_step)

            trace("quality.algorithm", facade_no=facade_no,
                  points=n_valid, grid=gsize, window_length_m=rsize,
                  ruler_width_m=float(getattr(profile, 'ruler_width_m', .055)), 
                  step_m=rstep)

            algo_started = time.perf_counter()

            params = RulerQualityParameters(
                ruler_length_m=rsize,
                ruler_width_m=float(getattr(profile, 'ruler_width_m', .055)),
                scan_step_m=rstep,
                strip_step_m=float(getattr(profile, 'strip_step_m', window_width)),
                select_band_m=float(getattr(profile, 'select_band_m', .01)),
                hole_band_m=float(getattr(profile, 'hole_band_m', .02)),
                bin_size_m=float(getattr(profile, 'bin_size_m', .04)),
                top_q=float(getattr(profile, 'top_q', 1.0)),
                flatness_limit_mm=float(flatness_limit) * 1000.0,
                verticality_limit_mm=float(verticality_limit_mm),
                verticality_bin_size_m=float(getattr(profile, 'verticality_bin_size_m', rsize)),
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
                retain_covered_source_ids=True)

            result = compute_ruler_quality(
                filtered_pts, quality_indices, plane_model,
                np.asarray(facade_ref['center'], dtype=float), 
                u_axis, v_axis, params)

            result = dict(result) if isinstance(result, dict) else {'ok': False, 'reason': 'invalid_result'}

            result['interval_size_m'] = gsize

            # Ensure overall contains all required fields with correct names
            result.setdefault('overall', {})
            result['overall'].update({
                'plane_model': plane_model.tolist(),
                'normal': plane_model[:3].tolist(),
                'center': facade_ref['center'],
            })
            # Persist the exact projection frame used by both quality
            # algorithms. Export/render code must not reconstruct an origin
            # from the raw cloud, otherwise heatmap locations drift.
            result['projection_origin'] = origin_on_plane.tolist()
            result['projection_u_axis'] = u_axis.tolist()
            result['projection_v_axis'] = v_axis.tolist()

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
            result['quality_domain'].update({
                'raw_point_count': int(n_valid),
                'reference_support_point_count': int(len(support_pts)),
                'u_min_m': float(q_u0), 'u_max_m': float(q_u1),
                'v_min_m': float(q_v0), 'v_max_m': float(q_v1),
            })

            global_result = compute_global_plane_quality(
                filtered_pts, global_plane, origin_on_plane, u_axis, v_axis,
                length_m=float(getattr(profile, 'global_window_length_m', rsize)),
                width_m=window_width,
                flatness_limit_mm=float(params.flatness_limit_mm),
                verticality_limit_mm=float(params.verticality_limit_mm),
                min_points=int(getattr(profile, 'global_window_min_points', 3)),
                uv_bounds=(float(q_u0), float(q_u1), float(q_v0), float(q_v1)),
                raw_ids=quality_indices,
            )
            ruler_overall = result.get('overall') or {}
            valid_windows = [w for w in result.get('windows', [])
                             if w.get('coverage_valid', True)]
            raw_total = len(np.unique(quality_indices))

            # ---- 真实有效立面面积：使用滤波后点云投影掩码（排除无点空白区）----
            global_overall = global_result.get('overall') or {}
            ruler_length = float(params.ruler_length_m)
            ruler_width = float(params.ruler_width_m)
            _DOMAIN_RES = 0.01  # UV 掩码分辨率 1cm

            def _window_rect(w, length_m, width_m):
                """窗口在 UV 面上的覆盖矩形（靠尺定向/全局轴对齐）。"""
                cx = np.asarray(w.get('center_xyz'), dtype=float)
                if cx.size != 3 or not np.all(np.isfinite(cx)):
                    return None
                rel = cx - origin_on_plane
                cu = float(rel @ u_axis)
                cv = float(rel @ v_axis)
                hl = float(length_m) / 2.0
                hw = float(width_m) / 2.0
                wdir = w.get('direction_deg')
                try:
                    wdir = float(wdir) if wdir is not None else np.nan
                except (TypeError, ValueError):
                    wdir = np.nan
                if np.isfinite(wdir):
                    adeg = float(wdir) % 180.0
                    if abs(adeg) <= 45.0 or abs(adeg - 180.0) <= 45.0:
                        # 水平靠尺：长边沿 u
                        return (cu - hl, cv - hw, cu + hl, cv + hw)
                    if abs(adeg - 90.0) <= 45.0:
                        # 竖直靠尺：长边沿 v
                        return (cu - hw, cv - hl, cu + hw, cv + hl)
                    # 45/135 对角：取覆盖盒
                    return (cu - hl, cv - hl, cu + hl, cv + hl)
                # 全局平面窗口：轴对齐，长边沿 v
                return (cu - hw, cv - hl, cu + hw, cv + hl)

            def _coverage_rates(windows_list, pass_key, length_m, width_m,
                                bounds=None):
                """统一去重的面积/点数合格率（布尔掩码并集）。

                面积合格率 = 被合格窗口覆盖且有点的掩码面积 / 有点的掩码总面积。
                点数合格率 = 合格掩码单元内点数 / 总点数（每点唯一归属一个单元）。
                布尔并集天然抵消重叠，恒 <= 1.0；有效面积基于真实点投影掩码，
                而非 UV 包围盒 / 凸包（两者都会计入无点区造成虚高）。
                """
                if not windows_list:
                    return {'area_rate': 0.0, 'point_rate': 0.0,
                            'pass_area_m2': 0.0, 'total_area_m2': 0.0,
                            'detection_area_m2': 0.0, 'fail_area_m2': 0.0,
                            'pass_points': 0, 'total_points': 0,
                            'fail_points': 0,
                            'pass_windows': 0, 'total_windows': 0}
                res = _DOMAIN_RES
                bu0, bv0, bu1, bv1 = bounds or (q_u0, q_v0, q_u1, q_v1)
                support = CoverageMask(bu0, bv0, bu1, bv1, resolution=res)
                rel_pts = filtered_pts - origin_on_plane
                u_pts = rel_pts @ u_axis
                v_pts = rel_pts @ v_axis
                in_domain = ((u_pts >= bu0) & (u_pts <= bu1) &
                             (v_pts >= bv0) & (v_pts <= bv1))
                u_domain, v_domain = u_pts[in_domain], v_pts[in_domain]
                support.mark_pts(u_domain, v_domain)
                total_area = support.area_m2()
                cell_u = np.clip(((u_domain - bu0) / res).astype(np.int64), 0, support.nu - 1)
                cell_v = np.clip(((v_domain - bv0) / res).astype(np.int64), 0, support.nv - 1)
                cell_id = cell_u * support.nv + cell_v
                cell_counts = np.bincount(cell_id, minlength=support.nu * support.nv)
                total_points = int(len(u_domain))

                pass_mask = CoverageMask(bu0, bv0, bu1, bv1, resolution=res)
                for w in windows_list:
                    if not w.get(pass_key, False):
                        continue
                    rect = _window_rect(w, length_m, width_m)
                    if rect is not None:
                        pass_mask.mark_rect(*rect)

                pass_cells = pass_mask.mask.reshape(-1) & (cell_counts > 0)
                pass_area = float(np.sum(pass_cells)) * res * res
                # Total/pass/fail must share the same physical-domain
                # denominator.  Window unions are only a sampling mechanism;
                # using their area for fail_area while using point occupancy
                # for total_area made fail_area exceed total_area.
                detection_area = total_area
                fail_area = max(total_area - pass_area, 0.0)
                pass_points = int(np.sum(cell_counts[pass_cells]))
                fail_points = max(total_points - pass_points, 0)
                pass_count = int(sum(1 for w in windows_list if w.get(pass_key, False)))
                return {
                    'area_rate': (pass_area / total_area) if total_area > 0 else 0.0,
                    'point_rate': (pass_points / total_points) if total_points > 0 else 0.0,
                    'pass_area_m2': pass_area,
                    'total_area_m2': total_area,
                    'detection_area_m2': detection_area,
                    'fail_area_m2': fail_area,
                    'pass_points': pass_points,
                    'total_points': total_points,
                    'fail_points': fail_points,
                    'pass_windows': pass_count,
                    'total_windows': len(windows_list),
                }

            # 真实有效立面积（点投影掩码），作为两套链路共用的面积口径
            _support = CoverageMask(q_u0, q_v0, q_u1, q_v1, resolution=_DOMAIN_RES)
            _rel_all = filtered_pts - origin_on_plane
            _support.mark_pts(_rel_all @ u_axis, _rel_all @ v_axis)
            valid_facade_area = _support.area_m2()  # 替代原凸包口径
            if not np.isfinite(valid_facade_area) or valid_facade_area <= 0.0:
                valid_facade_area = float(global_overall.get('valid_detection_area_m2', 0.0))
            global_overall['valid_detection_area_m2'] = valid_facade_area

            # 靠尺法平整度（米字）—— 掩码并集去重
            ruler_flat_all = [dict(w) for w in valid_windows
                              if np.isfinite(w.get('flatness_gap_mm', np.nan))]
            for w in ruler_flat_all:
                w.setdefault('actual_area_m2', ruler_length * ruler_width)
            ruler_flat_rates = _coverage_rates(ruler_flat_all, 'flatness_pass',
                                               ruler_length, ruler_width)

            # 靠尺法垂直度（I字）—— 掩码并集去重
            # Public quality contract intentionally keeps verticality summary
            # only; verticality measurements are merged into the canonical
            # result windows above.  Do not read the removed
            # ``verticality.windows`` key or the I-ruler area becomes zero.
            ruler_vert_all = [dict(w) for w in result.get('verticality', {}).get('rows', [])
                              if np.isfinite(w.get('verticality_deviation_mm', np.nan))]
            for w in ruler_vert_all:
                w.setdefault('actual_area_m2', ruler_length * ruler_width)
            ruler_vert_rates = _coverage_rates(ruler_vert_all, 'verticality_pass',
                                               ruler_length, ruler_width)

            global_length = float(
                (global_result.get('parameters') or {}).get(
                    'window_length_m',
                    getattr(profile, 'global_window_length_m', ruler_length),
                )
            )
            global_width = float(
                (global_result.get('parameters') or {}).get(
                    'window_width_m',
                    window_width,
                )
            )

            # 全局平面法平整度 —— 掩码并集去重。全局窗口尺寸必须使用
            # 全局算法自己的参数，不能悄悄复用靠尺尺寸。
            global_windows = list(global_result.get('windows') or [])
            global_flat_all = [dict(w) for w in global_windows
                               if np.isfinite(w.get('flatness_gap_mm', np.nan))]
            global_flat_rates = _coverage_rates(global_flat_all, 'flatness_pass',
                                                global_length, global_width)

            # 全局平面法垂直度 —— 掩码并集去重
            global_vert_all = [dict(w) for w in global_windows
                               if np.isfinite(w.get('verticality_deviation_mm', np.nan))]
            global_vert_rates = _coverage_rates(global_vert_all, 'verticality_pass',
                                                global_length, global_width)

            def _method_intervals(rows, include_vertical=False):
                """按共享 v 轴生成方法专属区间，避免 UI 复用 ruler intervals。"""
                if not rows:
                    return []
                size = max(float(gsize), 1e-6)
                centers = np.asarray([w.get('center_xyz', [np.nan] * 3)
                                      for w in rows], dtype=float)
                finite = np.all(np.isfinite(centers), axis=1)
                if not np.any(finite):
                    return []
                rows = [w for w, ok_ in zip(rows, finite) if ok_]
                centers = centers[finite]
                row_v = (centers - origin_on_plane) @ v_axis
                lo, hi = float(q_v0), float(q_v1)
                n = max(1, int(np.ceil(max(hi - lo, 0.0) / size)))
                edges = np.linspace(lo, hi, n + 1)
                result_intervals = []
                for i in range(n):
                    selected = [w for w, value in zip(rows, row_v)
                                if (edges[i] <= value < edges[i + 1]) or
                                (i == n - 1 and value == edges[i + 1])]
                    if not selected:
                        continue
                    flat = [float(w['flatness_gap_mm']) for w in selected
                            if np.isfinite(w.get('flatness_gap_mm', np.nan))]
                    vert = [float(w['verticality_deviation_mm']) for w in selected
                            if np.isfinite(w.get('verticality_deviation_mm', np.nan))]
                    fp = [bool(w.get('flatness_pass', False)) for w in selected
                          if np.isfinite(w.get('flatness_gap_mm', np.nan))]
                    vp = [bool(w.get('verticality_pass', False)) for w in selected
                          if np.isfinite(w.get('verticality_deviation_mm', np.nan))]
                    result_intervals.append({
                        'label': f'{edges[i]:.2f}–{edges[i + 1]:.2f}m',
                        'v_min_m': float(edges[i]), 'v_max_m': float(edges[i + 1]),
                        'window_count': len(selected),
                        'valid_window_count': len(selected),
                        'point_count': int(sum(w.get('point_count', 0) or 0 for w in selected)),
                        'flatness_max_gap_mm': max(flat, default=0.0),
                        'flatness_pass_rate': float(np.mean(fp)) if fp else 0.0,
                        'verticality_max_deviation_mm': max(vert, default=np.nan),
                        'verticality_area_rate': float(np.mean(vp)) if vp else 0.0,
                        'flatness_area_rate': float(np.mean(fp)) if fp else 0.0,
                        'status': 'ok' if flat or vert else 'no_valid_window',
                    })
                return result_intervals

            ruler_intervals = list(result.get('intervals') or [])
            global_intervals = _method_intervals(global_windows)

            def _attach_interval_rates(intervals, flat_rows, vert_rows,
                                       length_m, width_m):
                """Add canonical area/point rates to every method interval.

                The denominator is the occupied point domain *inside that
                interval* and the numerator is the clipped union of passing
                footprints.  This is deliberately not a mean of window flags.
                """
                for interval in intervals:
                    lo, hi = float(interval['v_min_m']), float(interval['v_max_m'])
                    def in_interval(row):
                        center = np.asarray(row.get('center_xyz', [np.nan] * 3), float)
                        if center.shape != (3,) or not np.all(np.isfinite(center)):
                            return False
                        value = float((center - origin_on_plane) @ v_axis)
                        return lo <= value <= hi
                    bounds = (q_u0, lo, q_u1, hi)
                    fr = _coverage_rates([w for w in flat_rows if in_interval(w)],
                                         'flatness_pass', length_m, width_m, bounds)
                    vr = _coverage_rates([w for w in vert_rows if in_interval(w)],
                                         'verticality_pass', length_m, width_m, bounds)
                    interval['flatness_area_rate'] = fr['area_rate']
                    interval['flatness_point_rate'] = fr['point_rate']
                    interval['verticality_area_rate'] = vr['area_rate']
                    interval['verticality_point_rate'] = vr['point_rate']
                    # Keep explicit window rate separate from physical area rate.
                    interval['flatness_window_rate'] = (
                        fr['pass_windows'] / fr['total_windows']
                        if fr['total_windows'] else 0.0)
                    interval['verticality_window_rate'] = (
                        vr['pass_windows'] / vr['total_windows']
                        if vr['total_windows'] else 0.0)

            _attach_interval_rates(
                ruler_intervals, ruler_flat_all, ruler_vert_all,
                ruler_length, ruler_width)
            _attach_interval_rates(
                global_intervals, global_flat_all, global_vert_all,
                global_length, global_width)

            # 全局平面法窗口为空时回退为完整复制原始窗口，避免链路空转
            if not global_flat_all:
                global_flat_all = [dict(w) for w in global_windows]
            if not global_vert_all:
                global_vert_all = [dict(w) for w in global_windows]
            global_overall['flatness_primary_area_rate'] = global_flat_rates['area_rate']
            global_overall['flatness_secondary_point_rate'] = global_flat_rates['point_rate']
            global_overall['verticality_primary_area_rate'] = global_vert_rates['area_rate']
            global_overall['verticality_secondary_point_rate'] = global_vert_rates['point_rate']
            global_overall['valid_detection_area_m2'] = valid_facade_area

            result['overall'] = ruler_overall

            methods = {
                'ruler': {
                    'flatness': {
                        'windows': ruler_flat_all,
                        'rates': ruler_flat_rates,
                    },
                    'verticality': {
                        'windows': ruler_vert_all,
                        'rates': ruler_vert_rates,
                    },
                    'parameters': params.snapshot(),
                    'intervals': ruler_intervals,
                },
                'global_plane': {
                    'flatness': {
                        'windows': global_flat_all,
                        'rates': global_flat_rates,
                    },
                    'verticality': {
                        'windows': global_vert_all,
                        'rates': global_vert_rates,
                    },
                    'parameters': global_result.get('parameters') or {},
                    'overall': global_overall,
                    'intervals': global_intervals,
                },
            }

            result['quality_comparison'] = {
                'domain': {
                    'u_min_m': float(q_u0), 'u_max_m': float(q_u1),
                    'v_min_m': float(q_v0), 'v_max_m': float(q_v1),
                    'area_m2': valid_facade_area,
                    'raw_point_count': raw_total,
                },
                'reference_plane': {
                    **global_fit,
                    'plane_model': global_plane.tolist(),
                },
                'methods': methods,
                # Compatibility bridge for legacy callers
                'ruler_method': methods['ruler'],
                'global_plane_method': methods['global_plane'],
            }

            voxel_ids = facade.get('voxel_ids')
            if voxel_ids is not None:
                result['quality_domain']['voxel_ids'] = voxel_ids
                result['quality_domain']['voxel_count'] = len(voxel_ids)
            result['quality_domain']['proxy_count'] = len(facade.get('proxy_indices', []))

            if profile is not None:
                result['profile_snapshot'] = profile.snapshot()

            # 仅保留一个轻量级的导出描述符。
            result['__export_context'] = {
                'results_dir': str(results_dir) if results_dir else None,
                'facade_no': facade_no,
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