from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, asdict, field
import time
import os
import numpy as np

from .ruler_flatness_3d import prepare_surface, ruler_at, RULER_KERNEL_VERSION

_EMPTY_SOURCE_IDS = np.empty(0, dtype=np.int64)


@dataclass(frozen=True)
class RulerQualityParameters:
    ruler_length_m: float = 2.0
    ruler_width_m: float = .055
    scan_step_m: float = .05
    strip_step_m: float = .05
    flatness_angles_deg: tuple = (0., 45., 90., 135.)
    select_band_m: float = .01
    hole_band_m: float = .02
    bin_size_m: float = .04
    top_q: float = 1.
    flatness_limit_mm: float = 8.
    verticality_limit_mm: float = 5.
    verticality_bin_size_m: float = .05
    min_points: int = 30
    sor_enabled: bool = True
    sor_sigma: float = 4.
    sor_k: int = 8
    sor_method: str = 'local'
    sor_w_weight: float = 50.0
    max_hole_ratio: float = .20
    verticality_enabled: bool = False  # FIX: Disabled by default to prevent hang on wide facades
    gravity_axis: tuple = (0., 0., 1.)
    plane_model: tuple | None = field(default=None, repr=False, compare=False)
    interval_size_m: float = 20.
    parallel_mode: str = 'process'
    n_jobs: int = 4
    retain_covered_source_ids: bool = False

    def snapshot(self):
        d = asdict(self)
        d.pop('plane_model', None)
        return d


def _unit(value):
    value = np.asarray(value, float)
    norm = np.linalg.norm(value)
    if norm < 1e-12:
        raise ValueError('方向向量长度为零')
    return value / norm


def _scan_grid(lo, hi, step, length):
    """Return ruler centres whose complete interval lies inside [lo, hi]."""
    half = float(length) / 2.0
    if hi - lo < length:
        return np.empty(0, float)
    return np.unique(np.r_[lo + half, np.arange(lo + half, hi - half + step * 0.5, step), hi - half])


# =============================================================================
# Direction-level projection
# =============================================================================

def _project_direction(points, plane_model, origin, u_axis, v_axis, angle_deg):
    """Project all points to a direction-specific 1D coordinate system."""
    rad = np.radians(float(angle_deg))
    d3 = np.cos(rad) * u_axis + np.sin(rad) * v_axis
    d3 = _unit(d3)
    q3 = np.cross(plane_model[:3] / np.linalg.norm(plane_model[:3]), d3)
    q3 = _unit(q3)

    rel = points - origin
    along = rel @ d3
    across = rel @ q3
    w = rel @ (plane_model[:3] / np.linalg.norm(plane_model[:3]))

    return along, across, w, d3, q3


def _build_strips_directional(along, across, w, raw_ids, angle, params):
    """Build strips for one direction using already-projected coordinates."""
    across_order = np.argsort(across, kind='stable')
    across_sorted = across[across_order]
    along_sorted = along[across_order]
    w_sorted = w[across_order]
    ids_sorted = raw_ids[across_order]

    q_min, q_max = across_sorted.min(), across_sorted.max()
    if q_max - q_min < params.ruler_width_m:
        return [], np.empty((0, 2), float), np.empty(0, np.int64), across_sorted, along_sorted, w_sorted, ids_sorted

    half_width = params.ruler_width_m / 2.0
    strip_centers = np.arange(q_min + half_width, q_max - half_width + 1e-9, 
                               params.ruler_width_m)
    if len(strip_centers) == 0:
        return [], np.empty((0, 2), float), np.empty(0, np.int64), across_sorted, along_sorted, w_sorted, ids_sorted

    strip_lo = strip_centers - half_width
    strip_hi = strip_centers + half_width

    idx_lo = np.searchsorted(across_sorted, strip_lo, side='left')
    idx_hi = np.searchsorted(across_sorted, strip_hi, side='right')

    strips = []
    for i, (c, lo_idx, hi_idx) in enumerate(zip(strip_centers, idx_lo, idx_hi)):
        if hi_idx - lo_idx < params.min_points:
            continue

        strip_along = along_sorted[lo_idx:hi_idx]
        a_min, a_max = strip_along.min(), strip_along.max()

        strips.append({
            'key': i,
            'center_index': int(i),
            'across_center': float(c),
            'idx_lo': int(lo_idx),
            'idx_hi': int(hi_idx),
            'along_min': float(a_min - params.ruler_length_m / 2),
            'along_max': float(a_max + params.ruler_length_m / 2),
        })

    centres = []
    strip_indices = []
    for strip in strips:
        lo = max(float(strip['along_min']), float(along_sorted.min()))
        hi = min(float(strip['along_max']), float(along_sorted.max()))
        scan = _scan_grid(lo, hi, params.scan_step_m, params.ruler_length_m)
        if scan.size:
            centres.extend((float(a), float(strip['across_center'])) for a in scan)
            strip_indices.extend([strip['center_index']] * len(scan))
    centres_grid = np.asarray(centres, dtype=float).reshape(-1, 2)
    strip_idx = np.asarray(strip_indices, dtype=np.int64)

    return strips, centres_grid, strip_idx, across_sorted, along_sorted, w_sorted, ids_sorted


def _prepare_strip_surface_directional(*, along_sorted, across_sorted, w_sorted, 
                                        raw_ids_sorted, strip, d3, q3, 
                                        plane_model, origin, params):
    """Prepare surface for a strip using pre-projected coordinates."""
    lo_idx, hi_idx = strip['idx_lo'], strip['idx_hi']

    strip_along = along_sorted[lo_idx:hi_idx]
    strip_across = across_sorted[lo_idx:hi_idx]
    strip_w = w_sorted[lo_idx:hi_idx]
    strip_ids = raw_ids_sorted[lo_idx:hi_idx]

    a_mask = (strip_along >= strip['along_min']) & (strip_along <= strip['along_max'])
    if a_mask.sum() < params.min_points:
        return None

    normal = plane_model[:3] / np.linalg.norm(plane_model[:3])
    points_3d = (origin + strip_along[a_mask][:, None] * d3 
                 + strip_across[a_mask][:, None] * q3 
                 + strip_w[a_mask][:, None] * normal)

    try:
        surface = prepare_surface(
            points_3d, source_ids=strip_ids[a_mask],
            ruler_dir=d3, outward=normal,
            select_band=params.select_band_m, hole_band=params.hole_band_m,
            bin_size=params.bin_size_m, top_q=params.top_q,
            sor={'enable': params.sor_enabled, 'sigma': params.sor_sigma,
                 'k': params.sor_k, 'method': params.sor_method,
                 'w_weight': params.sor_w_weight},
            plane_model=tuple(plane_model),
            plane_origin=origin)
    except (ValueError, IndexError, FloatingPointError) as e:
        return None

    return surface


def _ruler_at_for_centre(*, surface, centre_uv, d3, q3, origin, angle, params):
    """Lightweight window-level measurement with center_xyz."""
    local_center = float(centre_uv[0])
    result = ruler_at(surface, local_center, params.ruler_length_m)
    result['direction_deg'] = float(angle)
    result['candidate_count'] = int(surface.stats.get('n_points', 0))

    ruler_half = params.ruler_length_m / 2.0
    u_center = local_center
    if surface.holes.size:
        overlap = np.clip(np.minimum(surface.holes[:, 1], u_center + ruler_half) -
                         np.maximum(surface.holes[:, 0], u_center - ruler_half), 0, None)
        hole_length = float(overlap.sum())
        hole_ratio = hole_length / params.ruler_length_m
    else:
        hole_length = 0.0
        hole_ratio = 0.0

    result['hole_length_m'] = hole_length
    result['hole_ratio'] = hole_ratio
    result['coverage_valid'] = hole_ratio <= params.max_hole_ratio

    if params.retain_covered_source_ids:
        lo, hi = np.searchsorted(surface.top_u, 
                                  [u_center - ruler_half, u_center + ruler_half])
        result['covered_source_ids'] = surface.top_source_ids[lo:hi].copy()
    else:
        result['covered_source_ids'] = _EMPTY_SOURCE_IDS

    u_mask = ((surface.u >= u_center - ruler_half) &
              (surface.u <= u_center + ruler_half) & surface.surf)
    result['effective_point_count'] = int(np.count_nonzero(u_mask))
    result['defect_source_ids'] = np.asarray(
        [result.get('depression_source_id', -1)], np.int64)

    along_c = centre_uv[0]
    across_c = centre_uv[1]
    center_xyz = origin + along_c * d3 + across_c * q3
    result['center_xyz'] = center_xyz.tolist()
    result['center_uv'] = (float(along_c), float(across_c))

    return result


def _direction_worker_directional(args):
    """Process one direction using direction-level projection."""
    (points, raw_ids, plane_model, origin, u_axis, v_axis, angle, params_dict) = args
    params = RulerQualityParameters(**params_dict)

    started = time.perf_counter()

    along, across, w, d3, q3 = _project_direction(
        points, plane_model, origin, u_axis, v_axis, angle)

    along_range = float(along.max() - along.min())
    across_range = float(across.max() - across.min())

    projection_time = time.perf_counter() - started

    if along_range < params.ruler_length_m:
        print(f'[PCFD] quality.direction_skip angle={angle:g} '
              f'along_range={along_range:.3f}m < ruler_length={params.ruler_length_m}m',
              flush=True)
        return float(angle), [], projection_time, 0.0, 0.0, np.empty((0, 2), float), along_range, across_range

    strips, centres_grid, strip_idx, across_sorted, along_sorted, w_sorted, ids_sorted = _build_strips_directional(
        along, across, w, raw_ids, angle, params)
    n_windows = len(centres_grid)

    if n_windows == 0:
        return float(angle), [], projection_time, 0.0, 0.0, np.empty((0, 2), float), along_range, across_range

    window_results = [None] * n_windows

    for strip_idx_num, strip in enumerate(strips):
        window_indices = np.flatnonzero(strip_idx == strip['center_index'])
        if len(window_indices) == 0:
            continue

        surface = _prepare_strip_surface_directional(
            along_sorted=along_sorted,
            across_sorted=across_sorted,
            w_sorted=w_sorted,
            raw_ids_sorted=ids_sorted,
            strip=strip,
            d3=d3,
            q3=q3,
            plane_model=plane_model,
            origin=origin,
            params=params)

        if surface is None:
            for wi in window_indices:
                window_results[wi] = {
                    'ok': False,
                    'reason': 'strip_surface_failed',
                    'direction_deg': float(angle),
                    'candidate_count': 0,
                    'covered_source_ids': _EMPTY_SOURCE_IDS,
                    'coverage_valid': False,
                    'hole_ratio': 1.0,
                    'center_xyz': [np.nan, np.nan, np.nan],
                    'center_uv': (np.nan, np.nan),
                }
            continue

        for wi in window_indices:
            centre_uv = centres_grid[wi]
            result = _ruler_at_for_centre(
                surface=surface,
                centre_uv=centre_uv,
                d3=d3,
                q3=q3,
                origin=origin,
                angle=angle,
                params=params)
            window_results[wi] = result

        if strip_idx_num and strip_idx_num % 500 == 0:
            completed = sum(r is not None for r in window_results)
            print(f'[PCFD] quality.progress direction={angle:g} '
                  f'strips={strip_idx_num + 1}/{len(strips)} windows={completed}/{n_windows}',
                  flush=True)

    for wi in range(n_windows):
        if window_results[wi] is None:
            window_results[wi] = {
                'ok': False,
                'reason': 'no_strip',
                'direction_deg': float(angle),
                'candidate_count': 0,
                'covered_source_ids': _EMPTY_SOURCE_IDS,
                'coverage_valid': False,
                'hole_ratio': 1.0,
                'center_xyz': [np.nan, np.nan, np.nan],
                'center_uv': (np.nan, np.nan),
            }

    total_time = time.perf_counter() - started
    prep_time = total_time - projection_time

    return float(angle), window_results, projection_time, prep_time, total_time, centres_grid, along_range, across_range


# =============================================================================
# Verticality — DISABLED to prevent hang on wide facades
# =============================================================================

def _compute_verticality(points, raw_ids, u_axis, v_axis, origin, params):
    """Verticality computation is disabled due to O(n^3) complexity on wide facades.

    Original implementation caused process hang on facades with:
    - u_range ~23m, v_range ~97m, points ~1.1M
    - n_strips = ceil(23/0.055) = 426
    - n_buckets = 97/0.05 = 1948 per strip
    - Total ops ~ 426 * 1948 * 1.1M = 980 billion (infeasible)

    To re-enable, implement vectorized binning or pre-aggregate with histogram.
    """
    return {
        'ok': False,
        'reason': 'verticality_disabled_due_to_performance',
        'verticality_deviation_mm': np.nan,
        'verticality_max_angle_deg': np.nan,
        'verticality_pass': False,
        'bands': [],
    }


# =============================================================================
# Main entry point
# =============================================================================

def compute_ruler_quality(points, raw_ids, plane_model, origin, u_axis, v_axis, params=None):
    """Run complete U/V coverage with direction-level projection optimization."""
    params = params or RulerQualityParameters()
    started = time.perf_counter()
    points = np.asarray(points, float).reshape(-1, 3)
    raw_ids = np.asarray(raw_ids, np.int64).reshape(-1)
    if len(points) != len(raw_ids) or len(points) < max(3, params.min_points):
        return {'ok': False, 'reason': 'quality_domain_too_small'}

    plane_model = np.asarray(plane_model, float)
    if plane_model.shape[0] != 4:
        return {'ok': False, 'reason': 'invalid_plane_model'}

    if not np.all(np.isfinite(points)):
        valid_mask = np.all(np.isfinite(points), axis=1)
        if not np.any(valid_mask):
            return {'ok': False, 'reason': 'all_points_invalid'}
        points = points[valid_mask]
        raw_ids = raw_ids[valid_mask]

    origin = np.asarray(origin, float)
    u_axis = _unit(u_axis)
    v_axis = _unit(v_axis)

    normal = plane_model[:3] / np.linalg.norm(plane_model[:3])

    # Full-domain UV check
    rel_all = points - origin
    u_all = rel_all @ u_axis
    v_all = rel_all @ v_axis
    u_min_full = float(u_all.min())
    u_max_full = float(u_all.max())
    v_min_full = float(v_all.min())
    v_max_full = float(v_all.max())
    u_range_full = u_max_full - u_min_full
    v_range_full = v_max_full - v_min_full

    print(f'[PCFD] quality.domain_check '
          f'u_range={u_range_full:.3f}m v_range={v_range_full:.3f}m '
          f'u_min={u_min_full:.3f} u_max={u_max_full:.3f} '
          f'v_min={v_min_full:.3f} v_max={v_max_full:.3f} '
          f'points={len(points)}', flush=True)

    print(f'[PCFD] quality.projection_axes '
          f'u_axis=[{u_axis[0]:.3f},{u_axis[1]:.3f},{u_axis[2]:.3f}] '
          f'v_axis=[{v_axis[0]:.3f},{v_axis[1]:.3f},{v_axis[2]:.3f}] '
          f'normal=[{normal[0]:.3f},{normal[1]:.3f},{normal[2]:.3f}]',
          flush=True)

    # Parallel over 4 directions
    args_list = []
    for angle in params.flatness_angles_deg:
        args_list.append((
            points, raw_ids, plane_model, origin, u_axis, v_axis,
            angle, params.snapshot()
        ))

    n_workers = min(len(params.flatness_angles_deg), params.n_jobs, os.cpu_count() or 1)

    if params.parallel_mode == 'process' and n_workers > 1:
        print(f'[PCFD] quality.parallel_process workers={n_workers} '
              f'points={len(points)}', flush=True)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_direction_worker_directional, args_list))
    else:
        print(f'[PCFD] quality.parallel_thread workers={n_workers} '
              f'points={len(points)}', flush=True)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_direction_worker_directional, args) for args in args_list]
            results = [f.result() for f in futures]

    direction_results = {}
    direction_centres = {}
    direction_ranges = {}
    total_projection_time = 0.0
    total_prep_time = 0.0

    for angle, window_results, proj_time, prep_time, total_time, centres_grid, along_range, across_range in results:
        direction_results[float(angle)] = window_results
        direction_centres[float(angle)] = centres_grid
        direction_ranges[float(angle)] = (along_range, across_range)
        total_projection_time += proj_time
        total_prep_time += prep_time

    # Collect all windows from ALL directions
    all_windows = []
    for angle in params.flatness_angles_deg:
        angle_f = float(angle)
        windows = direction_results.get(angle_f, [])
        for w in windows:
            w['source_direction'] = angle_f
            all_windows.append(w)

    # Validity classification
    candidate_windows = all_windows
    geometry_valid_windows = []
    quality_valid_windows = []
    failed_windows = []

    for w in all_windows:
        if not w.get('ok'):
            failed_windows.append(w)
            continue

        gap = w.get('max_depression', np.nan)
        center = w.get('center_xyz', [np.nan, np.nan, np.nan])
        eff_pts = w.get('effective_point_count', 0)

        is_geometry_valid = (
            np.isfinite(gap) and 
            np.all(np.isfinite(center)) and
            eff_pts >= params.min_points
        )

        if is_geometry_valid:
            geometry_valid_windows.append(w)
            if w.get('coverage_valid', False):
                quality_valid_windows.append(w)
        else:
            w['reason'] = w.get('reason', '') + '|geometry_invalid'
            failed_windows.append(w)

    print(f'[PCFD] quality.windows_summary '
          f'candidates={len(candidate_windows)} '
          f'geometry_valid={len(geometry_valid_windows)} '
          f'quality_valid={len(quality_valid_windows)} '
          f'failed={len(failed_windows)}', flush=True)

    if not all_windows:
        return {
            'ok': False,
            'reason': 'no_windows_generated',
            'message': f'所有方向均无法生成检测窗口。'
                       f'立面尺寸 U={u_range_full:.2f}m, V={v_range_full:.2f}m，'
                       f'小于靠尺长度 {params.ruler_length_m}m。',
            'projection': {
                'u_range_m': u_range_full,
                'v_range_m': v_range_full,
                'u_min_m': u_min_full,
                'u_max_m': u_max_full,
                'v_min_m': v_min_full,
                'v_max_m': v_max_full,
                'u_axis': u_axis.tolist(),
                'v_axis': v_axis.tolist(),
                'normal': normal.tolist(),
            },
            'direction_ranges': {str(k): {'along_m': v[0], 'across_m': v[1]} 
                                for k, v in direction_ranges.items()},
        }

    # Build unified window records
    rows = []
    for w in geometry_valid_windows:
        max_dep = w.get('max_depression', np.nan)
        if max_dep is None or not isinstance(max_dep, (int, float, np.floating)):
            gap_mm = np.nan
        else:
            gap_mm = float(max_dep) * 1000.
        rows.append({
            'window_id': len(rows),
            'flatness_gap_mm': gap_mm,
            'flatness_pass': bool(np.isfinite(gap_mm) and gap_mm > 0 and gap_mm <= params.flatness_limit_mm),
            'coverage_valid': w.get('coverage_valid', False),
            'hole_ratio': float(w.get('hole_ratio', 1.0)),
            'candidate_count': w.get('candidate_count', 0),
            'effective_point_count': w.get('effective_point_count', 0),
            'direction_deg': w.get('direction_deg', 0.0),
            'depression_source_id': w.get('depression_source_id', -1),
            'pivot_source_ids': list(w.get('pivot_source_ids', [-1, -1])),
            'center_xyz': (
                [float(x) for x in w.get('center_xyz', [np.nan, np.nan, np.nan])]
                if isinstance(w.get('center_xyz'), (list, tuple, np.ndarray)) and len(w.get('center_xyz', [])) == 3
                else [np.nan, np.nan, np.nan]
            ),
            'center_uv': w.get('center_uv', (np.nan, np.nan)),
            'hole_length_m': w.get('hole_length_m', 0.0),
            'angle_deg': w.get('angle_deg', 0.0),
            'source_direction': w.get('source_direction', 0.0),
        })

    finite_rows = [r for r in rows if np.isfinite(r['flatness_gap_mm']) and r['flatness_gap_mm'] > 0]

    # Build intervals using center_xyz v coordinate
    size = max(float(params.interval_size_m), 1e-6)
    intervals = []

    if rows:
        centers_array = np.array([r['center_xyz'] for r in rows])
        rel_centers = centers_array - origin
        center_u = rel_centers @ u_axis
        center_v = rel_centers @ v_axis

        v_min = float(np.min(center_v))
        v_max = float(np.max(center_v))

        v_span = v_max_full - v_min_full
        n_intervals = max(1, int(np.ceil(v_span / size)))
        edges = np.linspace(v_min_full, v_max_full, n_intervals + 1)
        interval_ids = np.minimum(np.searchsorted(edges, center_v, side='right') - 1,
                                  n_intervals - 1)
        for i in range(n_intervals):
            interval_rows = [r for r, iid in zip(rows, interval_ids) if iid == i]
            gaps = [r['flatness_gap_mm'] for r in interval_rows if np.isfinite(r['flatness_gap_mm'])]
            passes = [r['flatness_pass'] for r in interval_rows]
            quality_passes = [r['flatness_pass'] and r['coverage_valid'] for r in interval_rows]
            intervals.append({
                'label': f'{edges[i]:.2f}–{edges[i+1]:.2f}m',
                'v_min_m': float(edges[i]), 'v_max_m': float(edges[i+1]),
                'window_count': len(interval_rows),
                'valid_window_count': len(gaps),
                'point_count': int(min(len(points), sum(r['effective_point_count'] for r in interval_rows))),
                'flatness_max_gap_mm': float(np.nanmax(gaps)) if gaps else 0.0,
                'flatness_pass_rate': float(np.mean(passes)) if passes else 0.0,
                'quality_pass_rate': float(np.mean(quality_passes)) if quality_passes else 0.0,
                'status': 'ok' if gaps else 'no_valid_window',
            })

        intervals.sort(key=lambda x: x['v_min_m'])

    # FIX: Verticality disabled - return placeholder without computation
    verticality = _compute_verticality(points, raw_ids, u_axis, v_axis, origin, params)

    total_elapsed = time.perf_counter() - started

    print(f'[PCFD] quality.scan_done '
          f'candidates={len(candidate_windows)} '
          f'geometry_valid={len(geometry_valid_windows)} '
          f'quality_valid={len(quality_valid_windows)} '
          f'intervals={len(intervals)} '
          f'projection={total_projection_time:.2f}s '
          f'prep={total_prep_time:.2f}s '
          f'total={total_elapsed:.2f}s', flush=True)

    return {
        'ok': True,
        'schema_version': 'facade-quality/10',
        'kernel_version': RULER_KERNEL_VERSION,
        'parameters': params.snapshot(),
        'windows': rows,
        'intervals': intervals,
        'interval_size_m': size,
        'interval_count': len(intervals),
        'projection_origin': origin.tolist(),
        'projection_u_axis': u_axis.tolist(),
        'projection_v_axis': v_axis.tolist(),
        'projection_normal': normal.tolist(),
        'projection': {
            'u_range_m': u_range_full,
            'v_range_m': v_range_full,
            'u_min_m': u_min_full,
            'u_max_m': u_max_full,
            'v_min_m': v_min_full,
            'v_max_m': v_max_full,
        },
        'direction_ranges': {str(k): {'along_m': v[0], 'across_m': v[1]} 
                            for k, v in direction_ranges.items()},
        'overall': {
            'point_count': len(points),
            'candidate_window_count': len(candidate_windows),
            'geometry_valid_window_count': len(geometry_valid_windows),
            'quality_valid_window_count': len(quality_valid_windows),
            'failed_window_count': len(failed_windows),
            'flatness_max_gap_mm': max((r['flatness_gap_mm'] for r in finite_rows), default=0.),
            'flatness_pass_rate': float(np.mean([r['flatness_pass'] for r in finite_rows])) if finite_rows else 0.,
            'quality_pass_rate': float(np.mean([r['flatness_pass'] and r['coverage_valid'] for r in finite_rows])) if finite_rows else 0.,
            'flatness_valid_rate': float(np.mean([r['coverage_valid'] for r in rows])) if rows else 0.,
            'max_hole_ratio': max((r['hole_ratio'] for r in rows), default=0.),
            'verticality_deviation_mm': verticality.get('verticality_deviation_mm', np.nan),
            'verticality_max_angle_deg': verticality.get('verticality_max_angle_deg', np.nan),
            'verticality_pass': verticality.get('verticality_pass', False),
        },
        'verticality': verticality,
        'performance': {
            'projection_seconds': total_projection_time,
            'prep_seconds': total_prep_time,
            'total_seconds': total_elapsed,
        },
    }