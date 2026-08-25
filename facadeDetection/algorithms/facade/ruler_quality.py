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
    verticality_enabled: bool = True
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


def _build_strips_directional(along, across, w, raw_ids, angle, params,
                              common_uv=None, u_axis=None, v_axis=None,
                              d3=None, q3=None):
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
    strip_step = max(float(params.strip_step_m), 1e-6)
    strip_centers = np.arange(q_min + half_width, q_max - half_width + 1e-9,
                               strip_step)
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
        if common_uv is None:
            lo = max(float(strip['along_min']), float(along_sorted.min()))
            hi = min(float(strip['along_max']), float(along_sorted.max()))
            scan = _scan_grid(lo, hi, params.scan_step_m, params.ruler_length_m)
            if scan.size:
                centres.extend((float(a), float(strip['across_center'])) for a in scan)
                strip_indices.extend([strip['center_index']] * len(scan))
        else:
            # The caller owns one common U/V grid.  Project all centers once,
            # then assign each center to its regular across-strip bucket.  Do
            # not scan the common grid once per strip.
            uv = np.asarray(common_uv, dtype=float)
            along_c = uv[:, 0] * float(u_axis @ d3) + uv[:, 1] * float(v_axis @ d3)
            across_c = uv[:, 0] * float(u_axis @ q3) + uv[:, 1] * float(v_axis @ q3)
            if strip is not strips[0]:
                continue
            centers_strip = np.rint((across_c - (q_min + half_width)) / strip_step).astype(np.int64)
            valid_strip = ((centers_strip >= 0) & (centers_strip < len(strip_centers)) &
                           (np.abs(across_c - (q_min + half_width + centers_strip * strip_step)) <= half_width + 1e-9))
            active = np.zeros(len(strip_centers), dtype=bool)
            active[[s['center_index'] for s in strips]] = True
            valid_strip &= active[np.clip(centers_strip, 0, len(strip_centers) - 1)]
            strip_min = np.full(len(strip_centers), np.inf, dtype=float)
            strip_max = np.full(len(strip_centers), -np.inf, dtype=float)
            for item in strips:
                strip_min[item['center_index']] = item['along_min']
                strip_max[item['center_index']] = item['along_max']
            safe_idx = np.clip(centers_strip, 0, len(strip_centers) - 1)
            valid_strip &= (along_c >= strip_min[safe_idx]) & (along_c <= strip_max[safe_idx])
            selected_ids = np.flatnonzero(valid_strip)
            centres_grid = np.column_stack((along_c[selected_ids], across_c[selected_ids]))
            strip_idx = centers_strip[selected_ids]
            break
    if common_uv is not None:
        return strips, centres_grid, strip_idx, across_sorted, along_sorted, w_sorted, ids_sorted
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
    
    # Map back to minimal fields needed for aggregation
    ruler_half = params.ruler_length_m / 2.0
    u_center = local_center
    
    # Effective point count under ruler
    u_mask = ((surface.u >= u_center - ruler_half) &
              (surface.u <= u_center + ruler_half) & surface.surf)
    
    along_c = centre_uv[0]
    across_c = centre_uv[1]
    center_xyz = origin + along_c * d3 + across_c * q3
    
    return {
        'ok': result.get('ok', False),
        'gap_mm': result.get('gap_mm', np.nan),
        'signed_gap_mm': result.get('signed_gap_mm', np.nan),
        'angle_deg': result.get('angle_deg', np.nan),
        'u_center': float(u_center),
        'direction_deg': float(angle),
        'candidate_count': int(surface.stats.get('n_points', 0)),
        'effective_point_count': int(np.count_nonzero(u_mask)),
        'coverage_valid': result.get('coverage_valid', False),
        'hole_ratio': result.get('hole_ratio', 1.0),
        'hole_length_m': result.get('hole_length', 0.0),
        'center_xyz': center_xyz.tolist(),
        'center_uv': (float(along_c), float(across_c)),
        'depression_source_id': result.get('depression_source_id', -1),
        'pivot_source_ids': list(result.get('pivot_source_ids', [-1, -1])),
        'covered_source_ids': result.get('covered_source_ids', _EMPTY_SOURCE_IDS),
    }


def _snap_window_to_base_grid(center_xyz, origin, u_axis, v_axis, u0, v0, step):
    """Snap a world-space window centre back to the facade base U/V grid."""
    center_xyz = np.asarray(center_xyz, dtype=float)
    rel = center_xyz - origin
    base_u = float(rel @ u_axis)
    base_v = float(rel @ v_axis)
    step = max(float(step), 1e-6)
    ku = int(np.round((base_u - u0) / step))
    kv = int(np.round((base_v - v0) / step))
    snap_u = float(u0 + ku * step)
    snap_v = float(v0 + kv * step)
    snap_xyz = origin + snap_u * u_axis + snap_v * v_axis
    distance = float(np.hypot(base_u - snap_u, base_v - snap_v))
    return (ku, kv), snap_u, snap_v, snap_xyz, distance


def _aggregate_star_rows(windows, origin, u_axis, v_axis, u0, v0, params):
    """Aggregate four directional ruler results onto one facade base grid.
    
    Key fix: For each grid cell, keep ALL directions and take MAX gap as flatness.
    """
    # Group by grid key
    grouped = {}
    for w in windows:
        center = w.get('center_xyz', [np.nan, np.nan, np.nan])
        if not np.all(np.isfinite(center)):
            continue
        key, snap_u, snap_v, snap_xyz, snap_distance = _snap_window_to_base_grid(
            center, origin, u_axis, v_axis, u0, v0, params.scan_step_m)
        grouped.setdefault(key, []).append((w, snap_u, snap_v, snap_xyz, snap_distance))

    rows = []
    for key, members in grouped.items():
        directional = []
        for w, _snap_u, _snap_v, _snap_xyz, _snap_distance in members:
            gap_mm = float(w.get('gap_mm', np.nan))
            directional.append({
                'gap_mm': gap_mm,
                'signed_gap_mm': float(w.get('signed_gap_mm', np.nan)),
                'coverage_valid': bool(w.get('coverage_valid', False)),
                'direction_deg': float(w.get('direction_deg', 0.0)),
                'effective_point_count': int(w.get('effective_point_count', 0)),
                'hole_ratio': float(w.get('hole_ratio', 1.0)),
            })

        # Use mean snap position
        snap_u = float(np.mean([m[1] for m in members]))
        snap_v = float(np.mean([m[2] for m in members]))
        snap_xyz = np.mean(np.asarray([m[3] for m in members], dtype=float), axis=0)
        snap_distance = float(np.min([m[4] for m in members]))

        # Classification
        finite_members = [m for m in directional if np.isfinite(m['gap_mm'])]
        valid_members = [m for m in finite_members if m['coverage_valid']]
        quality_valid = bool(valid_members)
        
        # Dominant direction: the one with MAX gap (worst case)
        dominant = max(valid_members if valid_members else finite_members,
                       key=lambda item: item['gap_mm'],
                       default=None)

        # Star flatness = max gap across all valid directions
        all_gaps = [m['gap_mm'] for m in finite_members]
        valid_gaps = [m['gap_mm'] for m in valid_members]
        
        flatness_gap = max(valid_gaps) if valid_gaps else np.nan
        raw_max_gap = max(all_gaps) if all_gaps else np.nan
        
        # Aggregate hole ratio: take MIN (most permissive) across directions
        hole_ratios = [m['hole_ratio'] for m in finite_members]
        agg_hole_ratio = min(hole_ratios) if hole_ratios else 1.0
        
        # Aggregate effective points: take MAX across directions
        eff_points = [m['effective_point_count'] for m in finite_members]
        agg_eff_points = max(eff_points) if eff_points else 0

        row = {
            'window_id': len(rows),
            'grid_key': key,
            'cell_u': int(key[0]),
            'cell_v': int(key[1]),
            'center_xyz': [float(x) for x in snap_xyz],
            'center_uv_base': (snap_u, snap_v),
            'flatness_gap_mm': flatness_gap,
            'flatness_raw_max_gap_mm': raw_max_gap,
            'flatness_pass': bool(quality_valid and np.isfinite(flatness_gap)
                                  and flatness_gap <= params.flatness_limit_mm),
            'coverage_valid': quality_valid,
            'direction_count': len(finite_members),
            'quality_direction_count': len(valid_members),
            'hole_ratio': agg_hole_ratio,
            'effective_point_count': agg_eff_points,
            'direction_deg': float(dominant['direction_deg']) if dominant else np.nan,
            'depression_source_id': -1,  # Simplified: no single source for star
            'pivot_source_ids': [-1, -1],
            'center_uv': (snap_u, snap_v),
            'hole_length_m': 0.0,  # Simplified
            'angle_deg': float(dominant['direction_deg']) if dominant else np.nan,
            'source_direction': float(dominant['direction_deg']) if dominant else np.nan,
            'snap_distance_m': snap_distance,
            'directional_measurements': [
                {
                    'direction_deg': float(item['direction_deg']),
                    'flatness_gap_mm': float(item['gap_mm']) if np.isfinite(item['gap_mm']) else np.nan,
                    'coverage_valid': bool(item['coverage_valid']),
                }
                for item in sorted(directional, key=lambda d: d['direction_deg'])
            ],
        }
        rows.append(row)

    rows.sort(key=lambda item: (item['center_uv_base'][1], item['center_uv_base'][0]))
    for idx, row in enumerate(rows):
        row['window_id'] = idx
    return rows


def _direction_worker_directional(args):
    """Process one direction using direction-level projection."""
    (points, raw_ids, plane_model, origin, u_axis, v_axis, angle,
     params_dict, common_uv) = args
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
        along, across, w, raw_ids, angle, params,
        common_uv=common_uv, u_axis=u_axis, v_axis=v_axis, d3=d3, q3=q3)
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
                    'gap_mm': np.nan,
                    'coverage_valid': False,
                    'hole_ratio': 1.0,
                    'center_xyz': [np.nan, np.nan, np.nan],
                    'center_uv': (np.nan, np.nan),
                    'direction_deg': float(angle),
                    'candidate_count': 0,
                    'effective_point_count': 0,
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

    # Fill None slots
    for wi in range(n_windows):
        if window_results[wi] is None:
            window_results[wi] = {
                'ok': False,
                'gap_mm': np.nan,
                'coverage_valid': False,
                'hole_ratio': 1.0,
                'center_xyz': [np.nan, np.nan, np.nan],
                'center_uv': (np.nan, np.nan),
                'direction_deg': float(angle),
                'candidate_count': 0,
                'effective_point_count': 0,
            }

    total_time = time.perf_counter() - started
    prep_time = total_time - projection_time

    return float(angle), window_results, projection_time, prep_time, total_time, centres_grid, along_range, across_range


def _unpack_direction_result(result):
    """Normalize direction-worker results across the 8/9-field protocols.

    Older quality workers briefly returned one additional diagnostic field,
    while the current worker returns eight fields.  Positional unpacking at
    the aggregation boundary made a mixed/stale worker fail with
    ``expected 9, got 8``.  Keep the algorithm fields in one stable order and
    ignore the optional trailing diagnostic value for compatibility.
    """
    if not isinstance(result, (tuple, list)):
        raise TypeError(f'方向质量 worker 返回异常类型: {type(result).__name__}')
    if len(result) not in (8, 9):
        raise ValueError(f'方向质量 worker 返回字段数异常: expected 8 or 9, got {len(result)}')

    # The first eight fields are the stable protocol.  A ninth field, when
    # present, is optional diagnostic data and is not needed for aggregation.
    return tuple(result[:8])


# =============================================================================
# Verticality: 2 m I-ruler on a cached 5 cm representative profile
# =============================================================================

def _compute_verticality_fast(points, raw_ids, plane_model, u_axis, v_axis, origin, params):
    """按工程定义计算垂直度，避免 ``窗口×条带点数`` 的重复扫描。

    每个 5 cm 横向条带只排序/分箱一次；每个纵向 5 cm bin 保留离面最高的
    代表点，然后用 ``searchsorted`` 取得 2 m 窗口。靠尺是窗口两端代表点的
    连线，指标为代表曲线到该连线的最大横截偏差。这样复杂度约为
    ``O(N log N + strips * windows * 40)``，而不是原实现的数十亿次点访问。
    """
    started = time.perf_counter()
    def empty(reason):
        return {'ok': False, 'reason': reason, 'verticality_deviation_mm': np.nan,
                'verticality_max_angle_deg': np.nan, 'verticality_pass': False,
                'verticality_pass_rate': 0.0, 'rows': []}

    if not params.verticality_enabled:
        return empty('verticality_disabled')

    plane_model = np.asarray(plane_model, dtype=float)
    if plane_model.shape[0] != 4:
        return empty('invalid_plane_model')

    normal = plane_model[:3] / np.linalg.norm(plane_model[:3])
    gravity = np.asarray(params.gravity_axis, dtype=float)
    gravity = gravity / np.linalg.norm(gravity)
    
    # Never assume v_axis is gravity: derive the true vertical in the facade plane.
    v_gravity_proj = gravity - np.dot(gravity, normal) * normal
    if np.linalg.norm(v_gravity_proj) < 1e-6:
        return empty('horizontal_facade_no_verticality')
    vertical_axis = _unit(v_gravity_proj)
    horizontal_axis = _unit(np.cross(normal, vertical_axis))
    
    # Project points to facade UV coordinates
    rel = points - origin
    u_coords = rel @ horizontal_axis
    v_coords = rel @ vertical_axis
    w = rel @ normal  # deviation from plane
    
    u_min, u_max = float(u_coords.min()), float(u_coords.max())
    v_min, v_max = float(v_coords.min()), float(v_coords.max())
    
    u_range = u_max - u_min
    v_range = v_max - v_min
    
    if v_range < params.ruler_length_m:
        return empty('verticality_domain_too_short')
    
    # Build vertical strips (horizontal bands)
    half_width = params.ruler_width_m / 2.0
    strip_step = max(float(params.strip_step_m), 1e-6)
    
    # Sort once by horizontal coordinate; each strip is obtained by searchsorted.
    u_order = np.argsort(u_coords, kind='stable')
    u_sorted = u_coords[u_order]
    v_sorted = v_coords[u_order]
    w_sorted = w[u_order]
    ids_sorted = raw_ids[u_order]
    
    u_strip_centers = np.arange(u_min + half_width, u_max - half_width + 1e-9, strip_step)
    if len(u_strip_centers) == 0:
        return empty('no_verticality_strips')
    
    # For each strip, scan vertically with 2m window
    ruler_half = params.ruler_length_m / 2.0
    v_step = max(float(params.verticality_bin_size_m), 1e-6)
    base_u = rel @ u_axis
    base_v = rel @ v_axis
    base_u_min = float(base_u.min())
    base_v_min = float(base_v.min())
    
    rows = []
    row_map = {}
    total_windows = 0
    valid_windows = 0
    print(f'[PCFD] verticality.begin strips={len(u_strip_centers)} '
          f'points={len(points)} step={strip_step:.3f}m/{v_step:.3f}m', flush=True)
    
    for u_center in u_strip_centers:
        u_lo = u_center - half_width
        u_hi = u_center + half_width
        
        left = int(np.searchsorted(u_sorted, u_lo, side='left'))
        right = int(np.searchsorted(u_sorted, u_hi, side='right'))
        if right <= left:
            continue
        strip_v = v_sorted[left:right]
        strip_w = w_sorted[left:right]
        strip_ids = ids_sorted[left:right]
        if strip_v.size < params.min_points:
            continue
        
        v_lo = float(strip_v.min())
        v_hi = float(strip_v.max())
        
        if v_hi - v_lo < params.ruler_length_m:
            continue
        
        # Sliding window along vertical direction
        # Construct one representative curve per strip (highest point per 5 cm bin).
        b = np.floor((strip_v - v_min) / v_step).astype(np.int64)
        order = np.argsort(b, kind='stable')
        bs, ws, is_ = b[order], strip_w[order], strip_ids[order]
        starts = np.r_[0, np.flatnonzero(bs[1:] != bs[:-1]) + 1]
        ends = np.r_[starts[1:], bs.size]
        rep_pos = np.array([s + int(np.argmax(ws[s:e])) for s, e in zip(starts, ends)], dtype=np.int64)
        curve_v = v_min + (bs[rep_pos] + .5) * v_step
        curve_w = ws[rep_pos]
        curve_ids = is_[rep_pos]
        valid_curve = np.isfinite(curve_v) & np.isfinite(curve_w)
        curve_v, curve_w, curve_ids = curve_v[valid_curve], curve_w[valid_curve], curve_ids[valid_curve]
        if curve_v.size < 3:
            continue
        v_centers = np.arange(v_lo + ruler_half, v_hi - ruler_half + 1e-9, v_step)
        total_windows += len(v_centers)
        
        for vc in v_centers:
            lo_i = int(np.searchsorted(curve_v, vc - ruler_half, side='left'))
            hi_i = int(np.searchsorted(curve_v, vc + ruler_half, side='right'))
            if hi_i - lo_i < 3:
                continue
            v_win, w_win, ids_win = curve_v[lo_i:hi_i], curve_w[lo_i:hi_i], curve_ids[lo_i:hi_i]
            try:
                a = float((w_win[-1] - w_win[0]) / (v_win[-1] - v_win[0]))
                c = float(w_win[0] - a * v_win[0])
            except (np.linalg.LinAlgError, ValueError):
                continue
            
            # Angle between fitted line and true vertical
            # Slope a = dw/dv, angle = arctan(a)
            angle_deg = float(np.degrees(np.arctan(abs(a))))
            
            fitted = a * v_win + c
            deviations = np.abs(w_win - fitted) / np.hypot(1.0, a)
            max_pos = int(np.argmax(deviations))
            max_dev = float(deviations[max_pos])
            max_dev_mm = max_dev * 1000.0
            verticality_dev = max_dev_mm
            valid_windows += 1
            
            # Snap to base grid
            center_xyz = origin + u_center * horizontal_axis + vc * vertical_axis
            key, snap_u, snap_v, snap_xyz, snap_distance = _snap_window_to_base_grid(
                center_xyz.tolist(), origin, u_axis, v_axis, base_u_min, base_v_min, params.scan_step_m)
            
            row = {
                'grid_key': key,
                'cell_u': int(key[0]),
                'cell_v': int(key[1]),
                'center_xyz': [float(x) for x in np.asarray(snap_xyz, dtype=float)],
                'center_uv_base': (snap_u, snap_v),
                'verticality_deviation_mm_2m': verticality_dev,
                'verticality_angle_deg': angle_deg,
                'verticality_pass': bool(np.isfinite(verticality_dev) and verticality_dev <= params.verticality_limit_mm),
                'hole_ratio': float(max(0, int(np.rint(params.ruler_length_m / v_step)) - (hi_i - lo_i)) /
                                      max(1, int(np.rint(params.ruler_length_m / v_step)))),
                'coverage_valid': True,
                'effective_point_count': int(hi_i - lo_i),
                'snap_distance_m': snap_distance,
                'pivot_source_ids': [int(ids_win[0]), int(ids_win[-1])],
                'depression_source_id': int(ids_win[max_pos]),
            }
            
            # Keep best (closest to grid center) per cell
            prev = row_map.get(key)
            if prev is None or snap_distance < prev['snap_distance_m']:
                row_map[key] = row

    rows = list(row_map.values())
    finite_rows = [r for r in rows if np.isfinite(r['verticality_deviation_mm_2m'])]
    pass_rows = [r['verticality_pass'] for r in finite_rows]

    elapsed = time.perf_counter() - started
    print(f'[PCFD] verticality.summary strips={len(u_strip_centers)} '
          f'windows={total_windows} valid={valid_windows} rows={len(rows)} '
          f'max_deviation_mm={max((r["verticality_deviation_mm_2m"] for r in finite_rows), default=np.nan):.3f} '
          f'seconds={elapsed:.2f}', flush=True)
    return {
        'ok': bool(rows),
        'reason': '' if rows else 'no_verticality_rows',
        'verticality_deviation_mm': max((r['verticality_deviation_mm_2m'] for r in finite_rows), default=np.nan),
        'verticality_max_angle_deg': max((r['verticality_angle_deg'] for r in finite_rows), default=np.nan),
        'verticality_pass': bool(pass_rows and all(pass_rows)),
        'verticality_pass_rate': float(np.mean(pass_rows)) if pass_rows else 0.0,
        'rows': rows,
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
          f'points={len(points)}', flush=True)

    # One physical star centre grid for all four directions.  Erode the base
    # rectangle by half the ruler length so every centre has a complete
    # 2-metre support interval in the facade frame; each direction then uses
    # this exact same centre and only changes its local projection.
    half = params.ruler_length_m / 2.0
    if u_range_full < params.ruler_length_m or v_range_full < params.ruler_length_m:
        common_uv = np.empty((0, 2), dtype=float)
    else:
        u_centres = np.arange(u_min_full + half,
                              u_max_full - half + params.scan_step_m * .5,
                              max(params.scan_step_m, 1e-6))
        v_centres = np.arange(v_min_full + half,
                              v_max_full - half + params.scan_step_m * .5,
                              max(params.scan_step_m, 1e-6))
        uu, vv = np.meshgrid(u_centres, v_centres, indexing='xy')
        common_uv = (np.column_stack((uu.ravel(), vv.ravel()))
                     if u_centres.size and v_centres.size
                     else np.empty((0, 2), dtype=float))
    print(f'[PCFD] quality.common_grid centers={len(common_uv)} '
          f'step={params.scan_step_m:.3f}m', flush=True)

    # Parallel over 4 directions for flatness
    args_list = []
    for angle in params.flatness_angles_deg:
        args_list.append((
            points, raw_ids, plane_model, origin, u_axis, v_axis,
            angle, params.snapshot(), common_uv
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
    direction_ranges = {}
    total_projection_time = 0.0
    total_prep_time = 0.0

    for direction_result in results:
        (angle, window_results, proj_time, prep_time, total_time,
         centres_grid, along_range, across_range) = _unpack_direction_result(direction_result)
        direction_results[float(angle)] = window_results
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

        gap = w.get('gap_mm', np.nan)
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
            },
            'direction_ranges': {str(k): {'along_m': v[0], 'across_m': v[1]} 
                                for k, v in direction_ranges.items()},
        }

    # Aggregate star rows
    rows = _aggregate_star_rows(
        geometry_valid_windows, origin, u_axis, v_axis,
        u_min_full, v_min_full, params)

    valid_rows = [r for r in rows if r.get('coverage_valid', False)]
    finite_rows = [r for r in valid_rows if np.isfinite(r['flatness_gap_mm'])]
    raw_rows = [r for r in rows if np.isfinite(r['flatness_raw_max_gap_mm'])]

    # Build intervals
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
            valid_interval_rows = [r for r in interval_rows if r.get('coverage_valid', False)]
            gaps = [r['flatness_gap_mm'] for r in valid_interval_rows if np.isfinite(r['flatness_gap_mm'])]
            raw_gaps = [r['flatness_raw_max_gap_mm'] for r in interval_rows if np.isfinite(r['flatness_raw_max_gap_mm'])]
            passes = [r['flatness_pass'] for r in valid_interval_rows]
            
            intervals.append({
                'label': f'{edges[i]:.2f}–{edges[i+1]:.2f}m',
                'v_min_m': float(edges[i]), 'v_max_m': float(edges[i+1]),
                'window_count': len(interval_rows),
                'valid_window_count': len(valid_interval_rows),
                'point_count': int(min(len(points), sum(r['effective_point_count'] for r in interval_rows))),
                'flatness_max_gap_mm': float(np.nanmax(gaps)) if gaps else 0.0,
                'flatness_raw_max_gap_mm': float(np.nanmax(raw_gaps)) if raw_gaps else 0.0,
                'flatness_pass_rate': float(np.mean(passes)) if passes else 0.0,
                'quality_pass_rate': float(np.mean(passes)) if passes else 0.0,
                'status': 'ok' if gaps else 'no_valid_window',
            })

        intervals.sort(key=lambda x: x['v_min_m'])

    # Compute verticality with FAST vectorized method
    verticality = _compute_verticality_fast(points, raw_ids, plane_model, u_axis, v_axis, origin, params)
    verticality_rows = {}
    for item in verticality.get('rows', []):
        key = item.get('grid_key')
        if key is not None:
            verticality_rows[tuple(key)] = item

    # Merge verticality into flatness rows
    for row in rows:
        vrow = verticality_rows.get(tuple(row.get('grid_key', ())))
        if vrow is None:
            row['verticality_angle_deg'] = np.nan
            row['verticality_deviation_mm_2m'] = np.nan
            row['verticality_pass'] = False
        else:
            row['verticality_angle_deg'] = float(vrow.get('verticality_angle_deg', np.nan))
            row['verticality_deviation_mm_2m'] = float(vrow.get('verticality_deviation_mm_2m', np.nan))
            row['verticality_pass'] = bool(vrow.get('verticality_pass', False))

    # Update intervals with verticality
    if intervals:
        for interval in intervals:
            vrows = [
                row for row in rows
                if interval['v_min_m'] <= row['center_uv_base'][1] <= interval['v_max_m']
                and np.isfinite(row.get('verticality_deviation_mm_2m', np.nan))
            ]
            interval['verticality_max_deviation_mm_2m'] = (
                float(np.nanmax([row['verticality_deviation_mm_2m'] for row in vrows])) if vrows else 0.0
            )
            interval['verticality_pass_rate'] = float(np.mean([row['verticality_pass'] for row in vrows])) if vrows else 0.0

    total_elapsed = time.perf_counter() - started

    print(f'[PCFD] quality.scan_done '
          f'candidates={len(candidate_windows)} '
          f'geometry_valid={len(geometry_valid_windows)} '
          f'quality_valid={len(quality_valid_windows)} '
          f'intervals={len(intervals)} '
          f'projection={total_projection_time:.2f}s '
          f'prep={total_prep_time:.2f}s '
          f'total={total_elapsed:.2f}s', flush=True)

    # Clean result: only fields needed by FacadeQualityDialog
    return {
        'ok': True,
        'schema_version': 'facade-quality/11',
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
            'candidate_window_count': len(rows),
            'geometry_valid_window_count': len(rows),
            'quality_valid_window_count': len(valid_rows),
            'failed_window_count': len(failed_windows),
            'flatness_max_gap_mm': max((r['flatness_gap_mm'] for r in finite_rows), default=0.),
            'flatness_raw_max_gap_mm': max((r['flatness_raw_max_gap_mm'] for r in raw_rows), default=0.),
            'flatness_pass_rate': float(np.mean([r['flatness_pass'] for r in valid_rows])) if valid_rows else 0.,
            'quality_pass_rate': float(np.mean([r['flatness_pass'] for r in valid_rows])) if valid_rows else 0.,
            'flatness_valid_rate': float(np.mean([r['coverage_valid'] for r in rows])) if rows else 0.,
            'max_hole_ratio': max((r['hole_ratio'] for r in rows), default=0.),
            'verticality_deviation_mm': verticality.get('verticality_deviation_mm', np.nan),
            'verticality_deviation_mm_2m': verticality.get('verticality_deviation_mm', np.nan),
            'verticality_max_angle_deg': verticality.get('verticality_max_angle_deg', np.nan),
            'verticality_pass': verticality.get('verticality_pass', False),
            'verticality_pass_rate': verticality.get('verticality_pass_rate', 0.0),
        },
        'verticality': {
            'ok': verticality.get('ok', False),
            'verticality_pass': verticality.get('verticality_pass', False),
            'verticality_pass_rate': verticality.get('verticality_pass_rate', 0.0),
        },
        'performance': {
            'projection_seconds': total_projection_time,
            'prep_seconds': total_prep_time,
            'total_seconds': total_elapsed,
        },
    }