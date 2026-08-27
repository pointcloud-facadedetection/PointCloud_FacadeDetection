from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, asdict, field
import time
import os
import numpy as np

from .ruler_flatness_3d import prepare_surface, ruler_at, fit_line

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
    verticality_limit_mm: float = 4. 
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
    half = float(length) / 2.0
    if hi - lo < length:
        return np.empty(0, float)
    return np.unique(np.r_[lo + half, np.arange(lo + half, hi - half + step * 0.5, step), hi - half])


# =============================================================================
# Flatness: direction-level projection 
# =============================================================================

def _project_direction(points, plane_model, origin, u_axis, v_axis, angle_deg):
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
    strip_centers = np.arange(q_min + half_width, q_max - half_width + 1e-9, strip_step)
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
            'key': i, 'center_index': int(i), 'across_center': float(c),
            'idx_lo': int(lo_idx), 'idx_hi': int(hi_idx),
            'along_min': float(a_min - params.ruler_length_m / 2),
            'along_max': float(a_max + params.ruler_length_m / 2),
        })

    if not strips:
        return [], np.empty((0, 2), float), np.empty(0, np.int64), across_sorted, along_sorted, w_sorted, ids_sorted

    # FIX v3: When common_uv is provided, project ALL common_uv points to this direction
    # and assign them to the appropriate strips based on their across coordinate.
    if common_uv is not None and len(strips) > 0 and u_axis is not None and d3 is not None:
        uv = np.asarray(common_uv, dtype=float)
        # Project common_uv (in base u,v coordinates) to current direction's (along, across)
        along_c = uv[:, 0] * float(u_axis @ d3) + uv[:, 1] * float(v_axis @ d3)
        across_c = uv[:, 0] * float(u_axis @ q3) + uv[:, 1] * float(v_axis @ q3)

        # Determine which strip each point belongs to
        strip_idx_for_uv = np.floor((across_c - (q_min + half_width)) / strip_step).astype(np.int64)

        # Basic validity: within strip bounds
        valid = ((strip_idx_for_uv >= 0) & 
                 (strip_idx_for_uv < len(strip_centers)) &
                 (np.abs(across_c - strip_centers[np.clip(strip_idx_for_uv, 0, len(strip_centers)-1)]) <= half_width + 1e-9))

        # Build active strip set and along range lookup
        active_strips = {s['center_index']: s for s in strips}

        # Check along range for each point based on its assigned strip
        for strip_idx_val, strip in active_strips.items():
            mask = (strip_idx_for_uv == strip_idx_val) & valid
            if not np.any(mask):
                continue
            a_c = along_c[mask]
            a_valid = (a_c >= strip['along_min']) & (a_c <= strip['along_max'])
            # Update validity for points in this strip
            valid_indices = np.flatnonzero(mask)
            valid[valid_indices[~a_valid]] = False

        # Keep only points assigned to active strips
        selected_ids = np.flatnonzero(valid)
        if len(selected_ids) > 0:
            final_strip_idx = strip_idx_for_uv[selected_ids]
            # Filter to only active strips
            active_mask = np.isin(final_strip_idx, list(active_strips.keys()))
            if np.any(active_mask):
                centres_grid = np.column_stack((along_c[selected_ids[active_mask]], 
                                                across_c[selected_ids[active_mask]]))
                strip_idx = final_strip_idx[active_mask]
                return strips, centres_grid, strip_idx, across_sorted, along_sorted, w_sorted, ids_sorted

    # Fallback: generate scan grid independently for each strip (when no common_uv)
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
            plane_model=tuple(plane_model), plane_origin=origin)
    except (ValueError, IndexError, FloatingPointError) as e:
        return None
    return surface


def _ruler_at_for_centre(*, surface, centre_uv, d3, q3, origin, angle, params):
    local_center = float(centre_uv[0])
    result = ruler_at(surface, local_center, params.ruler_length_m)
    ruler_half = params.ruler_length_m / 2.0
    u_center = local_center
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


def _aggregate_star_rows(windows_by_direction, origin, u_axis, v_axis, u0, v0, params):
    """Aggregate four directional ruler results onto one facade base grid.
    
    "米字形"聚合逻辑
    工程定义：在同一物理位置（snapped to base grid cell），
    四个方向各放置一根2m靠尺，取四个方向中的最大 gap 作为该位置的平整度。
    """
    all_snapped = []
    for direction_deg, windows in windows_by_direction.items():
        for w in windows:
            if not w.get('ok'):
                continue
            center = w.get('center_xyz', [np.nan, np.nan, np.nan])
            if not np.all(np.isfinite(center)):
                continue
            key, snap_u, snap_v, snap_xyz, snap_distance = _snap_window_to_base_grid(
                center, origin, u_axis, v_axis, u0, v0, params.scan_step_m)
            all_snapped.append((key, float(direction_deg), w, snap_u, snap_v, snap_xyz, snap_distance))
    
    grouped = {}
    for key, direction_deg, w, snap_u, snap_v, snap_xyz, snap_distance in all_snapped:
        grouped.setdefault(key, []).append({
            'direction_deg': direction_deg, 'window': w,
            'snap_u': snap_u, 'snap_v': snap_v,
            'snap_xyz': snap_xyz, 'snap_distance': snap_distance,
        })
    
    rows = []
    for key, members in grouped.items():
        directional = []
        for m in members:
            w = m['window']
            gap_mm = float(w.get('gap_mm', np.nan))
            directional.append({
                'direction_deg': m['direction_deg'],
                'gap_mm': gap_mm,
                'signed_gap_mm': float(w.get('signed_gap_mm', np.nan)),
                'coverage_valid': bool(w.get('coverage_valid', False)),
                'effective_point_count': int(w.get('effective_point_count', 0)),
                'hole_ratio': float(w.get('hole_ratio', 1.0)),
                'snap_distance': m['snap_distance'],
            })
        
        snap_u = float(np.mean([m['snap_u'] for m in members]))
        snap_v = float(np.mean([m['snap_v'] for m in members]))
        snap_xyz = np.mean(np.asarray([m['snap_xyz'] for m in members], dtype=float), axis=0)
        snap_distance = float(np.min([m['snap_distance'] for m in members]))
        
        finite_members = [d for d in directional if np.isfinite(d['gap_mm'])]
        valid_members = [d for d in finite_members if d['coverage_valid']]
        
        all_gaps = [d['gap_mm'] for d in finite_members]
        valid_gaps = [d['gap_mm'] for d in valid_members]
        
        flatness_gap = max(valid_gaps) if valid_gaps else np.nan
        raw_max_gap = max(all_gaps) if all_gaps else np.nan
        
        dominant = max(valid_members if valid_members else finite_members,
                       key=lambda item: item['gap_mm'], default=None)
        
        quality_valid = bool(valid_members)
        hole_ratios = [d['hole_ratio'] for d in finite_members]
        agg_hole_ratio = min(hole_ratios) if hole_ratios else 1.0
        eff_points = [d['effective_point_count'] for d in finite_members]
        agg_eff_points = max(eff_points) if eff_points else 0
        
        row = {
            'window_id': len(rows),
            'grid_key': key,
            'cell_u': int(key[0]), 'cell_v': int(key[1]),
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
            'depression_source_id': -1,
            'pivot_source_ids': [-1, -1],
            'center_uv': (snap_u, snap_v),
            'hole_length_m': 0.0,
            'angle_deg': float(dominant['direction_deg']) if dominant else np.nan,
            'source_direction': float(dominant['direction_deg']) if dominant else np.nan,
            'snap_distance_m': snap_distance,
            'directional_measurements': [
                {
                    'direction_deg': float(d['direction_deg']),
                    'flatness_gap_mm': float(d['gap_mm']) if np.isfinite(d['gap_mm']) else np.nan,
                    'coverage_valid': bool(d['coverage_valid']),
                    'effective_point_count': d['effective_point_count'],
                }
                for d in sorted(directional, key=lambda x: x['direction_deg'])
            ],
        }
        rows.append(row)

    rows.sort(key=lambda item: (item['center_uv_base'][1], item['center_uv_base'][0]))
    for idx, row in enumerate(rows):
        row['window_id'] = idx
    return rows


def _direction_worker_directional(args):
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
              f'along_range={along_range:.3f}m < ruler_length={params.ruler_length_m}m', flush=True)
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
            along_sorted=along_sorted, across_sorted=across_sorted,
            w_sorted=w_sorted, raw_ids_sorted=ids_sorted, strip=strip,
            d3=d3, q3=q3, plane_model=plane_model, origin=origin, params=params)

        if surface is None:
            for wi in window_indices:
                window_results[wi] = {
                    'ok': False, 'gap_mm': np.nan, 'coverage_valid': False,
                    'hole_ratio': 1.0, 'center_xyz': [np.nan, np.nan, np.nan],
                    'center_uv': (np.nan, np.nan), 'direction_deg': float(angle),
                    'candidate_count': 0, 'effective_point_count': 0,
                }
            continue

        for wi in window_indices:
            centre_uv = centres_grid[wi]
            result = _ruler_at_for_centre(
                surface=surface, centre_uv=centre_uv, d3=d3, q3=q3,
                origin=origin, angle=angle, params=params)
            window_results[wi] = result

        if strip_idx_num and strip_idx_num % 500 == 0:
            completed = sum(r is not None for r in window_results)
            print(f'[PCFD] quality.progress direction={angle:g} '
                  f'strips={strip_idx_num + 1}/{len(strips)} windows={completed}/{n_windows}', flush=True)

    for wi in range(n_windows):
        if window_results[wi] is None:
            window_results[wi] = {
                'ok': False, 'gap_mm': np.nan, 'coverage_valid': False,
                'hole_ratio': 1.0, 'center_xyz': [np.nan, np.nan, np.nan],
                'center_uv': (np.nan, np.nan), 'direction_deg': float(angle),
                'candidate_count': 0, 'effective_point_count': 0,
            }

    total_time = time.perf_counter() - started
    prep_time = total_time - projection_time
    return float(angle), window_results, projection_time, prep_time, total_time, centres_grid, along_range, across_range


def _unpack_direction_result(result):
    if not isinstance(result, (tuple, list)):
        raise TypeError(f'方向质量 worker 返回异常类型: {type(result).__name__}')
    if len(result) not in (8, 9):
        raise ValueError(f'方向质量 worker 返回字段数异常: expected 8 or 9, got {len(result)}')
    return tuple(result[:8])


# =============================================================================
# Verticality: CORRECTED high-performance implementation
# =============================================================================

def _compute_verticality_corrected(points, raw_ids, plane_model, u_axis, v_axis, origin, params):
    """修正后的高性能垂直度计算。
    
    工程定义（GB50210-2018）：
    "立面垂直度采用2m垂直检测尺检查"
    
    物理模型纠正：
    ─────────────────────────────────────────────────────────────
    旧版错误理解：
      测量表面点在2m窗口内的横向偏移范围
      → 计算所有点的 |u - u_median|
      → 包含离面很远的噪声点，导致48mm偏差
      
    新版正确理解：
      2m垂直检测尺紧贴墙面表面
      → 只测量表面点（|w| <= select_band）
      → 在2m竖向范围内，表面点的水平偏移 = 墙面竖直偏差
      → 垂直度 = max(|H - H_median|) for surface points only
    ─────────────────────────────────────────────────────────────
    
    性能优化：
    - 旧版：strips循环(468) × windows循环(137k) = 671s
    - 新版：向量化grid assignment + 聚合到base grid = <10s
    
    关键修正点：
    1. 严格筛选表面点（|w| <= select_band）
    2. 使用与平整度一致的base grid（避免重复计算）
    3. 向量化分组统计（np.argsort + reduceat）
    4. 每个grid cell只计算一次（与平整度聚合逻辑一致）
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
    
    # Step 1: 确定 true vertical 方向（世界竖直在立面内的投影）
    v_gravity_proj = gravity - np.dot(gravity, normal) * normal
    v_gravity_norm = np.linalg.norm(v_gravity_proj)
    if v_gravity_norm < 1e-6:
        return empty('horizontal_facade_no_verticality')
    
    vertical_axis = v_gravity_proj / v_gravity_norm
    horizontal_axis = np.cross(normal, vertical_axis)
    h_norm = np.linalg.norm(horizontal_axis)
    if h_norm < 1e-12:
        return empty('degenerate_horizontal_axis')
    horizontal_axis = horizontal_axis / h_norm
    
    # Step 2: 全局投影到 (H, V, N) 坐标系
    rel = points - origin
    u_h = rel @ horizontal_axis   # 水平横向（立面内）
    v_v = rel @ vertical_axis     # true vertical（立面内）
    w_n = rel @ normal            # 离面高度
    
    # Step 3: 【关键修正】严格筛选表面点
    # 只取 |w| <= select_band 的点 = 立面表面附近的点
    # 这是旧版最大的错误：用了所有点（包含离面很远的噪声点）
    surface_band = float(getattr(params, 'select_band_m', 0.01))
    surf_mask = np.abs(w_n) <= surface_band
    n_surf = int(np.count_nonzero(surf_mask))
    
    if n_surf < params.min_points:
        return empty(f'too_few_surface_points:{n_surf}')
    
    surf_u = u_h[surf_mask]
    surf_v = v_v[surf_mask]
    surf_ids = raw_ids[surf_mask]
    
    # Step 4: 构建与平整度一致的 base grid
    u_min_full = float(u_h.min())
    u_max_full = float(u_h.max())
    v_min_full = float(v_v.min())
    v_max_full = float(v_v.max())
    
    step = max(float(params.scan_step_m), 1e-6)
    half = params.ruler_length_m / 2.0
    
    if u_max_full - u_min_full < params.ruler_length_m or v_max_full - v_min_full < params.ruler_length_m:
        return empty('verticality_domain_too_short')
    
    # 与平整度完全一致的 grid
    u_centers = np.arange(u_min_full + half, u_max_full - half + step * 0.5, step)
    v_centers = np.arange(v_min_full + half, v_max_full - half + step * 0.5, step)
    
    if len(u_centers) == 0 or len(v_centers) == 0:
        return empty('no_verticality_grid')
    
    n_u = len(u_centers)
    n_v = len(v_centers)
    
    # Step 5: 向量化 grid assignment
    # 每个表面点分配到对应的 grid cell（考虑 strip width）
    half_width = params.ruler_width_m / 2.0
    
    # 计算每个表面点属于哪个 strip（横向）
    # strip index = floor((u - u_min) / strip_step)
    strip_step = max(float(params.strip_step_m), step)
    strip_idx = np.floor((surf_u - u_min_full) / strip_step).astype(np.int64)
    
    # 计算每个表面点属于哪个 scan window（竖向）
    # 使用 2m 滑动窗口：每个点的 v 坐标决定它属于哪些 window
    # 简化：按 base grid cell 分组，每个 cell 代表一个测量位置
    cell_u_idx = np.clip(np.floor((surf_u - u_min_full) / step).astype(np.int64), 0, n_u - 1)
    cell_v_idx = np.clip(np.floor((surf_v - v_min_full) / step).astype(np.int64), 0, n_v - 1)
    cell_idx = cell_v_idx * n_u + cell_u_idx
    
    # Step 6: 向量化分组统计
    # 使用 argsort + reduceat 进行高效分组
    order = np.argsort(cell_idx, kind='stable')
    sorted_cells = cell_idx[order]
    sorted_u = surf_u[order]
    sorted_v = surf_v[order]
    sorted_ids = surf_ids[order]
    
    # 找到每个 cell 的边界
    if len(sorted_cells) == 0:
        return empty('no_cells_after_sort')
    
    cell_starts = np.r_[0, np.flatnonzero(sorted_cells[1:] != sorted_cells[:-1]) + 1]
    cell_ends = np.r_[cell_starts[1:], len(sorted_cells)]
    unique_cells = sorted_cells[cell_starts]
    
    # Step 7: 遍历有数据的 cell（数量远小于总窗口数）
    rows = []
    base_u_min = float((points @ u_axis).min())
    base_v_min = float((points @ v_axis).min())
    
    total_cells = len(unique_cells)
    processed = 0
    
    for i, (cell, start, end) in enumerate(zip(unique_cells, cell_starts, cell_ends)):
        n_points = end - start
        if n_points < params.min_points:
            continue
        
        cell_u_vals = sorted_u[start:end]
        cell_v_vals = sorted_v[start:end]
        cell_ids = sorted_ids[start:end]
        
        # 检查 cell 内是否有足够的竖向范围（至少一半靠尺长度）
        v_range = float(cell_v_vals.max() - cell_v_vals.min())
        if v_range < params.ruler_length_m * 0.5:
            continue
        
        # 【核心计算】垂直度 = 表面点的水平偏移半幅值
        # 2m垂直检测尺放在 cell 中心，测量表面点相对尺身的最大横向偏离
        u_median = float(np.median(cell_u_vals))
        deviations = np.abs(cell_u_vals - u_median)
        max_dev_idx = int(np.argmax(deviations))
        max_dev = float(deviations[max_dev_idx])
        max_dev_mm = max_dev * 1000.0
        
        # 角度：墙面相对竖直的倾斜（首尾两点连线）
        if len(cell_v_vals) >= 2:
            sort_v = np.argsort(cell_v_vals)
            v_ends = cell_v_vals[sort_v][[0, -1]]
            u_ends = cell_u_vals[sort_v][[0, -1]]
            dv = float(v_ends[-1] - v_ends[0])
            if dv > 1e-6:
                tilt_slope = (u_ends[-1] - u_ends[0]) / dv
                angle_deg = float(np.degrees(np.arctan(abs(tilt_slope))))
            else:
                angle_deg = 0.0
        else:
            angle_deg = 0.0
        
        verticality_pass = np.isfinite(max_dev_mm) and max_dev_mm <= params.verticality_limit_mm
        
        # Cell center world coordinates
        ci = cell % n_u
        cj = cell // n_u
        center_u = float(u_centers[ci])
        center_v = float(v_centers[cj])
        center_xyz = origin + center_u * horizontal_axis + center_v * vertical_axis
        
        # Snap to base grid（与平整度一致）
        key, snap_u, snap_v, snap_xyz, snap_distance = _snap_window_to_base_grid(
            center_xyz.tolist(), origin, u_axis, v_axis, base_u_min, base_v_min, step)
        
        rows.append({
            'grid_key': key,
            'cell_u': int(key[0]),
            'cell_v': int(key[1]),
            'center_xyz': [float(x) for x in np.asarray(snap_xyz, dtype=float)],
            'center_uv_base': (snap_u, snap_v),
            'verticality_deviation_mm_2m': max_dev_mm,
            'verticality_angle_deg': angle_deg,
            'verticality_pass': bool(verticality_pass),
            'hole_ratio': 0.0,
            'coverage_valid': True,
            'effective_point_count': n_points,
            'snap_distance_m': snap_distance,
            'pivot_source_ids': [int(cell_ids[0]), int(cell_ids[-1])],
            'depression_source_id': int(cell_ids[max_dev_idx]),
        })
        
        processed += 1
        if processed % 10000 == 0:
            print(f'[PCFD] verticality.progress cells={processed}/{total_cells}', flush=True)
    
    finite_rows = [r for r in rows if np.isfinite(r['verticality_deviation_mm_2m'])]
    pass_rows = [r['verticality_pass'] for r in finite_rows]
    
    elapsed = time.perf_counter() - started
    print(f'[PCFD] verticality.summary '
          f'windows_total={len(v_centers) * len(u_centers)} '
          f'valid_windows={len(rows)} '
          f'rows={len(finite_rows)} '
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


def compute_ruler_quality(points, raw_ids, plane_model, origin, u_axis, v_axis, params=None):
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

    all_windows = []
    for angle in params.flatness_angles_deg:
        angle_f = float(angle)
        windows = direction_results.get(angle_f, [])
        for w in windows:
            w['source_direction'] = angle_f
            all_windows.append(w)

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
                'u_range_m': u_range_full, 'v_range_m': v_range_full,
                'u_min_m': u_min_full, 'u_max_m': u_max_full,
                'v_min_m': v_min_full, 'v_max_m': v_max_full,
            },
            'direction_ranges': {str(k): {'along_m': v[0], 'across_m': v[1]} 
                                for k, v in direction_ranges.items()},
        }

    windows_by_direction = {}
    for angle in params.flatness_angles_deg:
        angle_f = float(angle)
        windows_by_direction[angle_f] = direction_results.get(angle_f, [])
    
    rows = _aggregate_star_rows(
        windows_by_direction, origin, u_axis, v_axis,
        u_min_full, v_min_full, params)

    valid_rows = [r for r in rows if r.get('coverage_valid', False)]
    finite_rows = [r for r in valid_rows if np.isfinite(r['flatness_gap_mm'])]
    raw_rows = [r for r in rows if np.isfinite(r['flatness_raw_max_gap_mm'])]

    size = max(float(params.interval_size_m), 1e-6)
    intervals = []

    if rows:
        centers_array = np.array([r['center_xyz'] for r in rows])
        rel_centers = centers_array - origin
        center_u = rel_centers @ u_axis
        center_v = rel_centers @ v_axis
        v_span = v_max_full - v_min_full
        n_intervals = max(1, int(np.ceil(v_span / size)))
        edges = np.linspace(v_min_full, v_max_full, n_intervals + 1)
        interval_ids = np.minimum(np.searchsorted(edges, center_v, side='right') - 1,
                                  n_intervals - 1)
        for i in range(n_intervals):
            interval_rows = [r for r, iid in zip(rows, interval_ids) if iid == i]
            # 区间点数必须来自质量域原始点，而不是检测窗口有效点数之和。
            # 后者会重复计算同一个点，且会让每个区间看起来拥有整幅立面点数。
            v_lo, v_hi = float(edges[i]), float(edges[i + 1])
            point_v = (points - origin) @ v_axis
            point_mask = (point_v >= v_lo) & (
                (point_v < v_hi) if i < n_intervals - 1 else (point_v <= v_hi)
            )
            world_z = points[point_mask, 2] if np.any(point_mask) else np.empty(0)
            valid_interval_rows = [r for r in interval_rows if r.get('coverage_valid', False)]
            gaps = [r['flatness_gap_mm'] for r in valid_interval_rows if np.isfinite(r['flatness_gap_mm'])]
            raw_gaps = [r['flatness_raw_max_gap_mm'] for r in interval_rows if np.isfinite(r['flatness_raw_max_gap_mm'])]
            passes = [r['flatness_pass'] for r in valid_interval_rows]
            intervals.append({
                'label': (f'{float(world_z.min()):.2f}–{float(world_z.max()):.2f}m'
                          if len(world_z) else f'{v_lo:.2f}–{v_hi:.2f}m'),
                'v_min_m': v_lo, 'v_max_m': v_hi,
                'world_height_min_m': float(world_z.min()) if len(world_z) else None,
                'world_height_max_m': float(world_z.max()) if len(world_z) else None,
                'window_count': len(interval_rows),
                'valid_window_count': len(valid_interval_rows),
                'point_count': int(np.count_nonzero(point_mask)),
                'flatness_max_gap_mm': float(np.nanmax(gaps)) if gaps else 0.0,
                'flatness_raw_max_gap_mm': float(np.nanmax(raw_gaps)) if raw_gaps else 0.0,
                'flatness_pass_rate': float(np.mean(passes)) if passes else 0.0,
                'quality_pass_rate': float(np.mean(passes)) if passes else 0.0,
                'status': 'ok' if gaps else 'no_valid_window',
            })
        intervals.sort(key=lambda x: x['v_min_m'])

    # Use CORRECTED verticality implementation
    verticality = _compute_verticality_corrected(points, raw_ids, plane_model, u_axis, v_axis, origin, params)
    verticality_rows = {}
    for item in verticality.get('rows', []):
        key = item.get('grid_key')
        if key is not None:
            verticality_rows[tuple(key)] = item

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

    return {
        'ok': True,
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
            'u_range_m': u_range_full, 'v_range_m': v_range_full,
            'u_min_m': u_min_full, 'u_max_m': u_max_full,
            'v_min_m': v_min_full, 'v_max_m': v_max_full,
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