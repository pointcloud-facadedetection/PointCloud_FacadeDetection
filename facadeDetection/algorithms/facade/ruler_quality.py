"""靠尺法（模拟物理靠尺）质量评估：米字形平整度 + I 形靠尺垂直度。

职责边界：
  - 本模块只实现【靠尺法】自身的指标；

垂直度物理模型（上下水平靠线法，对应现实 2m 靠尺贴墙）：
    2m 靠尺竖直贴墙时，尺身上下两端与墙面接触形成两条水平靠线
    （各取窗口上/下 0.1m 靠线带内最凸出的表面点为接触点）；两接触点
    连线与尺宽方向张成局部窗口平面，其相对铅垂线的倾斜即该窗口垂直度：
      θ = arcsin(|n_w · g|)，verticality_mm = tan(θ) × 2000。
    上下接触点即 pivot_source_ids（导致窗口倾斜的关键接触点）；缺陷记录
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, asdict, field
import atexit
import shutil
import tempfile
import threading
import time
import os
import numpy as np

from .ruler_flatness_3d import (
    prepare_surface, ruler_at, fit_line,
    sor_mask_local, sor_mask_grid, sor_mask_exact,
)
from .local_baseline import (
    fit_interval_baseline_planes,
    compute_block_signed_distances_mm,
    find_block_for_v,
)

_EMPTY_SOURCE_IDS = np.empty(0, dtype=np.int64)


# =============================================================================
# 常驻进程池与共享数组落盘
# =============================================================================
_POOL_LOCK = threading.Lock()
_SHARED_POOL = None
_SHARED_POOL_WORKERS = 0


def _get_shared_process_pool(workers):
    """取常驻 ProcessPoolExecutor；worker 数变化时重建。"""
    global _SHARED_POOL, _SHARED_POOL_WORKERS
    with _POOL_LOCK:
        if _SHARED_POOL is None or _SHARED_POOL_WORKERS != int(workers):
            if _SHARED_POOL is not None:
                _SHARED_POOL.shutdown(wait=False)
            _SHARED_POOL = ProcessPoolExecutor(max_workers=int(workers))
            _SHARED_POOL_WORKERS = int(workers)
        return _SHARED_POOL


def _reset_shared_process_pool():
    global _SHARED_POOL, _SHARED_POOL_WORKERS
    with _POOL_LOCK:
        pool, _SHARED_POOL, _SHARED_POOL_WORKERS = _SHARED_POOL, None, 0
    if pool is not None:
        try:
            pool.shutdown(wait=False)
        except Exception:
            pass


atexit.register(_reset_shared_process_pool)


def _pool_map(workers, func, args_list):
    """常驻池 map；子进程异常死亡（BrokenProcessPool）时重建池重试一次。"""
    pool = _get_shared_process_pool(workers)
    try:
        return list(pool.map(func, args_list))
    except BrokenProcessPool:
        _reset_shared_process_pool()
        pool = _get_shared_process_pool(workers)
        return list(pool.map(func, args_list))


class _SharedArrayFiles:
    """进程模式共享数组载体：大数组落临时 .npy，子进程 mmap 自取。"""

    def __init__(self):
        self._dir = tempfile.mkdtemp(prefix='pcfd_quality_')

    def dump(self, name, array):
        path = os.path.join(self._dir, f'{name}.npy')
        with open(path, 'wb') as stream:
            np.save(stream, array)
        return path

    def cleanup(self):
        shutil.rmtree(self._dir, ignore_errors=True)


@dataclass(frozen=True)
class RulerQualityParameters:
    ruler_length_m: float = 2.0
    ruler_width_m: float = .055
    scan_step_m: float = 1.0
    strip_step_m: float = .055
    flatness_angles_deg: tuple = (0., 45., 90., 135.)
    select_band_m: float = .01
    hole_band_m: float = .02
    bin_size_m: float = .04
    top_q: float = 1.
    flatness_limit_mm: float = 8.
    verticality_limit_mm: float = 4.
    verticality_bin_size_m: float = 1.0
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
    step = max(float(step), 1e-9)
    if hi - lo < length:
        return np.array([(lo + hi) / 2.0]), np.array([True])
    first = lo + half
    last = hi - half
    centers = np.arange(first, last + 1e-9, step, dtype=float)
    if centers.size == 0 or abs(float(centers[-1]) - last) > 1e-6:
        centers = np.concatenate((centers, np.array([last], dtype=float)))
    clipped = np.zeros(centers.size, dtype=bool)
    clipped[-1] = True
    return centers, clipped


def _full_window_centers(lo, hi, step, length):
    """Generate sliding-window centers and force the final edge position."""
    lo, hi = float(lo), float(hi)
    length = max(float(length), 1e-9)
    step = max(float(step), 1e-9)
    if hi - lo <= length:
        return np.asarray([(lo + hi) * 0.5], dtype=float)
    first, last = lo + length * 0.5, hi - length * 0.5
    centers = np.arange(first, last + 1e-9, step, dtype=float)
    if centers.size == 0 or abs(float(centers[-1]) - last) > 1e-7:
        centers = np.r_[centers, last]
    return centers


def _project_direction(points, plane_model, origin, u_axis, v_axis, angle_deg):
    """Project points to a specific ruler direction coordinate system."""
    rad = np.radians(float(angle_deg))
    d3 = np.cos(rad) * u_axis + np.sin(rad) * v_axis
    d3 = _unit(d3)
    normal = plane_model[:3] / np.linalg.norm(plane_model[:3])
    q3 = np.cross(normal, d3)
    q3 = _unit(q3)

    rel = points - origin
    along = rel @ d3
    across = rel @ q3
    w = rel @ normal

    return along, across, w, d3, q3


def _global_sor_keep(u_all, v_all, dist_mm, params) -> np.ndarray:
    """全局一次性 SOR 去噪掩码（与 strip/方向无关，全链路共用）。

    sor_mask_local 的统计仅依赖 w；grid/exact 依赖横向坐标，此处用全局
    u（strip 级与窗口级计算的并集），语义一致且避免 4 方向 × strip 级
    及窗口重叠造成的重复计算。
    """
    n = len(u_all)
    if not params.sor_enabled or n <= 16:
        return np.ones(n, dtype=bool)
    w = dist_mm / 1000.0
    method = params.sor_method
    if method == 'grid':
        keep = sor_mask_grid(u_all, w, params.sor_k, params.sor_sigma,
                             w_weight=params.sor_w_weight)
    elif method == 'exact':
        keep = sor_mask_exact(u_all, w, params.sor_k, params.sor_sigma,
                              w_weight=params.sor_w_weight)
    else:
        keep = sor_mask_local(u_all, w, params.sor_k, params.sor_sigma)
    if np.sum(keep) < 3:
        keep[:] = True
    return keep


def _build_strips_directional(along, across, w, raw_ids, angle, params,
                              keep=None):
    """Build strips and window centers for a given direction.

    keep: 与 points 同序的全局 SOR 掩码，随 across_order 同步重排，
          避免 strip 切片时索引错位。
    """
    across_order = np.argsort(across, kind='stable')
    across_sorted = across[across_order]
    along_sorted = along[across_order]
    w_sorted = w[across_order]
    ids_sorted = raw_ids[across_order]
    keep_sorted = (np.asarray(keep, dtype=bool)[across_order]
                   if keep is not None else None)

    q_min, q_max = across_sorted.min(), across_sorted.max()
    empty_ret = ([], np.empty((0, 2), float), np.empty(0, np.int64),
                 across_sorted, along_sorted, w_sorted, ids_sorted,
                 np.empty(0, dtype=bool),
                 np.zeros(len(across_sorted), dtype=bool)
                 if keep_sorted is None else keep_sorted)
    if q_max - q_min < params.ruler_width_m:
        return empty_ret

    half_width = params.ruler_width_m / 2.0
    strip_step = max(float(params.strip_step_m), 1e-6)
    strip_centers = _full_window_centers(q_min, q_max, strip_step,
                                         params.ruler_width_m)
    if len(strip_centers) == 0:
        return empty_ret

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
            'along_min': float(a_min),
            'along_max': float(a_max),
        })

    if not strips:
        return empty_ret

    centres = []
    strip_indices = []
    scan_clipped = []
    for strip in strips:
        lo = max(float(strip['along_min']), float(along_sorted.min()))
        hi = min(float(strip['along_max']), float(along_sorted.max()))
        scan, clipped = _scan_grid(lo, hi, params.scan_step_m, params.ruler_length_m)
        if scan.size:
            centres.extend((float(a), float(strip['across_center'])) for a in scan)
            strip_indices.extend([strip['center_index']] * len(scan))
            scan_clipped.extend(clipped.tolist())

    centres_grid = np.asarray(centres, dtype=float).reshape(-1, 2)
    strip_idx = np.asarray(strip_indices, dtype=np.int64)
    scan_clipped_arr = np.asarray(scan_clipped, dtype=bool)
    return strips, centres_grid, strip_idx, across_sorted, along_sorted, \
        w_sorted, ids_sorted, scan_clipped_arr, keep_sorted


def _prepare_strip_surface_directional(*, along_sorted, across_sorted, w_sorted,
                                        raw_ids_sorted, strip, d3, q3,
                                        plane_model, origin, params,
                                        keep_sorted=None):
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
            plane_model=tuple(plane_model), plane_origin=origin,
            precomputed_keep=(keep_sorted[lo_idx:hi_idx][a_mask]
                              if keep_sorted is not None else None))
    except (ValueError, IndexError, FloatingPointError) as e:
        return None
    return surface


def _ruler_at_for_centre(*, surface, centre_uv, d3, q3, origin, angle, params):
    local_center = float(centre_uv[0])
    result = ruler_at(
        surface, local_center, params.ruler_length_m,
        threshold_mm=params.flatness_limit_mm)
    ruler_half = params.ruler_length_m / 2.0
    u_center = local_center
    u_mask = ((surface.u >= u_center - ruler_half) &
              (surface.u <= u_center + ruler_half) & surface.surf)
    along_c = centre_uv[0]
    across_c = centre_uv[1]
    center_xyz = origin + along_c * d3 + across_c * q3
    row = {
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
    if 'ruler_defect_grid' in result:
        row['ruler_defect_grid'] = result['ruler_defect_grid']
        row['defect_point_indices'] = result.get('defect_point_indices', [])
        row['defect_values_mm'] = result.get('defect_values_mm', [])
    return row


def _snap_window_to_base_grid(center_xyz, origin, u_axis, v_axis, u0, v0,
                              u_step=None, v_step=None):
    """把窗口中心吸附到 (u, v) 双步长网格，返回网格 key 与吸附后坐标。"""
    center_xyz = np.asarray(center_xyz, dtype=float)
    rel = center_xyz - origin
    base_u = float(rel @ u_axis)
    base_v = float(rel @ v_axis)
    if u_step is None and v_step is None:
        raise ValueError('_snap_window_to_base_grid 需要至少一个步长')
    u_step = max(float(u_step if u_step is not None else v_step), 1e-6)
    v_step = max(float(v_step if v_step is not None else u_step), 1e-6)
    ku = int(np.round((base_u - u0) / u_step))
    kv = int(np.round((base_v - v0) / v_step))
    snap_u = float(u0 + ku * u_step)
    snap_v = float(v0 + kv * v_step)
    snap_xyz = origin + snap_u * u_axis + snap_v * v_axis
    distance = float(np.hypot(base_u - snap_u, base_v - snap_v))
    return (ku, kv), snap_u, snap_v, snap_xyz, distance


def _aggregate_star_rows(windows_by_direction, origin, u_axis, v_axis,
                         u_min_full, v_min_full, params):
    """聚合四方向结果 into shared 2m x 2m star windows."""
    all_windows = []
    for angle, windows in windows_by_direction.items():
        for w in windows:
            if not w.get('ok'):
                continue
            center = w.get('center_xyz')
            if center is None or not np.all(np.isfinite(center)):
                continue
            gap = w.get('gap_mm', np.nan)
            if not np.isfinite(gap):
                continue
            all_windows.append({
                'direction_deg': float(angle),
                'gap_mm': float(gap),
                'signed_gap_mm': float(w.get('signed_gap_mm', np.nan)),
                'coverage_valid': bool(w.get('coverage_valid', False)),
                'center_xyz': np.asarray(center, dtype=float),
                'center_uv': w.get('center_uv', (np.nan, np.nan)),
                'effective_point_count': int(w.get('effective_point_count', 0)),
                'hole_ratio': float(w.get('hole_ratio', 1.0)),
                'angle_deg': float(w.get('angle_deg', np.nan)),
                'depression_source_id': int(w.get('depression_source_id', -1)),
                'pivot_source_ids': list(w.get('pivot_source_ids', [-1, -1])),
                'covered_source_ids': np.asarray(w.get('covered_source_ids', _EMPTY_SOURCE_IDS), dtype=np.int64),
                'ruler_defect_grid': w.get('ruler_defect_grid'),
                'u_center': float(w.get('u_center', np.nan)),
                'defect_point_indices': w.get('defect_point_indices', []),
                'defect_values_mm': w.get('defect_values_mm', []),
            })

    if not all_windows:
        return []

    centers = np.array([w['center_xyz'] for w in all_windows])
    rel = centers - origin
    base_u = rel @ u_axis
    base_v = rel @ v_axis

    step_u = max(float(params.ruler_width_m), 1e-6)
    step_v = max(float(params.ruler_length_m) / 2.0, 1e-6)

    ku = np.round((base_u - u_min_full) / step_u).astype(np.int64)
    kv = np.round((base_v - v_min_full) / step_v).astype(np.int64)

    grid_groups = {}
    for i, (u, v) in enumerate(zip(ku, kv)):
        key = (int(u), int(v))
        grid_groups.setdefault(key, []).append(all_windows[i])

    rows = []
    for key, members in grid_groups.items():
        dir_groups = {}
        for m in members:
            dir_groups.setdefault(m['direction_deg'], []).append(m)

        directional = []
        for angle in params.flatness_angles_deg:
            angle_f = float(angle)
            if angle_f not in dir_groups:
                continue
            best = max(dir_groups[angle_f],
                      key=lambda x: x['gap_mm'] if np.isfinite(x['gap_mm']) else -np.inf)
            directional.append(best)

        if not directional:
            continue

        snap_u = u_min_full + key[0] * step_u
        snap_v = v_min_full + key[1] * step_v
        snap_xyz = origin + snap_u * u_axis + snap_v * v_axis

        valid_gaps = [d['gap_mm'] for d in directional if d['coverage_valid'] and np.isfinite(d['gap_mm'])]
        all_gaps = [d['gap_mm'] for d in directional if np.isfinite(d['gap_mm'])]

        if valid_gaps:
            flatness_gap = max(valid_gaps)
        else:
            flatness_gap = max(all_gaps) if all_gaps else np.nan

        quality_valid = bool(valid_gaps)
        flatness_pass = bool(quality_valid and np.isfinite(flatness_gap)
                             and flatness_gap <= params.flatness_limit_mm)

        finite_members = [d for d in directional if np.isfinite(d['gap_mm'])]
        valid_members = [d for d in finite_members if d['coverage_valid']]
        dominant = max(valid_members if valid_members else finite_members,
                       key=lambda x: x['gap_mm'] if np.isfinite(x['gap_mm']) else -np.inf,
                       default=None)

        row = {
            'window_id': len(rows),
            'grid_key': key,
            'cell_u': key[0], 'cell_v': key[1],
            'center_xyz': [float(x) for x in snap_xyz],
            'center_uv_base': (float(snap_u), float(snap_v)),
            'flatness_gap_mm': flatness_gap,
            'flatness_raw_max_gap_mm': max(all_gaps) if all_gaps else np.nan,
            'flatness_pass': flatness_pass,
            'coverage_valid': quality_valid,
            'direction_count': len(finite_members),
            'quality_direction_count': len(valid_members),
            'hole_ratio': min([d['hole_ratio'] for d in finite_members]) if finite_members else 1.0,
            'hole_length_m': max([d.get('hole_length_m', 0.0) for d in finite_members]) if finite_members else 0.0,
            'effective_point_count': max([d['effective_point_count'] for d in finite_members]) if finite_members else 0,
            'direction_deg': float(dominant['direction_deg']) if dominant else np.nan,
            'depression_source_id': int(dominant['depression_source_id']) if dominant else -1,
            'pivot_source_ids': list(dominant['pivot_source_ids']) if dominant else [-1, -1],
            'center_uv': (float(snap_u), float(snap_v)),
            'angle_deg': float(dominant['angle_deg']) if dominant else np.nan,
            'source_direction': float(dominant['direction_deg']) if dominant else np.nan,
            'snap_distance_m': 0.0,
            'u_center': float(dominant.get('u_center', np.nan)) if dominant else np.nan,
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
        if dominant and dominant.get('ruler_defect_grid') is not None:
            row['ruler_defect_grid'] = dominant['ruler_defect_grid']
            row['defect_point_indices'] = dominant.get('defect_point_indices', [])
            row['defect_values_mm'] = dominant.get('defect_values_mm', [])
        all_defect_ids = []
        for d in directional:
            ids_local = d.get('defect_point_indices', [])
            vals_local = d.get('defect_values_mm', [])
            if ids_local:
                all_defect_ids.extend(zip(ids_local, vals_local))
        if all_defect_ids:
            seen = set()
            merged_ids, merged_vals = [], []
            for sid, val in all_defect_ids:
                if sid not in seen:
                    seen.add(sid)
                    merged_ids.append(sid)
                    merged_vals.append(val)
            row['all_defect_source_ids'] = merged_ids
            row['all_defect_values_mm'] = merged_vals
        if params.retain_covered_source_ids:
            row['covered_source_ids'] = np.unique(np.concatenate([
                d['covered_source_ids'] for d in directional
                if len(d['covered_source_ids'])
            ])).astype(np.int64).tolist() if any(
                len(d['covered_source_ids']) for d in directional) else []
        rows.append(row)

    rows.sort(key=lambda item: (item['center_uv_base'][1], item['center_uv_base'][0]))
    for idx, row in enumerate(rows):
        row['window_id'] = idx

    return rows


# =============================================================================
# 靠尺法垂直度 Worker（strip 级 SOR 复用 + 上下水平靠线模型）
# =============================================================================
def _verticality_strip_worker(args):
    """处理单个横向 strip 的所有竖直窗口（靠尺法垂直度）。

    ★ SOR 优化：keep 掩码在 strip 级由调用方预计算（全局一次 SOR 的
      strip 切片），窗口内直接复用，不再逐窗口重复统计。
    ★ 垂直度物理模型：上下水平靠线法 —— 窗口上/下 0.1m 靠线带内最凸
      出的表面点为靠尺实际接触点；两接触点连线与尺宽方向张成局部窗口
      平面，其相对铅垂线的倾斜即该窗口垂直度。靠线带点不足时回退为
      全点平面拟合。
    """
    (uj, u_c,
     surf_u, surf_v, surf_ids, surf_w, surf_keep,
     v_centers, v_lo_bounds, v_hi_bounds,
     horizontal_axis, vertical_axis, origin,
     u_axis, v_axis, base_u_min, base_v_min, v_step, u_step,
     half_width, min_points, ruler_length_m, verticality_limit_mm,
     sor_enabled, sor_k, sor_sigma, sor_w_weight, sor_method,
     select_band, hole_band, normal) = args

    # ---- 在全局有序的 surf_u 上快速定位 strip ----
    lo_idx = np.searchsorted(surf_u, u_c - half_width, side='left')
    hi_idx = np.searchsorted(surf_u, u_c + half_width, side='right')

    if hi_idx - lo_idx < min_points:
        return uj, []

    # 提取 strip 数据
    strip_u = surf_u[lo_idx:hi_idx]
    strip_v = surf_v[lo_idx:hi_idx]
    strip_ids_local = surf_ids[lo_idx:hi_idx]
    strip_w = surf_w[lo_idx:hi_idx]
    strip_keep = surf_keep[lo_idx:hi_idx]

    # Strip 内按 V (vertical) 排序，用于沿竖直方向滑动窗口
    v_order = np.argsort(strip_v, kind='quicksort')
    strip_v = strip_v[v_order]
    strip_u = strip_u[v_order]
    strip_ids_local = strip_ids_local[v_order]
    strip_w = strip_w[v_order]
    strip_keep = strip_keep[v_order]

    # 向量化计算所有 v 窗口边界
    v_win_lo = np.searchsorted(strip_v, v_lo_bounds, side='left')
    v_win_hi = np.searchsorted(strip_v, v_hi_bounds, side='right')
    n_win_points = v_win_hi - v_win_lo

    valid_mask = n_win_points >= min_points
    if not np.any(valid_mask):
        return uj, []

    # ---- 逐窗口独立测量（SOR 掩码 strip 级复用）----
    rows = []
    for vi in np.flatnonzero(valid_mask):
        wlo = v_win_lo[vi]
        whi = v_win_hi[vi]
        n_win = whi - wlo

        win_u = strip_u[wlo:whi]  # H (horizontal)
        win_v = strip_v[wlo:whi]  # V (vertical)
        win_w = strip_w[wlo:whi]  # N (signed distance to block plane)
        win_ids = strip_ids_local[wlo:whi]

        # ================================================================
        #  统一数据清洗层（SOR 复用全局预计算掩码）
        # ================================================================
        keep = strip_keep[wlo:whi].copy()
        if np.sum(keep) < 3:
            keep[:] = True

        # 2) 表面点筛选（select_band 等价）：拟合 win_w vs win_v 的参考线
        try:
            a_ref, b_ref = fit_line(win_v[keep], win_w[keep])
        except (np.linalg.LinAlgError, ValueError):
            continue

        residual = win_w - (a_ref * win_v + b_ref)
        surf = keep & (np.abs(residual) <= select_band)
        if np.sum(surf) < 3:
            surf = keep  # fallback：放宽到 keep 点

        # 3) 孔洞检测（hole_band）
        effective_hole_band = max(hole_band, select_band)
        v0 = float(win_v.min())
        bin_size = max(v_step * 0.5, 0.02)
        bins = np.floor((win_v - v0) / bin_size).astype(np.int64)
        solid_mask = np.zeros(int(bins.max()) + 1, dtype=bool)
        solid_mask[bins[keep & (np.abs(residual) <= effective_hole_band)]] = True

        solid_bins = np.flatnonzero(solid_mask)
        final_mask = surf & np.isin(bins, solid_bins)
        if np.sum(final_mask) < 3:
            continue

        # ================================================================
        #  垂直度：上下两条水平靠线 → 窗口平面 vs 铅垂线（靠尺法物理模型）
        #
        #  现实操作：2m 靠尺竖直贴墙，尺身上下两端与墙面接触形成两条水
        #  平靠线；两靠线张成该窗口段的局部平面，其相对铅垂线的倾斜即
        #  垂直度。靠线带内最凸出的表面点为实际接触点（w 外法向为正，
        #  正值=凸起）。
        # ================================================================
        h_ax = np.asarray(horizontal_axis, dtype=np.float64)
        v_ax_l = np.asarray(vertical_axis, dtype=np.float64)
        n_vec = np.asarray(normal, dtype=np.float64)

        band_h = min(0.1, ruler_length_m * 0.1)   # 靠线带高：上下各 0.1m
        v_hi_w = float(win_v.max())
        v_lo_w = float(win_v.min())
        top_m = final_mask & (win_v >= v_hi_w - band_h)
        bot_m = final_mask & (win_v <= v_lo_w + band_h)

        top_count = int(np.count_nonzero(top_m))
        bot_count = int(np.count_nonzero(bot_m))
        if top_count < 1 or bot_count < 1:
            # 上下任一侧无表面点：该窗口无法放置靠尺，跳过
            continue

        # 单点接触：靠线带内 w 最大点即为该侧实际接触点
        top_pos = np.flatnonzero(top_m)
        bot_pos = np.flatnonzero(bot_m)
        top_idx = int(top_pos[int(np.argmax(win_w[top_m]))])
        bot_idx = int(bot_pos[int(np.argmax(win_w[bot_m]))])

        p_top = (origin + win_u[top_idx] * h_ax + win_v[top_idx] * v_ax_l
                 + win_w[top_idx] * n_vec)
        p_bot = (origin + win_u[bot_idx] * h_ax + win_v[bot_idx] * v_ax_l
                 + win_w[bot_idx] * n_vec)
        span = p_top - p_bot
        span_norm = float(np.linalg.norm(span))
        if span_norm < 1e-9:
            # 上下接触点重合：垂直度无法定义
            continue
        # 窗口平面 = 尺宽方向(h_ax) 与两接触点连线张成
        n_w = np.cross(h_ax, span)
        nn = float(np.linalg.norm(n_w))
        if nn < 1e-12:
            continue
        n_w = n_w / nn
        if float(np.dot(n_w, n_vec)) < 0:
            n_w = -n_w
        plane_d = -float(np.dot(n_w, p_bot))
        contact_pos = (bot_idx, top_idx)   # 窗口内索引（下、上）

        gravity = np.asarray((0., 0., 1.), dtype=np.float64)
        cos_ng = abs(float(np.dot(n_w, gravity)))
        theta = float(np.arcsin(min(cos_ng, 1.0)))
        deviation_mm = float(np.tan(theta) * ruler_length_m * 1000.0)
        angle_deg = float(np.degrees(theta))
        verticality_pass = (np.isfinite(deviation_mm)
                            and deviation_mm <= verticality_limit_mm)

        # ---- 孔洞覆盖统计（2m 靠尺范围内）----
        ruler_half = ruler_length_m / 2.0
        v_center = float(v_centers[vi])
        v_lo_ruler = v_center - ruler_half
        v_hi_ruler = v_center + ruler_half

        pad = np.r_[True, solid_mask, True]
        hs = np.flatnonzero(~pad[1:-1] & pad[:-2])
        he = np.flatnonzero(~pad[1:-1] & pad[2:]) + 1
        hole_length = 0.0
        if hs.size:
            h_lo = v0 + hs * bin_size
            h_hi = v0 + he * bin_size
            overlap = np.clip(np.minimum(h_hi, v_hi_ruler) - np.maximum(h_lo, v_lo_ruler), 0, None)
            hole_length = float(overlap.sum())
        hole_ratio = hole_length / ruler_length_m if ruler_length_m > 0 else 1.0

        center_xyz = origin + u_c * h_ax + v_center * v_ax_l

        key, snap_u, snap_v, snap_xyz, snap_distance = _snap_window_to_base_grid(
            center_xyz.tolist(), origin, u_axis, v_axis, base_u_min, base_v_min,
            u_step, v_step)

        # 靠点 = 上下靠线接触点（导致窗口倾斜的关键接触点，随 pivot 存储）
        bot_idx, top_idx = contact_pos
        dep_source_id = int(win_ids[top_idx])
        dep_xyz = (origin + win_u[top_idx] * h_ax + win_v[top_idx] * v_ax_l
                   + win_w[top_idx] * n_vec).tolist()

        # 底部 0 基准：以下靠线接触点为 0，上接触点相对偏移的正负
        # 判定外凸（+，外倾）/ 内陷（−，内倾）
        rel_top_mm = float((win_w[top_idx] - win_w[bot_idx]) * 1000.0)
        outward = rel_top_mm > 0.0

        # 垂直度缺陷专用数组（仅在超阈值时填充；存储导致倾斜的上接触点）
        verticality_defect_point_indices = []
        verticality_defect_values_mm = []
        verticality_defect_types = []
        verticality_defect_point_xyzs = []
        if not verticality_pass:
            signed_vert = float(deviation_mm if outward else -deviation_mm)
            verticality_defect_point_indices = [dep_source_id]
            verticality_defect_values_mm = [signed_vert]
            verticality_defect_types = [
                'outward_lean' if outward else 'inward_sag']
            verticality_defect_point_xyzs = [dep_xyz]

        rows.append({
            'grid_key': key,
            'cell_u': int(key[0]), 'cell_v': int(key[1]),
            'center_xyz': [float(x) for x in np.asarray(snap_xyz, dtype=float)],
            'center_uv_base': (snap_u, snap_v),
            'verticality_deviation_mm': deviation_mm,
            'verticality_angle_deg': angle_deg,
            'verticality_pass': bool(verticality_pass),
            'hole_ratio': float(hole_ratio),
            'coverage_valid': hole_ratio <= 0.2,
            'effective_point_count': int(np.sum(final_mask)),
            'snap_distance_m': snap_distance,
            'pivot_source_ids': [int(win_ids[bot_idx]), int(win_ids[top_idx])],
            'depression_source_id': dep_source_id,
            'trend_slope': float(np.tan(theta)),
            'trend_intercept': float(plane_d),
            'n_sor_removed': int(np.sum(~keep)),
            'n_used': int(np.sum(final_mask)),
            'verticality_defect_point_indices': verticality_defect_point_indices,
            'verticality_defect_values_mm': verticality_defect_values_mm,
            'verticality_defect_types': verticality_defect_types,
            'verticality_defect_point_xyzs': verticality_defect_point_xyzs,
        })

    return uj, rows


def _compute_verticality(points, raw_ids, plane_model, u_axis, v_axis, origin,
                         params, u_min_full=None, v_min_full=None,
                         precomputed_w_mm=None, global_keep=None):
    """
    靠尺法垂直度 = 每个 2m I 形窗口独立测量，表征该窗口段墙面相对
    铅垂线的倾斜（不使用区间高度块局部基准面；块基准面深度仅用于
    表面点筛选 band）。

    物理模型：上下水平靠线（带内最凸接触点）张成局部窗口平面，
      θ = arcsin(|n_w · g|)，verticality_mm = tan(θ) × ruler_length_m × 1000。
    靠线带点不足时回退为全点平面拟合。
    """
    started = time.perf_counter()

    def empty(reason):
        return {'ok': False, 'reason': reason,
                'verticality_deviation_mm': np.nan,
                'verticality_avg_deviation_mm': np.nan,
                'verticality_max_angle_deg': np.nan,
                'verticality_pass': False,
                'verticality_pass_rate': 0.0,
                'rows': []}

    if not params.verticality_enabled:
        return empty('verticality_disabled')

    plane_model = np.asarray(plane_model, dtype=float)
    if plane_model.shape[0] != 4:
        return empty('invalid_plane_model')

    # ---- Step 1: 真实竖直方向 ----
    normal = plane_model[:3] / np.linalg.norm(plane_model[:3])
    gravity = np.asarray(params.gravity_axis, dtype=float)
    gravity = gravity / np.linalg.norm(gravity)

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

    # ---- Step 2: 投影到 (H, V, N) ----
    rel = points - origin
    u_h = rel @ horizontal_axis
    v_v = rel @ vertical_axis
    w_n = rel @ normal
    if precomputed_w_mm is not None:
        cached = np.asarray(precomputed_w_mm, dtype=float).reshape(-1)
        if cached.shape == (len(points),):
            w_n = cached / 1000.0

    # ---- Step 3: 表面点过滤 ----
    surface_band = float(getattr(params, 'select_band_m', 0.02))
    surf_mask = np.abs(w_n) <= surface_band
    if global_keep is not None:
        gk = np.asarray(global_keep, dtype=bool).reshape(-1)
        if gk.shape == (len(points),):
            surf_mask &= gk
    n_surf = int(np.count_nonzero(surf_mask))
    if n_surf < params.min_points:
        return empty(f'too_few_surface_points:{n_surf}')

    surf_u = u_h[surf_mask]
    surf_v = v_v[surf_mask]
    surf_ids = raw_ids[surf_mask]
    surf_w = w_n[surf_mask]
    surf_keep = (np.asarray(global_keep, dtype=bool)[surf_mask]
                 if global_keep is not None else np.ones(n_surf, dtype=bool))

    v_min = float(surf_v.min())
    v_max = float(surf_v.max())
    u_min = float(surf_u.min())
    u_max = float(surf_u.max())

    if (v_max - v_min) < params.ruler_length_m:
        return empty('verticality_domain_too_short')

    half = params.ruler_length_m / 2.0
    half_width = params.ruler_width_m / 2.0

    # ★ 统一步长：纵向 = 半尺长，横向 = 尺宽
    v_step = max(float(params.verticality_bin_size_m), 1e-6)
    u_step = max(float(params.ruler_width_m), 1e-6)

    v_centers = _full_window_centers(v_min, v_max, v_step, params.ruler_length_m)
    u_centers = _full_window_centers(u_min, u_max, u_step, params.ruler_width_m)

    if len(v_centers) == 0 or len(u_centers) == 0:
        return empty('no_verticality_grid')

    print(f'[PCFD] verticality.grid '
          f'v_centers={len(v_centers)} u_centers={len(u_centers)} '
          f'v_step={v_step:.3f}m u_step={u_step:.3f}m '
          f'total_windows_est={len(v_centers) * len(u_centers)}', flush=True)

    # ---- Step 4: 全局按 H 排序（searchsorted 需要有序数组） ----
    u_order = np.argsort(surf_u, kind='stable')
    surf_u = surf_u[u_order]
    surf_v = surf_v[u_order]
    surf_ids = surf_ids[u_order]
    surf_w = surf_w[u_order]
    surf_keep = surf_keep[u_order]

    base_u_min = (float(u_min_full) if u_min_full is not None
                  else float((points @ u_axis).min()))
    base_v_min = (float(v_min_full) if v_min_full is not None
                  else float((points @ v_axis).min()))

    v_lo_bounds = v_centers - half
    v_hi_bounds = v_centers + half

    # ---- Step 5: 并行处理（进程/线程/串行） ----
    n_workers = min(len(u_centers), params.n_jobs, os.cpu_count() or 1)

    if n_workers > 1 and params.parallel_mode == 'process':
        print(f'[PCFD] verticality.parallel_process workers={n_workers} '
              f'strips={len(u_centers)}', flush=True)
        shared = _SharedArrayFiles()
        try:
            surf_paths = (shared.dump('surf_u', surf_u),
                          shared.dump('surf_v', surf_v),
                          shared.dump('surf_ids', surf_ids),
                          shared.dump('surf_keep', surf_keep),
                          shared.dump('surf_w', surf_w))
            args_list = []
            for uj, u_c in enumerate(u_centers):
                args_list.append((
                    uj, float(u_c), surf_paths,
                    v_centers, v_lo_bounds, v_hi_bounds,
                    horizontal_axis, vertical_axis, origin,
                    u_axis, v_axis, base_u_min, base_v_min, v_step, u_step,
                    half_width, params.min_points, params.ruler_length_m,
                    params.verticality_limit_mm,
                    params.sor_enabled, params.sor_k, params.sor_sigma,
                    params.sor_w_weight, params.sor_method,
                    params.select_band_m, params.hole_band_m,
                    normal,
                ))
            strip_results = _pool_map(
                min(int(params.n_jobs), os.cpu_count() or 1),
                _verticality_strip_worker_mmap, args_list)
        finally:
            shared.cleanup()
    elif n_workers > 1:
        print(f'[PCFD] verticality.parallel_thread workers={n_workers} '
              f'strips={len(u_centers)}', flush=True)
        args_list = []
        for uj, u_c in enumerate(u_centers):
            args_list.append((
                uj, float(u_c),
                surf_u, surf_v, surf_ids, surf_w, surf_keep,
                v_centers, v_lo_bounds, v_hi_bounds,
                horizontal_axis, vertical_axis, origin,
                u_axis, v_axis, base_u_min, base_v_min, v_step, u_step,
                half_width, params.min_points, params.ruler_length_m,
                params.verticality_limit_mm,
                params.sor_enabled, params.sor_k, params.sor_sigma,
                params.sor_w_weight, params.sor_method,
                params.select_band_m, params.hole_band_m,
                normal,
            ))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_verticality_strip_worker, args)
                       for args in args_list]
            strip_results = [f.result() for f in futures]
    else:
        print(f'[PCFD] verticality.sequential strips={len(u_centers)}', flush=True)
        args_list = []
        for uj, u_c in enumerate(u_centers):
            args_list.append((
                uj, float(u_c),
                surf_u, surf_v, surf_ids, surf_w, surf_keep,
                v_centers, v_lo_bounds, v_hi_bounds,
                horizontal_axis, vertical_axis, origin,
                u_axis, v_axis, base_u_min, base_v_min, v_step, u_step,
                half_width, params.min_points, params.ruler_length_m,
                params.verticality_limit_mm,
                params.sor_enabled, params.sor_k, params.sor_sigma,
                params.sor_w_weight, params.sor_method,
                params.select_band_m, params.hole_band_m,
                normal,
            ))
        strip_results = [_verticality_strip_worker(args) for args in args_list]

    strip_results.sort(key=lambda x: x[0])

    # ---- Step 6: 合并 ----
    rows = []
    total_valid = 0
    for uj, strip_rows in strip_results:
        if strip_rows:
            rows.extend(strip_rows)
            total_valid += len(strip_rows)

    finite_rows = [r for r in rows if np.isfinite(r['verticality_deviation_mm'])]
    pass_rows = [r['verticality_pass'] for r in finite_rows]
    deviation_values = [r['verticality_deviation_mm'] for r in finite_rows]
    average_deviation = (float(np.mean(deviation_values))
                         if deviation_values else np.nan)

    elapsed = time.perf_counter() - started
    print(f'[PCFD] verticality.summary '
          f'windows_total={len(v_centers) * len(u_centers)} '
          f'valid_windows={len(rows)} '
          f'rows={len(finite_rows)} '
          f'max_deviation_mm='
          f'{max((r["verticality_deviation_mm"] for r in finite_rows), default=np.nan):.3f} '
          f'avg_deviation_mm={average_deviation:.3f} '
          f'max_angle_deg='
          f'{max((r["verticality_angle_deg"] for r in finite_rows), default=np.nan):.3f} '
          f'pass_rate={float(np.mean(pass_rows)) if pass_rows else 0.0:.3f} '
          f'seconds={elapsed:.2f}', flush=True)

    return {
        'ok': bool(rows),
        'reason': '' if rows else 'no_verticality_rows',
        'verticality_deviation_mm': max(
            (r['verticality_deviation_mm'] for r in finite_rows), default=np.nan),
        'verticality_avg_deviation_mm': average_deviation,
        'verticality_max_angle_deg': max(
            (r['verticality_angle_deg'] for r in finite_rows), default=np.nan),
        'verticality_pass': bool(pass_rows and all(pass_rows)),
        'verticality_pass_rate': float(np.mean(pass_rows)) if pass_rows else 0.0,
        'rows': rows,          # ← 完整行，供统计与热力图统一使用
    }


def _direction_worker_mmap(args):
    """进程模式入口：与 _direction_worker 数值路径完全一致，
    仅改为按路径 mmap 自取共享数组（避免逐任务 pickle 复制）。"""
    (points_path, raw_ids_path, precomputed_w_path, keep_path,
     plane_model, origin, u_axis, v_axis, angle, params_dict) = args
    points = np.load(points_path, mmap_mode='r')
    raw_ids = np.load(raw_ids_path, mmap_mode='r')
    precomputed_w = (np.load(precomputed_w_path, mmap_mode='r')
                     if precomputed_w_path else None)
    keep = np.load(keep_path, mmap_mode='r') if keep_path else None
    return _direction_worker((points, raw_ids, plane_model, origin,
                              u_axis, v_axis, angle, params_dict,
                              precomputed_w, keep))


def _verticality_strip_worker_mmap(args):
    """进程模式入口：与 _verticality_strip_worker 数值路径完全一致，
    仅改为按路径 mmap 自取 surf 五件套 + normal。"""
    (uj, u_c, surf_paths, *rest) = args
    surf_u = np.load(surf_paths[0], mmap_mode='r')
    surf_v = np.load(surf_paths[1], mmap_mode='r')
    surf_ids = np.load(surf_paths[2], mmap_mode='r')
    surf_keep = np.load(surf_paths[3], mmap_mode='r')
    surf_w = np.load(surf_paths[4], mmap_mode='r')
    normal = rest[-1]
    rest_without_normal = rest[:-1]
    return _verticality_strip_worker(
        (uj, u_c, surf_u, surf_v, surf_ids, surf_w, surf_keep,
         *rest_without_normal, normal))


def _direction_worker(args):
    if len(args) == 10:
        (points, raw_ids, plane_model, origin, u_axis, v_axis, angle,
         params_dict, precomputed_w, global_keep) = args
    elif len(args) == 9:
        (points, raw_ids, plane_model, origin, u_axis, v_axis, angle,
         params_dict, precomputed_w) = args
        global_keep = None
    else:
        points, raw_ids, plane_model, origin, u_axis, v_axis, angle, params_dict = args
        precomputed_w = None
        global_keep = None
    params = RulerQualityParameters(**params_dict)
    started = time.perf_counter()

    along, across, w, d3, q3 = _project_direction(
        points, plane_model, origin, u_axis, v_axis, angle)

    # ★ 使用共享局部基准面深度替代全局平面深度
    if precomputed_w is not None:
        w = precomputed_w / 1000.0

    # ★ 全局 SOR 掩码随投影重排（投影为逐点变换，顺序保持）
    keep_sorted = (np.asarray(global_keep, dtype=bool)
                   if global_keep is not None else None)

    along_range = float(along.max() - along.min())
    across_range = float(across.max() - across.min())
    projection_time = time.perf_counter() - started

    if along_range < params.ruler_length_m:
        print(f'[PCFD] quality.direction_skip angle={angle:g} '
              f'along_range={along_range:.3f}m < ruler_length={params.ruler_length_m}m', flush=True)
        return float(angle), [], projection_time, 0.0, 0.0, np.empty((0, 2), float), along_range, across_range

    strips, centres_grid, strip_idx, across_sorted, along_sorted, \
        w_sorted, ids_sorted, scan_clipped, keep_sorted = \
        _build_strips_directional(along, across, w, raw_ids, angle, params,
                                  keep=keep_sorted)
    n_windows = len(centres_grid)

    if n_windows == 0:
        return float(angle), [], projection_time, 0.0, 0.0, \
            np.empty((0, 2), float), along_range, across_range

    window_results = [None] * n_windows
    for strip_idx_num, strip in enumerate(strips):
        window_indices = np.flatnonzero(strip_idx == strip['center_index'])
        if len(window_indices) == 0:
            continue

        surface = _prepare_strip_surface_directional(
            along_sorted=along_sorted, across_sorted=across_sorted,
            w_sorted=w_sorted, raw_ids_sorted=ids_sorted, strip=strip,
            d3=d3, q3=q3, plane_model=plane_model, origin=origin, params=params,
            keep_sorted=keep_sorted)

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


def compute_ruler_quality(points, raw_ids, plane_model, origin, u_axis, v_axis,
                          params=None, baseline_blocks=None):
    """靠尺法主入口：米字形平整度 + I 形靠尺垂直度。
    """
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

    # ==============================================================
    # 共享局部基准面：沿 v 方向不重叠 block
    # ==============================================================
    blocks = baseline_blocks
    if blocks is None:
        blocks = fit_interval_baseline_planes(
            points,
            reference_plane=plane_model,
            interval_size_m=params.interval_size_m,
            origin=origin,
            u_axis=u_axis,
            v_axis=v_axis,
        )
    precomputed_w_mm = compute_block_signed_distances_mm(
        points, blocks, origin, v_axis)

    # ★ 全局一次性 SOR（四方向 + 垂直度共用）
    global_keep = _global_sor_keep(u_all, v_all, precomputed_w_mm, params)

    print(f'[PCFD] quality.baseline_planes '
          f'blocks={len(blocks)} interval={params.interval_size_m}m '
          f'sor_kept={int(np.count_nonzero(global_keep))}/{len(global_keep)}', flush=True)

    # ==============================================================
    # 平整度（星形米字）—— 四方向并行
    # ==============================================================
    n_workers = min(len(params.flatness_angles_deg), params.n_jobs,
                    os.cpu_count() or 1)

    if params.parallel_mode == 'process' and n_workers > 1:
        print(f'[PCFD] quality.parallel_process workers={n_workers} '
              f'points={len(points)}', flush=True)
        shared = _SharedArrayFiles()
        try:
            points_path = shared.dump('points', points)
            raw_ids_path = shared.dump('raw_ids', raw_ids)
            precomputed_w_path = shared.dump('precomputed_w', precomputed_w_mm)
            keep_path = shared.dump('global_keep', global_keep)
            args_list = []
            for angle in params.flatness_angles_deg:
                args_list.append((
                    points_path, raw_ids_path, precomputed_w_path, keep_path,
                    plane_model, origin, u_axis, v_axis,
                    angle, params.snapshot()
                ))
            results = _pool_map(
                min(int(params.n_jobs), os.cpu_count() or 1),
                _direction_worker_mmap, args_list)
        finally:
            shared.cleanup()
    else:
        args_list = []
        for angle in params.flatness_angles_deg:
            args_list.append((
                points, raw_ids, plane_model, origin, u_axis, v_axis,
                angle, params.snapshot(), precomputed_w_mm, global_keep
            ))
        print(f'[PCFD] quality.parallel_thread workers={n_workers} '
              f'points={len(points)}', flush=True)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_direction_worker, args) for args in args_list]
            results = [f.result() for f in futures]

    direction_results = {}
    direction_ranges = {}
    total_projection_time = 0.0
    total_prep_time = 0.0

    for direction_result in results:
        (angle, window_results, proj_time, prep_time, total_time,
         centres_grid, along_range, across_range) = _unpack_direction_result(
            direction_result)
        direction_results[float(angle)] = window_results
        direction_ranges[float(angle)] = (along_range, across_range)
        total_projection_time += proj_time
        total_prep_time += prep_time

    rows = _aggregate_star_rows(
        direction_results, origin, u_axis, v_axis,
        u_min_full, v_min_full, params)

    valid_rows = [r for r in rows if r.get('coverage_valid', False)]
    finite_rows = [r for r in valid_rows
                   if np.isfinite(r['flatness_gap_mm'])]
    raw_rows = [r for r in rows
                if np.isfinite(r['flatness_raw_max_gap_mm'])]

    n_candidates = sum(len(v) for v in direction_results.values())
    n_geometry = sum(1 for v in direction_results.values()
                     for r in v if r.get('ok'))
    n_quality = sum(1 for v in direction_results.values()
                    for r in v if r.get('ok') and r.get('coverage_valid'))
    n_failed = n_candidates - n_geometry

    print(f'[PCFD] quality.windows_summary '
          f'candidates={n_candidates} '
          f'geometry_valid={n_geometry} '
          f'quality_valid={n_quality} '
          f'failed={n_failed} '
          f'aggregated={len(rows)}', flush=True)

    # ---- 平整度区间统计 ----
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
        interval_ids = np.minimum(
            np.searchsorted(edges, center_v, side='right') - 1,
            n_intervals - 1)
        for i in range(n_intervals):
            interval_rows = [r for r, iid in zip(rows, interval_ids)
                             if iid == i]
            v_lo, v_hi = float(edges[i]), float(edges[i + 1])
            point_v = v_all
            point_mask = (point_v >= v_lo) & (
                (point_v < v_hi) if i < n_intervals - 1
                else (point_v <= v_hi)
            )
            world_z = points[point_mask, 2] if np.any(point_mask) \
                else np.empty(0)
            valid_interval_rows = [r for r in interval_rows
                                   if r.get('coverage_valid', False)]
            gaps = [r['flatness_gap_mm'] for r in valid_interval_rows
                    if np.isfinite(r['flatness_gap_mm'])]
            raw_gaps = [r['flatness_raw_max_gap_mm'] for r in interval_rows
                        if np.isfinite(r['flatness_raw_max_gap_mm'])]
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
                'flatness_avg_gap_mm': float(np.mean(gaps)) if gaps else np.nan,
                'flatness_raw_max_gap_mm': float(np.nanmax(raw_gaps)) if raw_gaps else 0.0,
                'flatness_pass_rate': float(np.mean(passes)) if passes else 0.0,
                'quality_pass_rate': float(np.mean(passes)) if passes else 0.0,
                'status': 'ok' if gaps else 'no_valid_window',
            })
        intervals.sort(key=lambda x: x['v_min_m'])

    # ==============================================================
    # 靠尺法垂直度（I 形靠尺；不使用块基准面，全局 SOR 复用）
    # ==============================================================
    verticality = _compute_verticality(
        points, raw_ids, plane_model, u_axis, v_axis, origin, params,
        u_min_full=u_min_full, v_min_full=v_min_full,
        precomputed_w_mm=precomputed_w_mm, global_keep=global_keep)

    # 将垂直度数据按 grid_key 合并到 windows 行（接口契约要求）
    verticality_by_key = {}
    for vrow in verticality.get('rows', []):
        key = vrow.get('grid_key')
        if key is not None:
            verticality_by_key[key] = vrow
    for row in rows:
        key = row.get('grid_key')
        vrow = verticality_by_key.get(key)
        if vrow is not None:
            row['verticality_deviation_mm'] = vrow['verticality_deviation_mm']
            row['verticality_angle_deg'] = vrow['verticality_angle_deg']
            row['verticality_pass'] = vrow['verticality_pass']
        else:
            row['verticality_deviation_mm'] = np.nan
            row['verticality_angle_deg'] = np.nan
            row['verticality_pass'] = False

    if intervals:
        for interval in intervals:
            vrows = [
                row for row in verticality.get('rows', [])
                if interval['v_min_m']
                <= float(row.get('center_uv_base', (np.nan, np.nan))[1])
                <= interval['v_max_m']
                and np.isfinite(row.get('verticality_deviation_mm', np.nan))
            ]
            passed = [bool(row.get('verticality_pass', False))
                      for row in vrows]
            interval['verticality_max_deviation_mm'] = (
                float(np.nanmax([row['verticality_deviation_mm']
                                 for row in vrows])) if vrows else 0.0)
            interval['verticality_avg_deviation_mm'] = (
                float(np.mean([row['verticality_deviation_mm'] for row in vrows]))
                if vrows else np.nan)
            interval['verticality_pass_rate'] = (
                float(np.mean(passed)) if passed else np.nan)

    total_elapsed = time.perf_counter() - started

    print(f'[PCFD] quality.scan_done '
          f'candidates={n_candidates} '
          f'geometry_valid={n_geometry} '
          f'quality_valid={n_quality} '
          f'intervals={len(intervals)} '
          f'projection={total_projection_time:.2f}s '
          f'prep={total_prep_time:.2f}s '
          f'total={total_elapsed:.2f}s', flush=True)

    overall = {
        'point_count': len(points),
        'candidate_window_count': n_candidates,
        'geometry_valid_window_count': n_geometry,
        'quality_valid_window_count': n_quality,
        'failed_window_count': n_failed,
        'flatness_max_gap_mm': max(
            (r['flatness_gap_mm'] for r in finite_rows), default=0.),
        'flatness_avg_gap_mm': (float(np.mean(
            [r['flatness_gap_mm'] for r in finite_rows]))
            if finite_rows else np.nan),
        'flatness_raw_max_gap_mm': max(
            (r['flatness_raw_max_gap_mm'] for r in raw_rows), default=0.),
        'flatness_pass_rate': (float(np.mean(
            [r['flatness_pass'] for r in valid_rows]))
            if valid_rows else 0.),
        'quality_pass_rate': (float(np.mean(
            [r['flatness_pass'] for r in valid_rows]))
            if valid_rows else 0.),
        'flatness_valid_rate': (float(np.mean(
            [r['coverage_valid'] for r in rows])) if rows else 0.),
        'max_hole_ratio': max(
            (r['hole_ratio'] for r in rows), default=0.),
        'verticality_deviation_mm': verticality.get(
            'verticality_deviation_mm', np.nan),
        'verticality_avg_deviation_mm': verticality.get(
            'verticality_avg_deviation_mm', np.nan),
        'verticality_max_deviation_mm': verticality.get(
            'verticality_deviation_mm', np.nan),
        'verticality_max_angle_deg': verticality.get(
            'verticality_max_angle_deg', np.nan),
        'verticality_pass': verticality.get('verticality_pass', False),
        'verticality_pass_rate': verticality.get(
            'verticality_pass_rate', 0.0),
        'plane_model': plane_model.tolist(),
        'normal': plane_model[:3].tolist(),
        'center': origin.tolist(),
    }

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
        'overall': overall,
        'verticality': {
            'ok': verticality.get('ok', False),
            'verticality_pass': verticality.get('verticality_pass', False),
            'verticality_pass_rate': verticality.get(
                'verticality_pass_rate', 0.0),
            'verticality_deviation_mm': verticality.get(
                'verticality_deviation_mm', np.nan),
            'verticality_avg_deviation_mm': verticality.get(
                'verticality_avg_deviation_mm', np.nan),
            'verticality_max_angle_deg': verticality.get(
                'verticality_max_angle_deg', np.nan),
            'rows': verticality.get('rows', []),   # 完整垂直度行
        },
        'performance': {
            'projection_seconds': total_projection_time,
            'prep_seconds': total_prep_time,
            'total_seconds': total_elapsed,
        },
    }