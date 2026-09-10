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

_EMPTY_SOURCE_IDS = np.empty(0, dtype=np.int64)


# =============================================================================
# 常驻进程池与共享数组落盘
#
# Windows spawn 下每个子进程都要重新启动解释器 + import，一次性池的固定
# 开销在每次质量计算重复支付；模块级懒加载常驻池只在首次支付，atexit
# 注册关闭（解释器退出在 PySide6 主窗口清理之后，shutdown(wait=False)
# 不触碰 Qt 对象，退出顺序安全）。
#
# 进程模式任务不再 pickle 携带完整 points/surf 大数组：父进程把共享数组
# 落一份临时 .npy，任务只传 (路径, 索引/标量, 参数)，子进程
# np.load(mmap_mode='r') 自取，页由 OS 在子进程间共享。数值路径不变。
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
    scan_step_m: float = 2.0
    strip_step_m: float = .055
    flatness_angles_deg: tuple = (0., 45., 90., 135.)
    select_band_m: float = .01
    hole_band_m: float = .02
    bin_size_m: float = .04
    top_q: float = 1.
    flatness_limit_mm: float = 8.
    verticality_limit_mm: float = 4. 
    verticality_bin_size_m: float = 2.0
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
        # 立面范围小于尺长，生成单个居中窗口保证边界覆盖，标记截断
        return np.array([(lo + hi) / 2.0]), np.array([True])
    # 标准滑动：首窗口中心 = lo + half，末窗口中心 = hi - half
    # 步长 = step，末窗口强制覆盖边界
    first = lo + half
    last = hi - half
    centers = np.arange(first, last + 1e-9, step, dtype=float)
    # 强制最后一个窗口贴边覆盖，避免 arange 的浮点误差漏掉边界。
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


def _build_strips_directional(along, across, w, raw_ids, angle, params):
    """Build strips and window centers for a given direction."""
    across_order = np.argsort(across, kind='stable')
    across_sorted = across[across_order]
    along_sorted = along[across_order]
    w_sorted = w[across_order]
    ids_sorted = raw_ids[across_order]

    q_min, q_max = across_sorted.min(), across_sorted.max()
    if q_max - q_min < params.ruler_width_m:
        return [], np.empty((0, 2), float), np.empty(0, np.int64), across_sorted, along_sorted, w_sorted, ids_sorted, np.empty(0, dtype=bool)

    half_width = params.ruler_width_m / 2.0
    strip_step = max(float(params.strip_step_m), 1e-6)
    strip_centers = _full_window_centers(q_min, q_max, strip_step,
                                         params.ruler_width_m)
    if len(strip_centers) == 0:
        return [], np.empty((0, 2), float), np.empty(0, np.int64), across_sorted, along_sorted, w_sorted, ids_sorted, np.empty(0, dtype=bool)

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
        return [], np.empty((0, 2), float), np.empty(0, np.int64), across_sorted, along_sorted, w_sorted, ids_sorted, np.empty(0, dtype=bool)

    # Generate scan grid independently for each strip
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
    return strips, centres_grid, strip_idx, across_sorted, along_sorted, w_sorted, ids_sorted, scan_clipped_arr


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

def _snap_window_to_base_grid(center_xyz, origin, u_axis, v_axis, u0, v0,
                              u_step=None, v_step=None):
    """把窗口中心吸附到 (u, v) 双步长网格，返回网格 key 与吸附后坐标。

    u/v 两个方向的网格间距可以不同：u 方向为横向 strip_step（默认 0.055m），
    v 方向为竖直 scan_step（默认 2.0m）。若只传一个步长，则两个方向共用。
    """
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
    """聚合四方向结果 into shared 2m x 2m star windows.

    A star centre is a physical location, not a rounded centre from four
    independently generated scans.  Its lattice is transverse ruler width by
    longitudinal half ruler length, as required by the inspection standard.
    """
    # 收集所有有效窗口
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
            })

    if not all_windows:
        return []

    # 转为数组
    centers = np.array([w['center_xyz'] for w in all_windows])
    rel = centers - origin
    base_u = rel @ u_axis
    base_v = rel @ v_axis

    # 使用两个步长划分网格
    step_u = max(float(params.ruler_width_m), 1e-6)
    step_v = max(float(params.ruler_length_m) / 2.0, 1e-6)

    # 计算网格索引（四舍五入）
    ku = np.round((base_u - u_min_full) / step_u).astype(np.int64)
    kv = np.round((base_v - v_min_full) / step_v).astype(np.int64)

    # 分组
    grid_groups = {}
    for i, (u, v) in enumerate(zip(ku, kv)):
        key = (int(u), int(v))
        grid_groups.setdefault(key, []).append(all_windows[i])

    rows = []
    for key, members in grid_groups.items():
        # 按方向分组
        dir_groups = {}
        for m in members:
            d = m['direction_deg']
            dir_groups.setdefault(d, []).append(m)

        # 对每个方向取最大 gap（或可改为取最大值）
        directional = []
        for angle in params.flatness_angles_deg:
            angle_f = float(angle)
            if angle_f not in dir_groups:
                continue
            # 取最大 gap（绝对值最大）
            best = max(dir_groups[angle_f], 
                      key=lambda x: x['gap_mm'] if np.isfinite(x['gap_mm']) else -np.inf)
            directional.append(best)

        if not directional:
            continue

        # 计算网格中心（用于显示）
        snap_u = u_min_full + key[0] * step_u
        snap_v = v_min_full + key[1] * step_v
        snap_xyz = origin + snap_u * u_axis + snap_v * v_axis

        # 取所有方向的最大绝对偏差（平整度值）
        valid_gaps = [d['gap_mm'] for d in directional if d['coverage_valid'] and np.isfinite(d['gap_mm'])]
        all_gaps = [d['gap_mm'] for d in directional if np.isfinite(d['gap_mm'])]

        # 若没有有效方向，则使用所有方向的最大值（但可能不合格）
        if valid_gaps:
            flatness_gap = max(valid_gaps)
        else:
            flatness_gap = max(all_gaps) if all_gaps else np.nan

        # 通过条件：至少有一个有效方向且 flatness_gap <= limit
        quality_valid = bool(valid_gaps)
        flatness_pass = bool(quality_valid and np.isfinite(flatness_gap)
                             and flatness_gap <= params.flatness_limit_mm)

        # 选取主导方向（具有最大 gap 的有效方向）
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
        # This is intentionally opt-in: the normal algorithm contract remains
        # compact, while report export can request the original point domain
        # needed for point-level heatmap painting.
        if params.retain_covered_source_ids:
            row['covered_source_ids'] = np.unique(np.concatenate([
                d['covered_source_ids'] for d in directional
                if len(d['covered_source_ids'])
            ])).astype(np.int64).tolist() if any(
                len(d['covered_source_ids']) for d in directional) else []
        rows.append(row)

    # 排序：按 v 升序，u 升序
    rows.sort(key=lambda item: (item['center_uv_base'][1], item['center_uv_base'][0]))
    for idx, row in enumerate(rows):
        row['window_id'] = idx

    return rows



# =============================================================================
# 垂直度并行化 Worker（模块顶层，确保 Windows spawn 可 pickle）
# =============================================================================
def _verticality_strip_worker(args):
    """处理单个横向 strip 的所有竖直窗口。
    
    数据清洗流程与平整度对齐：
        SOR 去噪 → select_band 表面筛选 → hole_band 孔洞检测 → 鲁棒拟合
    返回 (uj, rows_list)。
    """
    (uj, u_c,
     surf_u, surf_v, surf_ids,
     v_centers, v_lo_bounds, v_hi_bounds,
     horizontal_axis, vertical_axis, origin,
     u_axis, v_axis, base_u_min, base_v_min, v_step, u_step,
     half_width, min_points, ruler_length_m, verticality_limit_mm,
     sor_enabled, sor_k, sor_sigma, sor_w_weight, sor_method,
     select_band, hole_band) = args

    # ---- 在全局有序的 surf_u 上快速定位 strip ----
    lo_idx = np.searchsorted(surf_u, u_c - half_width, side='left')
    hi_idx = np.searchsorted(surf_u, u_c + half_width, side='right')

    if hi_idx - lo_idx < min_points:
        return uj, []

    # 提取 strip 数据
    strip_u = surf_u[lo_idx:hi_idx]
    strip_v = surf_v[lo_idx:hi_idx]
    strip_ids_local = surf_ids[lo_idx:hi_idx]

    # Strip 内按 V (vertical) 排序，用于沿竖直方向滑动窗口
    v_order = np.argsort(strip_v, kind='quicksort')
    strip_v = strip_v[v_order]
    strip_u = strip_u[v_order]
    strip_ids_local = strip_ids_local[v_order]

    # 向量化计算所有 v 窗口边界
    v_win_lo = np.searchsorted(strip_v, v_lo_bounds, side='left')
    v_win_hi = np.searchsorted(strip_v, v_hi_bounds, side='right')
    n_win_points = v_win_hi - v_win_lo
    
    valid_mask = n_win_points >= min_points
    if not np.any(valid_mask):
        return uj, []

    # ---- 逐窗口独立拟合（带统一数据清洗）----
    rows = []
    for vi in np.flatnonzero(valid_mask):
        wlo = v_win_lo[vi]
        whi = v_win_hi[vi]
        n_win = whi - wlo

        win_u = strip_u[wlo:whi]  # H (horizontal)
        win_v = strip_v[wlo:whi]  # V (vertical)
        win_ids = strip_ids_local[wlo:whi]

        # ================================================================
        #  统一数据清洗层
        # ================================================================
        
        # 1) SOR 去噪
        keep = np.ones(n_win, dtype=bool)
        if sor_enabled and n_win > 16:
            if sor_method == 'local':
                keep = sor_mask_local(win_v, win_u, sor_k, sor_sigma)
            elif sor_method == 'grid':
                keep = sor_mask_grid(win_v, win_u, sor_k, sor_sigma, w_weight=sor_w_weight)
            elif sor_method == 'exact':
                keep = sor_mask_exact(win_v, win_u, sor_k, sor_sigma, w_weight=sor_w_weight)
            else:
                keep = sor_mask_local(win_v, win_u, sor_k, sor_sigma)
            
            if np.sum(keep) < 3:
                keep[:] = True
        # 2) 表面点筛选（select_band 等价）
        try:
            a_ref, b_ref = fit_line(win_v[keep], win_u[keep])
        except (np.linalg.LinAlgError, ValueError):
            continue
        
        residual = win_u - (a_ref * win_v + b_ref)
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

        # Window-Level 鲁棒拟合
        try:
            a, b = fit_line(win_v[final_mask], win_u[final_mask])
        except (np.linalg.LinAlgError, ValueError):
            continue

        deviation_mm = abs(a) * ruler_length_m * 1000.0
        angle_deg = float(np.degrees(np.arctan(abs(a))))
        verticality_pass = np.isfinite(deviation_mm) and deviation_mm <= verticality_limit_mm

        # 孔洞覆盖统计（2m 靠尺范围内）
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

        center_xyz = origin + u_c * horizontal_axis + v_center * vertical_axis

        key, snap_u, snap_v, snap_xyz, snap_distance = _snap_window_to_base_grid(
            center_xyz.tolist(), origin, u_axis, v_axis, base_u_min, base_v_min,
            u_step, v_step)

        # 偏离趋势线最大的点作为 depression_source_id
        trend_vals = a * win_v + b
        dep_idx = int(np.argmax(np.abs(win_u - trend_vals)))

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
            'pivot_source_ids': [int(win_ids[final_mask][0]), int(win_ids[final_mask][-1])],
            'depression_source_id': int(win_ids[dep_idx]),
            'trend_slope': float(a),
            'trend_intercept': float(b),
            'n_sor_removed': int(np.sum(~keep)),
            'n_used': int(np.sum(final_mask)),
        })

    return uj, rows

def _compute_verticality(points, raw_ids, plane_model, u_axis, v_axis, origin, params, u_min_full=None, v_min_full=None):
    """
    垂直度 = 墙面整体相对于铅垂线的倾斜程度
    
    建模逻辑：
        1. 统一数据清洗（SOR → select_band → hole_band）与平整度对齐
        2. 并行策略与平整度一致（parallel_mode + n_jobs）
        3. 最终指标：|a| * 2000mm（2m 高度处的水平偏移）
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

    # ===================================================================
    # Step 1: 确定真实竖直方向
    # ===================================================================
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

    # ===================================================================
    # Step 2: 投影到 (H, V, N) 坐标系
    # ===================================================================
    rel = points - origin
    u_h = rel @ horizontal_axis   # 水平横向偏移 (H)
    v_v = rel @ vertical_axis     # 真实竖直坐标 (V)
    w_n = rel @ normal            # 距立面平面的距离 (N)

    # ===================================================================
    # Step 3: 表面点过滤
    # ===================================================================
    surface_band = float(getattr(params, 'select_band_m', 0.02))
    surf_mask = np.abs(w_n) <= surface_band
    n_surf = int(np.count_nonzero(surf_mask))

    if n_surf < params.min_points:
        return empty(f'too_few_surface_points:{n_surf}')

    surf_u = u_h[surf_mask]  # H (horizontal lateral)
    surf_v = v_v[surf_mask]  # V (true vertical)
    surf_ids = raw_ids[surf_mask]

    v_min = float(surf_v.min())
    v_max = float(surf_v.max())
    u_min = float(surf_u.min())
    u_max = float(surf_u.max())

    v_span = v_max - v_min
    u_span = u_max - u_min

    if v_span < params.ruler_length_m:
        return empty('verticality_domain_too_short')

    half = params.ruler_length_m / 2.0             # 1.0m
    half_width = params.ruler_width_m / 2.0          # 0.0275m

    v_step = max(float(params.scan_step_m), 1e-6)   # 5cm vertical step
    u_step = max(float(params.strip_step_m), 1e-6)   # 5cm lateral step

    # Window centers
    v_centers = _full_window_centers(v_min, v_max, v_step, params.ruler_length_m)
    u_centers = _full_window_centers(u_min, u_max, u_step, params.ruler_width_m)

    if len(v_centers) == 0 or len(u_centers) == 0:
        return empty('no_verticality_grid')

    print(f'[PCFD] verticality.grid '
          f'v_centers={len(v_centers)} u_centers={len(u_centers)} '
          f'total_windows_est={len(v_centers)*len(u_centers)}', flush=True)

    # ===================================================================
    # Step 4: 全局按 H (horizontal) 排序 —— searchsorted 要求有序数组
    # ===================================================================
    u_order = np.argsort(surf_u, kind='stable')
    surf_u = surf_u[u_order]
    surf_v = surf_v[u_order]
    surf_ids = surf_ids[u_order]

    # 统一 grid_key 基准，与平整度一致
    base_u_min = float(u_min_full) if u_min_full is not None else float((points @ u_axis).min())
    base_v_min = float(v_min_full) if v_min_full is not None else float((points @ v_axis).min())

    # 预计算所有 v 窗口边界
    v_lo_bounds = v_centers - half
    v_hi_bounds = v_centers + half

    # ===================================================================
    # Step 5: 并行逐横向 strip 处理
    # ===================================================================
    n_workers = min(len(u_centers), params.n_jobs, os.cpu_count() or 1)

    if n_workers > 1 and params.parallel_mode == 'process':
        print(f'[PCFD] verticality.parallel_process workers={n_workers} '
              f'strips={len(u_centers)}', flush=True)
        shared = _SharedArrayFiles()
        try:
            surf_paths = (shared.dump('surf_u', surf_u),
                          shared.dump('surf_v', surf_v),
                          shared.dump('surf_ids', surf_ids))
            args_list = []
            for uj, u_c in enumerate(u_centers):
                args_list.append((
                    uj, float(u_c), surf_paths,
                    v_centers, v_lo_bounds, v_hi_bounds,
                    horizontal_axis, vertical_axis, origin,
                    u_axis, v_axis, base_u_min, base_v_min, v_step, u_step,
                    half_width, params.min_points, params.ruler_length_m, params.verticality_limit_mm,
                    params.sor_enabled, params.sor_k, params.sor_sigma, params.sor_w_weight, params.sor_method,
                    params.select_band_m, params.hole_band_m,
                ))
            strip_results = _pool_map(
                min(int(params.n_jobs), os.cpu_count() or 1),
                _verticality_strip_worker_mmap, args_list)
        finally:
            shared.cleanup()
    elif n_workers > 1:
        args_list = []
        for uj, u_c in enumerate(u_centers):
            args_list.append((
                uj, float(u_c),
                surf_u, surf_v, surf_ids,
                v_centers, v_lo_bounds, v_hi_bounds,
                horizontal_axis, vertical_axis, origin,
                u_axis, v_axis, base_u_min, base_v_min, v_step, u_step,
                half_width, params.min_points, params.ruler_length_m, params.verticality_limit_mm,
                params.sor_enabled, params.sor_k, params.sor_sigma, params.sor_w_weight, params.sor_method,
                params.select_band_m, params.hole_band_m,
            ))
        print(f'[PCFD] verticality.parallel_thread workers={n_workers} '
              f'strips={len(u_centers)}', flush=True)
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_verticality_strip_worker, args) for args in args_list]
            strip_results = [f.result() for f in futures]
    else:
        args_list = []
        for uj, u_c in enumerate(u_centers):
            args_list.append((
                uj, float(u_c),
                surf_u, surf_v, surf_ids,
                v_centers, v_lo_bounds, v_hi_bounds,
                horizontal_axis, vertical_axis, origin,
                u_axis, v_axis, base_u_min, base_v_min, v_step, u_step,
                half_width, params.min_points, params.ruler_length_m, params.verticality_limit_mm,
                params.sor_enabled, params.sor_k, params.sor_sigma, params.sor_w_weight, params.sor_method,
                params.select_band_m, params.hole_band_m,
            ))
        print(f'[PCFD] verticality.sequential strips={len(u_centers)}', flush=True)
        strip_results = [_verticality_strip_worker(args) for args in args_list]

    # 按 uj 排序保证确定性输出顺序
    strip_results.sort(key=lambda x: x[0])

    # 合并结果
    rows = []
    total_valid = 0
    processed_strips = 0

    for uj, strip_rows in strip_results:
        if strip_rows:
            processed_strips += 1
            rows.extend(strip_rows)
            total_valid += len(strip_rows)

            if processed_strips % 50 == 0:
                elapsed = time.perf_counter() - started
                print(f'[PCFD] verticality.progress '
                      f'strips={processed_strips}/{len(u_centers)} '
                      f'valid_windows={total_valid} '
                      f'seconds={elapsed:.1f}', flush=True)

    # 最终进度日志
    if processed_strips > 0:
        elapsed = time.perf_counter() - started
        print(f'[PCFD] verticality.progress '
              f'strips={processed_strips}/{len(u_centers)} '
              f'valid_windows={total_valid} '
              f'seconds={elapsed:.1f}', flush=True)

    # ===================================================================
    # Step 6: 汇总统计
    # ===================================================================
    finite_rows = [r for r in rows if np.isfinite(r['verticality_deviation_mm'])]
    pass_rows = [r['verticality_pass'] for r in finite_rows]
    deviation_values = [r['verticality_deviation_mm'] for r in finite_rows]
    average_deviation = (float(np.mean(deviation_values))
                         if deviation_values else np.nan)

    elapsed = time.perf_counter() - started
    print(f'[PCFD] verticality.summary '
          f'windows_total={len(v_centers)*len(u_centers)} '
          f'valid_windows={len(rows)} '
          f'rows={len(finite_rows)} '
          f'max_deviation_mm={max((r["verticality_deviation_mm"] for r in finite_rows), default=np.nan):.3f} '
          f'avg_deviation_mm={average_deviation:.3f} '
          f'max_angle_deg={max((r["verticality_angle_deg"] for r in finite_rows), default=np.nan):.3f} '
          f'pass_rate={float(np.mean(pass_rows)) if pass_rows else 0.0:.3f} '
          f'seconds={elapsed:.2f}', flush=True)

    return {
        'ok': bool(rows),
        'reason': '' if rows else 'no_verticality_rows',
        'verticality_deviation_mm': max((r['verticality_deviation_mm'] for r in finite_rows), default=np.nan),
        'verticality_avg_deviation_mm': average_deviation,
        'verticality_max_angle_deg': max((r['verticality_angle_deg'] for r in finite_rows), default=np.nan),
        'verticality_pass': bool(pass_rows and all(pass_rows)),
        'verticality_pass_rate': float(np.mean(pass_rows)) if pass_rows else 0.0,
        'rows': rows,
    }

def _direction_worker_mmap(args):
    """进程模式入口：与 _direction_worker 数值路径完全一致，
    仅改为按路径 mmap 自取共享数组（避免逐任务 pickle 复制）。"""
    (points_path, raw_ids_path, plane_model, origin, u_axis, v_axis,
     angle, params_dict) = args
    points = np.load(points_path, mmap_mode='r')
    raw_ids = np.load(raw_ids_path, mmap_mode='r')
    return _direction_worker((points, raw_ids, plane_model, origin,
                              u_axis, v_axis, angle, params_dict))


def _verticality_strip_worker_mmap(args):
    """进程模式入口：与 _verticality_strip_worker 数值路径完全一致，
    仅改为按路径 mmap 自取 surf 三件套。"""
    (uj, u_c, surf_paths, *rest) = args
    surf_u = np.load(surf_paths[0], mmap_mode='r')
    surf_v = np.load(surf_paths[1], mmap_mode='r')
    surf_ids = np.load(surf_paths[2], mmap_mode='r')
    return _verticality_strip_worker(
        (uj, u_c, surf_u, surf_v, surf_ids, *rest))


def _direction_worker(args):
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
              f'along_range={along_range:.3f}m < ruler_length={params.ruler_length_m}m', flush=True)
        return float(angle), [], projection_time, 0.0, 0.0, np.empty((0, 2), float), along_range, across_range

    strips, centres_grid, strip_idx, across_sorted, along_sorted, w_sorted, ids_sorted, scan_clipped = _build_strips_directional(
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

    # Parallel computation for each direction
    n_workers = min(len(params.flatness_angles_deg), params.n_jobs, os.cpu_count() or 1)

    if params.parallel_mode == 'process' and n_workers > 1:
        print(f'[PCFD] quality.parallel_process workers={n_workers} '
              f'points={len(points)}', flush=True)
        shared = _SharedArrayFiles()
        try:
            points_path = shared.dump('points', points)
            raw_ids_path = shared.dump('raw_ids', raw_ids)
            args_list = []
            for angle in params.flatness_angles_deg:
                args_list.append((
                    points_path, raw_ids_path, plane_model, origin,
                    u_axis, v_axis, angle, params.snapshot()
                ))
            # 常驻池 worker 数按 n_jobs 取齐，与垂直度共用同一池，
            # spawn 固定开销整个会话只付一次
            results = _pool_map(min(int(params.n_jobs), os.cpu_count() or 1),
                                _direction_worker_mmap, args_list)
        finally:
            shared.cleanup()
    else:
        args_list = []
        for angle in params.flatness_angles_deg:
            args_list.append((
                points, raw_ids, plane_model, origin, u_axis, v_axis,
                angle, params.snapshot()
            ))
        print(f'[PCFD] quality.parallel_thread workers={n_workers} '
              f'points={len(points)}', flush=True)
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_direction_worker, args) for args in args_list]
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

    # 用物理坐标聚类实现真米字
    rows = _aggregate_star_rows(
        direction_results, origin, u_axis, v_axis,
        u_min_full, v_min_full, params)

    valid_rows = [r for r in rows if r.get('coverage_valid', False)]
    finite_rows = [r for r in valid_rows if np.isfinite(r['flatness_gap_mm'])]
    raw_rows = [r for r in rows if np.isfinite(r['flatness_raw_max_gap_mm'])]

    n_candidates = sum(len(v) for v in direction_results.values())
    n_geometry = sum(1 for v in direction_results.values() for r in v if r.get('ok'))
    n_quality = sum(1 for v in direction_results.values() for r in v if r.get('ok') and r.get('coverage_valid'))
    n_failed = n_candidates - n_geometry

    print(f'[PCFD] quality.windows_summary '
          f'candidates={n_candidates} '
          f'geometry_valid={n_geometry} '
          f'quality_valid={n_quality} '
          f'failed={n_failed} '
          f'aggregated={len(rows)}', flush=True)

    # Interval statistics
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
            v_lo, v_hi = float(edges[i]), float(edges[i + 1])
            # 复用顶部已算好的 v_all（同一表达式 (points-origin)@v_axis），
            # 不再每个 interval 对全量点重算
            point_v = v_all
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
                'flatness_avg_gap_mm': float(np.mean(gaps)) if gaps else np.nan,
                'flatness_raw_max_gap_mm': float(np.nanmax(raw_gaps)) if raw_gaps else 0.0,
                'flatness_pass_rate': float(np.mean(passes)) if passes else 0.0,
                'quality_pass_rate': float(np.mean(passes)) if passes else 0.0,
                'status': 'ok' if gaps else 'no_valid_window',
            })
        intervals.sort(key=lambda x: x['v_min_m'])

    verticality = _compute_verticality(points, raw_ids, plane_model, u_axis, v_axis, 
                                          origin, params, u_min_full=u_min_full, v_min_full=v_min_full)

    # Build verticality lookup by grid_key
    verticality_rows = {}
    for item in verticality.get('rows', []):
        key = item.get('grid_key')
        if key is not None:
            verticality_rows[key] = item

    # Merge verticality into flatness rows
    for row in rows:
        vrow = verticality_rows.get(tuple(row.get('grid_key', ())))
        if vrow is None:
            row['verticality_angle_deg'] = np.nan
            row['verticality_deviation_mm'] = np.nan
            row['verticality_pass'] = False
        else:
            row['verticality_angle_deg'] = float(vrow.get('verticality_angle_deg', np.nan))
            row['verticality_deviation_mm'] = float(vrow.get('verticality_deviation_mm', np.nan))
            row['verticality_pass'] = bool(vrow.get('verticality_pass', False))

    # Update interval statistics with the independent I-ruler rows.  The
    # verticality grid is not guaranteed to share the star flatness grid_key;
    # never turn an unmatched I-ruler window into a synthetic failure.
    if intervals:
        for interval in intervals:
            vrows = [
                row for row in verticality.get('rows', [])
                if interval['v_min_m'] <= float(row.get('center_uv_base', (np.nan, np.nan))[1]) <= interval['v_max_m']
                and np.isfinite(row.get('verticality_deviation_mm', np.nan))
            ]
            passed = [bool(row.get('verticality_pass', False)) for row in vrows]
            interval['verticality_max_deviation_mm'] = (
                float(np.nanmax([row['verticality_deviation_mm'] for row in vrows])) if vrows else 0.0
            )
            interval['verticality_avg_deviation_mm'] = (
                float(np.mean([row['verticality_deviation_mm'] for row in vrows]))
                if vrows else np.nan
            )
            interval['verticality_pass_rate'] = float(np.mean(passed)) if passed else np.nan

    total_elapsed = time.perf_counter() - started

    print(f'[PCFD] quality.scan_done '
          f'candidates={n_candidates} '
          f'geometry_valid={n_geometry} '
          f'quality_valid={n_quality} '
          f'intervals={len(intervals)} '
          f'projection={total_projection_time:.2f}s '
          f'prep={total_prep_time:.2f}s '
          f'total={total_elapsed:.2f}s', flush=True)

    # FIX: Ensure all required fields are present for dialog
    overall = {
        'point_count': len(points),
        'candidate_window_count': n_candidates,
        'geometry_valid_window_count': n_geometry,
        'quality_valid_window_count': n_quality,
        'failed_window_count': n_failed,
        'flatness_max_gap_mm': max((r['flatness_gap_mm'] for r in finite_rows), default=0.),
        'flatness_avg_gap_mm': float(np.mean([r['flatness_gap_mm'] for r in finite_rows])) if finite_rows else np.nan,
        'flatness_raw_max_gap_mm': max((r['flatness_raw_max_gap_mm'] for r in raw_rows), default=0.),
        'flatness_pass_rate': float(np.mean([r['flatness_pass'] for r in valid_rows])) if valid_rows else 0.,
        'quality_pass_rate': float(np.mean([r['flatness_pass'] for r in valid_rows])) if valid_rows else 0.,
        'flatness_valid_rate': float(np.mean([r['coverage_valid'] for r in rows])) if rows else 0.,
        'max_hole_ratio': max((r['hole_ratio'] for r in rows), default=0.),
        # FIX: Ensure verticality fields use correct names and have valid values
        'verticality_deviation_mm': verticality.get('verticality_deviation_mm', np.nan),
        'verticality_avg_deviation_mm': verticality.get('verticality_avg_deviation_mm', np.nan),
        'verticality_max_deviation_mm':verticality.get('verticality_deviation_mm', np.nan),
        'verticality_max_angle_deg': verticality.get('verticality_max_angle_deg', np.nan),
        'verticality_pass': verticality.get('verticality_pass', False),
        'verticality_pass_rate': verticality.get('verticality_pass_rate', 0.0),
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
            'verticality_pass_rate': verticality.get('verticality_pass_rate', 0.0),
        },
        'performance': {
            'projection_seconds': total_projection_time,
            'prep_seconds': total_prep_time,
            'total_seconds': total_elapsed,
        },
    }