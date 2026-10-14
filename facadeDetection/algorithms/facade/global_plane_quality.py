"""Robust global facade plane fitting and batched window measurements."""
from __future__ import annotations

import numpy as np

# 从 local_baseline 导入平面拟合与分块工具（已迁移至此，避免循环依赖）
from algorithms.facade.local_baseline import (
    fit_global_plane,
    fit_interval_baseline_planes,
    compute_block_signed_distances_mm,
    find_block_for_v,
)


def _unit(value):
    value = np.asarray(value, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(value)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError('zero direction vector')
    return value / norm


def _compute_hull_area(points_2d):
    """Monotone chain algorithm for 2D convex hull area (pure NumPy)."""
    pts = np.asarray(points_2d, dtype=float).reshape(-1, 2)
    if len(pts) < 3:
        return 0.0
    pts = np.unique(np.round(pts, 6), axis=0)
    if len(pts) < 3:
        return 0.0
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-12:
            lower.pop()
        lower.append(p)

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-12:
            upper.pop()
        upper.append(p)

    hull = np.vstack([lower[:-1], upper[:-1]])
    if len(hull) < 3:
        return 0.0

    x, y = hull[:, 0], hull[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _fit_line(u, w):
    """Robust line fit w = a*u + b with iterative outlier rejection."""
    u, w = np.asarray(u, float), np.asarray(w, float)
    if len(u) < 2 or np.ptp(u) < 1e-12:
        raise ValueError("profile too short for line fit")
    m = np.ones(len(u), dtype=bool)
    for _ in range(3):
        a, c = np.polyfit(u[m], w[m], 1)
        r = w - (a * u + c)
        mad = 1.4826 * np.median(np.abs(r[m] - np.median(r[m]))) + 1e-9
        new = np.abs(r) <= max(2.5 * mad, 1e-5)
        if np.sum(new) < 3 or np.array_equal(new, m):
            break
        m = new
    return float(a), float(c)


def _window_centers(lo, hi, length, step):
    """Return centres for a full-footprint sliding window, including edges."""
    lo, hi = float(lo), float(hi)
    length = max(float(length), 1e-9)
    step = max(float(step), 1e-9)
    if hi - lo <= length:
        return np.asarray([(lo + hi) * 0.5], dtype=float)
    first, last = lo + length * 0.5, hi - length * 0.5
    values = np.arange(first, last + 1e-9, step, dtype=float)
    if values.size == 0 or abs(values[-1] - last) > 1e-7:
        values = np.r_[values, last]
    return values


# =============================================================================
# 分块局部平面拟合（委托到 local_baseline，无重叠 block）
# =============================================================================

def fit_local_plane_blocks(points, reference_plane, interval_size_m,
                           origin, u_axis, v_axis, **fit_kwargs):
    """委托到 local_baseline.fit_interval_baseline_planes（无重叠 block）。

    旧接口兼容：返回字典列表，保持与 compute_block_plane_quality 的契约。
    """
    blocks = fit_interval_baseline_planes(
        points, reference_plane, interval_size_m, origin, u_axis, v_axis,
        **fit_kwargs)
    return [
        {
            'v0': b.v0,
            'v1': b.v1,
            'plane_model': b.plane_model,
            'point_indices': b.point_indices,
            'fit_accepted': True,
        }
        for b in blocks
    ]


def _find_block_for_v(v_coord, blocks):
    """委托到 local_baseline.find_block_for_v。"""
    idx = find_block_for_v(v_coord, blocks)
    return idx


# =============================================================================
# 分块局部平面质量评估
# =============================================================================

def compute_block_plane_quality(points, blocks, origin, u_axis, v_axis,
                                flatness_limit_mm=8., verticality_limit_mm=10.,
                                raw_ids=None, grid_res=0.05,
                                ruler_length_m=2.0, ruler_width_m=0.055):
    """分块局部平面质量评估：2m×5.5cm 靠尺窗口滑动，每窗口使用中心所在 block 的局部平面。

    变量命名契约（本函数内全程一致）：
      - ``pts``        : filtered_pts 副本
      - ``ids``        : raw 全局行号（= raw_ids[ix]）
      - ``u_all``/``v_all`` : pts 在 UV 框架下的投影坐标
      - ``all_dist_mm``: 每个点在自己 block 局部平面上的 signed distance (mm)
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    ids = (np.asarray(raw_ids, np.int64).reshape(-1)
           if raw_ids is not None else np.arange(len(pts)))
    u_ax = _unit(u_axis)
    v_ax = _unit(v_axis)
    origin = np.asarray(origin, dtype=float).reshape(3)

    rel = pts - origin
    u_all = rel @ u_ax
    v_all = rel @ v_ax

    u0, u1 = float(u_all.min()), float(u_all.max())
    v0, v1 = float(v_all.min()), float(v_all.max())

    # 预计算每个点在各自 block 局部平面上的 signed distance (mm)
    # 兼容两种 block 格式：旧 dict 或新 BaselineBlock dataclass
    all_dist_mm = np.full(len(pts), np.nan, dtype=np.float64)
    for block in blocks:
        # 旧 dict 格式
        if isinstance(block, dict):
            idx = block.get('point_indices', np.array([], dtype=np.int64))
            if len(idx) == 0:
                continue
            plane = np.asarray(block.get('plane_model', [0,0,1,0]), dtype=float).reshape(4)
            plane[:3] = _unit(plane[:3])
            bpts = pts[idx]
            all_dist_mm[idx] = (bpts @ plane[:3] + plane[3]) * 1000.0
        else:
            # 新 BaselineBlock dataclass
            idx = block.point_indices
            if len(idx) == 0:
                continue
            plane = block.plane_model
            bpts = pts[idx]
            all_dist_mm[idx] = (bpts @ plane[:3] + plane[3]) * 1000.0

    # ---- 靠尺窗口滑动：步长统一为半尺长/尺宽 ----
    length_m = max(float(ruler_length_m), 1e-9)
    width_m = max(float(ruler_width_m), 1e-9)
    # ★ 纵向步长 = 半尺长 (1.0m)，横向步长 = 尺宽 (0.055m)
    u_step = width_m
    v_step = length_m / 2.0
    u_centers = _window_centers(u0, u1, width_m, u_step)
    v_centers = _window_centers(v0, v1, length_m, v_step)

    windows = []
    for a, uc in enumerate(u_centers):
        for b, vc in enumerate(v_centers):
            u_lo, u_hi = uc - width_m / 2.0, uc + width_m / 2.0
            v_lo, v_hi = vc - length_m / 2.0, vc + length_m / 2.0
            ix = np.flatnonzero((u_all >= u_lo) & (u_all <= u_hi) &
                                (v_all >= v_lo) & (v_all <= v_hi))
            if len(ix) < 3:
                continue

            center_block_idx = _find_block_for_v(vc, blocks)
            if center_block_idx is None or center_block_idx < 0:
                continue

            win_dist = all_dist_mm[ix]
            if not np.any(np.isfinite(win_dist)):
                continue

            valid_mask = np.isfinite(win_dist)
            win_dist_valid = win_dist[valid_mask]
            ix_valid = ix[valid_mask]

            depression = max(0.0, float(-win_dist_valid.min()))
            protrusion = max(0.0, float(win_dist_valid.max()))
            gap = max(depression, protrusion)

            # 逐点缺陷（平整度）
            defect_mask = np.abs(win_dist_valid) > flatness_limit_mm
            if np.any(defect_mask):
                defect_rows = ix_valid[defect_mask].astype(np.int64).tolist()
                defect_point_indices = ids[ix_valid][defect_mask].tolist()
                defect_values_mm = win_dist_valid[defect_mask].tolist()
                defect_types = [
                    'depression' if v < 0 else 'protrusion'
                    for v in defect_values_mm
                ]
            else:
                defect_rows, defect_point_indices = [], []
                defect_values_mm, defect_types = [], []

            # 垂直度：LOCAL trend of deviation vs height
            v_pts = v_all[ix_valid]
            verticality_mm = np.nan
            if len(v_pts) >= 3 and np.ptp(v_pts) > 0.01:
                try:
                    slope, _ = _fit_line(v_pts, win_dist_valid)
                    verticality_mm = float(abs(slope) * length_m)
                except (ValueError, np.linalg.LinAlgError):
                    pass

            vp = (np.isfinite(verticality_mm)
                  and verticality_mm <= verticality_limit_mm)

            verticality_defect_point_indices = []
            verticality_defect_point_rows = []
            verticality_defect_values_mm = []
            verticality_defect_types = []
            if (np.isfinite(verticality_mm)
                    and verticality_mm > verticality_limit_mm):
                contact_local_idx = int(np.argmax(np.abs(win_dist_valid)))
                contact_dist = win_dist_valid[contact_local_idx]
                contact_row = int(ix_valid[contact_local_idx])
                contact_raw_id = int(ids[contact_row])
                signed_vert = float(
                    verticality_mm if contact_dist > 0 else -verticality_mm)
                verticality_defect_point_indices = [contact_raw_id]
                verticality_defect_point_rows = [contact_row]
                verticality_defect_values_mm = [signed_vert]
                verticality_defect_types = [
                    'outward_lean' if contact_dist > 0 else 'inward_sag']

            clipped_u = (u1 - u0) < width_m
            clipped_v = (v1 - v0) < length_m
            area = max(0.0, min(width_m, u1 - u0) * min(length_m, v1 - v0))
            center_xyz = (origin + u_ax * uc + v_ax * vc).tolist()

            windows.append({
                'grid_u': a,
                'grid_v': b,
                'point_count': int(len(ix_valid)),
                'actual_area_m2': float(area),
                'is_clipped': bool(clipped_u or clipped_v),
                'covered_source_ids': ids[ix_valid],
                'depression_mm': depression,
                'protrusion_mm': protrusion,
                'flatness_gap_mm': gap,
                'verticality_deviation_mm': verticality_mm,
                'flatness_pass': bool(gap <= flatness_limit_mm),
                'verticality_pass': bool(vp),
                'center_xyz': center_xyz,
                'defect_point_indices': defect_point_indices,
                'defect_point_rows': defect_rows,
                'defect_values_mm': defect_values_mm,
                'defect_types': defect_types,
                'verticality_defect_point_indices': verticality_defect_point_indices,
                'verticality_defect_point_rows': verticality_defect_point_rows,
                'verticality_defect_values_mm': verticality_defect_values_mm,
                'verticality_defect_types': verticality_defect_types,
            })

    total_pts = len(pts)

    def _metric_rates(windows_list, pass_key):
        w_area = sum(w.get('actual_area_m2', 0.0) for w in windows_list)
        p_area = sum(w.get('actual_area_m2', 0.0) for w in windows_list
                     if w.get(pass_key))
        w_pts = sum(w.get('point_count', 0) for w in windows_list)
        p_pts = sum(w.get('point_count', 0) for w in windows_list
                    if w.get(pass_key))
        return {
            'area_rate': (p_area / w_area) if w_area > 0 else 0.0,
            'point_rate': (p_pts / w_pts) if w_pts > 0 else 0.0,
            'pass_area_m2': p_area,
            'fail_area_m2': max(w_area - p_area, 0.0),
            'total_area_m2': w_area,
            'pass_points': p_pts,
            'total_points': w_pts,
        }

    flat_rates = _metric_rates(windows, 'flatness_pass')
    vert_rates = _metric_rates(windows, 'verticality_pass')

    flat_sample_ids, flat_sample_rows = [], []
    flat_sample_values, flat_sample_types = [], []
    vert_sample_ids, vert_sample_rows = [], []
    vert_sample_values, vert_sample_types = [], []
    for window in windows:
        flat_sample_ids.extend(window.get('defect_point_indices', []))
        flat_sample_rows.extend(window.get('defect_point_rows', []))
        flat_sample_values.extend(window.get('defect_values_mm', []))
        flat_sample_types.extend(window.get('defect_types', []))
        vert_sample_ids.extend(window.get('verticality_defect_point_indices', []))
        vert_sample_rows.extend(window.get('verticality_defect_point_rows', []))
        vert_sample_values.extend(window.get('verticality_defect_values_mm', []))
        vert_sample_types.extend(window.get('verticality_defect_types', []))

    area_resolution = max(min(width_m, length_m, 0.01), 1e-4)
    occupied_u = np.floor((u_all - u0) / area_resolution).astype(np.int64)
    occupied_v = np.floor((v_all - v0) / area_resolution).astype(np.int64)
    occupied = np.unique(np.column_stack((occupied_u, occupied_v)), axis=0)
    valid_area = float(len(occupied) * area_resolution * area_resolution)

    return {
        'windows': windows,
        'overall': {
            'window_count': len(windows),
            'point_count': total_pts,
            'valid_detection_area_m2': valid_area,
            'flatness_primary_area_rate': flat_rates['area_rate'],
            'flatness_secondary_point_rate': flat_rates['point_rate'],
            'verticality_primary_area_rate': vert_rates['area_rate'],
            'verticality_secondary_point_rate': vert_rates['point_rate'],
            'global_tilt_verticality_mm': np.nan,
        },
        'parameters': {
            'window_length_m': length_m,
            'window_width_m': width_m,
            'step_u_m': width_m,
            'step_v_m': length_m,
        },
        'defect_samples': {
            'flatness': {
                'raw_ids': np.asarray(flat_sample_ids, dtype=np.int64),
                'source_rows': np.asarray(flat_sample_rows, dtype=np.int64),
                'values_mm': np.asarray(flat_sample_values, dtype=np.float64),
                'types': list(flat_sample_types),
                'index_space': 'raw_global_rows',
            },
            'verticality': {
                'raw_ids': np.asarray(vert_sample_ids, dtype=np.int64),
                'source_rows': np.asarray(vert_sample_rows, dtype=np.int64),
                'values_mm': np.asarray(vert_sample_values, dtype=np.float64),
                'types': list(vert_sample_types),
                'index_space': 'raw_global_rows',
            },
        },
    }


# =============================================================================
# 全局平面质量评估（每窗口独立使用同一个全局平面）
# =============================================================================

def compute_global_plane_quality(points, plane_model, origin, u_axis, v_axis,
                                 length_m=2., width_m=.055,
                                 flatness_limit_mm=8.,
                                 verticality_limit_mm=10.,
                                 min_points=3,
                                 uv_bounds=None, raw_ids=None,
                                 gravity_axis=(0., 0., 1.)):
    """Measure the facade with an overlapping I-ruler sweep.

    变量命名契约（本函数内全程一致）：
      - ``u``/``v`` : 过滤后点集在 UV 框架下的投影坐标
    """
    source = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    valid = np.all(np.isfinite(source), axis=1)
    pts = source[valid]
    # full_ids: 原始全局行号（对应 processed_raw_points）
    # local_ids: 过滤后点集内的连续行号（向后兼容 source_rows）
    full_ids = (np.asarray(raw_ids, np.int64).reshape(-1)[valid]
                if raw_ids is not None else np.arange(len(pts), dtype=np.int64))
    local_ids = np.arange(len(pts), dtype=np.int64)

    plane = np.asarray(plane_model, float).reshape(4)
    plane[:3] = _unit(plane[:3])
    origin = np.asarray(origin, float)
    u_axis = _unit(u_axis)
    v_axis = _unit(v_axis)

    rel = pts - origin
    u = rel @ u_axis
    v = rel @ v_axis

    if uv_bounds is not None:
        u0, u1, v0, v1 = map(float, uv_bounds)
    else:
        u0, u1 = np.quantile(u, [.005, .995])
        v0, v1 = np.quantile(v, [.005, .995])
        u0, u1 = float(u0), float(u1)
        v0, v1 = float(v0), float(v1)

    inside = (u >= u0) & (u <= u1) & (v >= v0) & (v <= v1)
    if inside.sum() < min_points:
        inside = np.ones(len(pts), bool)
        u0, u1 = float(u.min()), float(u.max())
        v0, v1 = float(v.min()), float(v.max())

    pts, full_ids, local_ids, u, v = (
        pts[inside], full_ids[inside], local_ids[inside], u[inside], v[inside])

    # Signed distance to plane in MILLIMETRES
    distances_mm = (pts @ plane[:3] + plane[3]) * 1000.0

    width_m = max(float(width_m), 1e-9)
    length_m = max(float(length_m), 1e-9)
    u_centers = _window_centers(u0, u1, width_m, width_m)
    v_centers = _window_centers(v0, v1, length_m, length_m)

    gravity = _unit(gravity_axis)
    normal = plane[:3]
    normal_gravity = float(np.clip(abs(np.dot(normal, gravity)), 0.0, 1.0))
    plane_tilt_angle_rad = float(np.arcsin(normal_gravity))
    plane_verticality_mm = float(
        np.tan(plane_tilt_angle_rad) * length_m * 1000.0)

    windows = []
    for a, uc in enumerate(u_centers):
        for b, vc in enumerate(v_centers):
            u_lo, u_hi = uc - width_m / 2.0, uc + width_m / 2.0
            v_lo, v_hi = vc - length_m / 2.0, vc + length_m / 2.0
            ix = np.flatnonzero((u >= u_lo) & (u <= u_hi) &
                                (v >= v_lo) & (v <= v_hi))
            if len(ix) < min_points:
                continue

            win_dist = distances_mm[ix]
            depression = max(0.0, float(-win_dist.min()))
            protrusion = max(0.0, float(win_dist.max()))
            gap = max(depression, protrusion)

            defect_mask = np.abs(win_dist) > flatness_limit_mm
            if np.any(defect_mask):
                # local_ids: 过滤后点集内行号（向后兼容 source_rows）
                defect_rows = local_ids[ix][defect_mask].astype(np.int64).tolist()
                # full_ids: 原始全局行号（对应 processed_raw_points）
                defect_point_indices = full_ids[ix][defect_mask].tolist()
                defect_values_mm = win_dist[defect_mask].tolist()
                defect_types = [
                    'depression' if val < 0 else 'protrusion'
                    for val in defect_values_mm
                ]
            else:
                defect_rows, defect_point_indices = [], []
                defect_values_mm, defect_types = [], []

            v_pts = v[ix]
            verticality_mm = np.nan
            if len(v_pts) >= 3 and np.ptp(v_pts) > 0.01:
                try:
                    slope, _ = _fit_line(v_pts, win_dist)
                    verticality_mm = float(abs(slope) * length_m)
                except (ValueError, np.linalg.LinAlgError):
                    pass

            vp = (np.isfinite(verticality_mm)
                  and verticality_mm <= verticality_limit_mm)

            verticality_defect_point_indices = []
            verticality_defect_point_rows = []
            verticality_defect_values_mm = []
            verticality_defect_types = []
            if (np.isfinite(verticality_mm)
                    and verticality_mm > verticality_limit_mm):
                contact_local_idx = int(np.argmax(np.abs(win_dist)))
                contact_dist = win_dist[contact_local_idx]
                contact_row = int(ix[contact_local_idx])
                contact_raw_id = int(full_ids[contact_row])
                signed_vert = float(
                    verticality_mm if contact_dist > 0 else -verticality_mm)
                verticality_defect_point_indices = [contact_raw_id]
                verticality_defect_point_rows = [contact_row]
                verticality_defect_values_mm = [signed_vert]
                verticality_defect_types = [
                    'outward_lean' if contact_dist > 0 else 'inward_sag']

            clipped_u = (u1 - u0) < width_m
            clipped_v = (v1 - v0) < length_m
            area = max(0.0, min(width_m, u1 - u0) * min(length_m, v1 - v0))
            center_xyz = (origin + u_axis * uc + v_axis * vc).tolist()

            windows.append({
                'grid_u': a,
                'grid_v': b,
                'point_count': int(len(ix)),
                'actual_area_m2': float(area),
                'is_clipped': bool(clipped_u or clipped_v),
                'covered_source_ids': full_ids[ix],
                'depression_mm': depression,
                'protrusion_mm': protrusion,
                'flatness_gap_mm': gap,
                'verticality_deviation_mm': verticality_mm,
                'flatness_pass': bool(gap <= flatness_limit_mm),
                'verticality_pass': bool(vp),
                'center_xyz': center_xyz,
                'defect_point_indices': defect_point_indices,
                'defect_point_rows': defect_rows,
                'defect_values_mm': defect_values_mm,
                'defect_types': defect_types,
                'verticality_defect_point_indices': verticality_defect_point_indices,
                'verticality_defect_point_rows': verticality_defect_point_rows,
                'verticality_defect_values_mm': verticality_defect_values_mm,
                'verticality_defect_types': verticality_defect_types,
            })

    total_pts = len(pts)

    def _metric_rates(windows_list, pass_key):
        w_area = sum(w.get('actual_area_m2', 0.0) for w in windows_list)
        p_area = sum(w.get('actual_area_m2', 0.0) for w in windows_list
                     if w.get(pass_key))
        w_pts = sum(w.get('point_count', 0) for w in windows_list)
        p_pts = sum(w.get('point_count', 0) for w in windows_list
                    if w.get(pass_key))
        return {
            'area_rate': (p_area / w_area) if w_area > 0 else 0.0,
            'point_rate': (p_pts / w_pts) if w_pts > 0 else 0.0,
            'pass_area_m2': p_area,
            'fail_area_m2': max(w_area - p_area, 0.0),
            'total_area_m2': w_area,
            'pass_points': p_pts,
            'total_points': w_pts,
        }

    flat_rates = _metric_rates(windows, 'flatness_pass')
    vert_rates = _metric_rates(windows, 'verticality_pass')

    flat_sample_ids, flat_sample_rows = [], []
    flat_sample_values, flat_sample_types = [], []
    vert_sample_ids, vert_sample_rows = [], []
    vert_sample_values, vert_sample_types = [], []
    for window in windows:
        flat_sample_ids.extend(window.get('defect_point_indices', []))
        flat_sample_rows.extend(window.get('defect_point_rows', []))
        flat_sample_values.extend(window.get('defect_values_mm', []))
        flat_sample_types.extend(window.get('defect_types', []))
        vert_sample_ids.extend(window.get('verticality_defect_point_indices', []))
        vert_sample_rows.extend(window.get('verticality_defect_point_rows', []))
        vert_sample_values.extend(window.get('verticality_defect_values_mm', []))
        vert_sample_types.extend(window.get('verticality_defect_types', []))

    area_resolution = max(min(width_m, length_m, 0.01), 1e-4)
    occupied_u = np.floor((u - u0) / area_resolution).astype(np.int64)
    occupied_v = np.floor((v - v0) / area_resolution).astype(np.int64)
    occupied = np.unique(np.column_stack((occupied_u, occupied_v)), axis=0)
    valid_area = float(len(occupied) * area_resolution * area_resolution)

    return {
        'windows': windows,
        'overall': {
            'window_count': len(windows),
            'point_count': total_pts,
            'valid_detection_area_m2': valid_area,
            'flatness_primary_area_rate': flat_rates['area_rate'],
            'flatness_secondary_point_rate': flat_rates['point_rate'],
            'verticality_primary_area_rate': vert_rates['area_rate'],
            'verticality_secondary_point_rate': vert_rates['point_rate'],
            'global_tilt_verticality_mm': plane_verticality_mm,
        },
        'parameters': {
            'window_length_m': length_m,
            'window_width_m': width_m,
            'step_u_m': width_m,
            'step_v_m': length_m,
        },
        'defect_samples': {
            'flatness': {
                'raw_ids': np.asarray(flat_sample_ids, dtype=np.int64),
                'source_rows': np.asarray(flat_sample_rows, dtype=np.int64),
                'values_mm': np.asarray(flat_sample_values, dtype=np.float64),
                'types': list(flat_sample_types),
                'index_space': 'raw_global_rows',
            },
            'verticality': {
                'raw_ids': np.asarray(vert_sample_ids, dtype=np.int64),
                'source_rows': np.asarray(vert_sample_rows, dtype=np.int64),
                'values_mm': np.asarray(vert_sample_values, dtype=np.float64),
                'types': list(vert_sample_types),
                'index_space': 'raw_global_rows',
            },
        },
    }