"""Robust global facade plane fitting and batched window measurements.

职责边界（模拟墙面法 / global_plane 链路）：
  - 平整度：I 形 2m×0.055m 窗口逐点测量；预处理与靠尺法完全对齐
    （SOR + select_band）；每点使用所属高度块局部基准面的带符号深度，
    超阈值点保留凹陷/凸起符号记入缺陷字典。
  - 垂直度
    窗口内全部清洗后点拟合局部窗口平面（SVD），法向 vs 铅垂线
    θ = arcsin(|n·g|)，verticality_mm = tan(θ)×2000；并以窗口底部最凸点
    为 0 基准计算各点外凸(+)/内陷(−)，存储导致倾斜的极值点供热力图。

返回字段与原版完全一致（不改动业务层契约）。
"""
from __future__ import annotations

import numpy as np

from algorithms.facade.local_baseline import (
    fit_global_plane,
    fit_interval_baseline_planes,
    compute_block_signed_distances_mm,
    compute_block_local_normals,
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


def _sor_mask_local(u, w, k=8, sigma=4.0):
    """Linear-time ordered-profile SOR approximation（与靠尺法同源）。"""
    n = len(w)
    if n <= k + 1:
        return np.ones(n, dtype=bool)
    block = 4 * int(k) + 1
    count = (n + block - 1) // block
    padded = np.pad(np.asarray(w, float), (0, count * block - n), mode='edge')
    dev = np.abs(
        padded.reshape(count, block)
        - np.median(padded.reshape(count, block), axis=1)[:, None]
    ).ravel()[:n]
    med = np.median(dev)
    threshold = med + sigma * (1.4826 * np.median(np.abs(dev - med)) + 1e-12)
    return dev <= threshold

def _fit_plane_huber(points, iterations=5, delta_m=0.005):
    """Huber 加权平面拟合（模拟墙面垂直度抗离群）。

    大残差点（窗台沿/檐口/孔洞缘等第二表面）按 Huber 权重降权，
    法向贴合窗口内主导墙面；与靠尺法"鲁棒拟合 + 残差带"的
    去噪策略同源，杜绝稀疏离群点把 SVD 法向拉斜（662mm 类离群）。
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    c = pts.mean(axis=0)
    X = pts - c
    w = np.ones(len(pts), dtype=np.float64)
    n = np.array([0., 0., 1.])
    for _ in range(max(1, int(iterations))):
        cov = (X * w[:, None]).T @ X / max(float(w.sum()), 1e-12)
        try:
            _, _, vh = np.linalg.svd(cov, full_matrices=False)
            n = vh[-1]
        except np.linalg.LinAlgError:
            break
        r = np.abs(X @ n)
        w = np.where(r <= delta_m, 1.0, delta_m / (r + 1e-12))
    nn = np.linalg.norm(n)
    return (n / nn if nn > 1e-12 else n), c

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
# 分块局部平面质量评估（模拟墙面法：平整度 + 垂直度）
# =============================================================================

def compute_block_plane_quality(points, blocks, origin, u_axis, v_axis,
                                flatness_limit_mm=8., verticality_limit_mm=10.,
                                raw_ids=None, grid_res=0.05,
                                ruler_length_m=2.0, ruler_width_m=0.055,
                                select_band_m=0.01, sor_enabled=True,
                                sor_k=8, sor_sigma=4.0, progress_cb=None):
    """模拟墙面法：2m×5.5cm I 形窗口逐点测量，每点使用所属 block 局部平面。

    ★ 修正（不改变任何返回字段）：
      1. 预处理与靠尺法完全对齐：全局一次 SOR + select_band（块基准面残差），
         替代原先无预处理的直接统计，窗框/离群点不再进入缺陷计数；
      2. 缺陷点保留带符号值：负值=凹陷，正值=凸起（与原字段语义一致：
         depression = max(0, -min)，protrusion = max(0, max)）；
      3. 垂直度：窗口内全部清洗后点拟合局部窗口平面（SVD），法向 vs
         铅垂线 θ = arcsin(|n·g|)，verticality_mm = tan(θ)×2000；
         以窗口底部最凸点为 0 基准计算各点外凸(+)/内陷(−)，超阈值窗口
         存储最外凸/最内陷两个倾斜致因点供热力图。

    符号约定（与全链路统一）：dist_mm = 点相对块基准面的外法向距离，
        正值=凸起，负值=凹陷。
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

    # ---- 预计算每个点在各自 block 局部平面上的 signed distance (mm) ----
    all_dist_mm = compute_block_signed_distances_mm(pts, blocks, origin, v_ax)

    # ---- 预处理：SOR 统计去噪（与靠尺法对齐，全局一次）----
    finite_dist = np.isfinite(all_dist_mm)
    if sor_enabled and len(pts) > 16:
        keep = _sor_mask_local(u_all, all_dist_mm, sor_k, sor_sigma)
        all_dist_mm[keep & finite_dist] = all_dist_mm[keep & finite_dist]
        all_dist_mm[~keep] = np.nan

    # ---- 预处理：select_band 表面筛选 ----
    select_band_mm = max(float(select_band_m), 1e-6) * 1000.0
    surface_mask = np.isfinite(all_dist_mm) & (np.abs(all_dist_mm) <= select_band_mm)

    # ---- 靠尺窗口滑动：步长统一为半尺长/尺宽 ----
    length_m = max(float(ruler_length_m), 1e-9)
    width_m = max(float(ruler_width_m), 1e-9)
    u_step = width_m
    v_step = length_m / 2.0
    u_centers = _window_centers(u0, u1, width_m, u_step)
    v_centers = _window_centers(v0, v1, length_m, v_step)

    gravity = np.asarray((0., 0., 1.), dtype=np.float64)
    u_order = np.argsort(u_all, kind='stable')
    u_sorted = u_all[u_order]

    windows = []
    for a, uc in enumerate(u_centers):
        u_lo, u_hi = uc - width_m / 2.0, uc + width_m / 2.0
        u_start = np.searchsorted(u_sorted, u_lo, side='left')
        u_stop = np.searchsorted(u_sorted, u_hi, side='right')
        strip_rows = u_order[u_start:u_stop]
        if len(strip_rows) >= 3:
            strip_rows = strip_rows[np.argsort(v_all[strip_rows], kind='stable')]
            v_sorted = v_all[strip_rows]
        else:
            v_sorted = np.empty(0, dtype=np.float64)
        for b, vc in enumerate(v_centers):
            v_lo, v_hi = vc - length_m / 2.0, vc + length_m / 2.0
            v_start = np.searchsorted(v_sorted, v_lo, side='left')
            v_stop = np.searchsorted(v_sorted, v_hi, side='right')
            ix = np.sort(strip_rows[v_start:v_stop])
            if len(ix) < 3:
                continue

            # ★ 仅统计通过预处理的表面点
            ix_used = ix[surface_mask[ix]]
            if len(ix_used) < 3:
                continue

            center_block_idx = _find_block_for_v(vc, blocks)
            if center_block_idx is None or center_block_idx < 0:
                continue

            win_dist = all_dist_mm[ix_used]
            if not np.any(np.isfinite(win_dist)):
                continue

            # ---- ★★ 窗口内局部去噪（与靠尺法同源）★★ ----
            # 1) 鲁棒 detrend：拟合 w~v 局部参考线（同 ruler 的 fit_line 预处理），
            #    剔除窗口内离群/第二表面点；2) MAD 自适应残差门限。
            win_v = v_all[ix_used]
            try:
                a_loc, b_loc = _fit_line(win_v, win_dist)
                resid = win_dist - (a_loc * win_v + b_loc)
                r_med = float(np.median(resid))
                r_mad = 1.4826 * float(np.median(np.abs(resid - r_med))) + 1e-9
                gate_mm = float(min(max(3.0 * r_mad, 2.0), select_band_mm))
                clean = np.abs(resid - r_med) <= gate_mm
            except ValueError:
                clean = np.isfinite(win_dist)
            if np.count_nonzero(clean) < 3:
                clean = np.isfinite(win_dist)
            if np.count_nonzero(clean) < 3:
                continue
            ix_clean = ix_used[clean]
            dist_clean = all_dist_mm[ix_clean]
            v_clean = v_all[ix_clean]

            # 平整度统计在清洗后的点集上进行（长波起伏被 detrend 吸收）
            depression = max(0.0, float(-dist_clean.min()))
            protrusion = max(0.0, float(dist_clean.max()))
            gap = max(depression, protrusion)

            # 逐点缺陷（保留符号：负值=凹陷，正值=凸起）
            defect_mask = np.abs(dist_clean) > flatness_limit_mm
            if np.any(defect_mask):
                defect_rows = ix_clean[defect_mask].astype(np.int64).tolist()
                defect_point_indices = ids[ix_clean][defect_mask].tolist()
                defect_values_mm = dist_clean[defect_mask].tolist()
                defect_types = [
                    'depression' if v < 0 else 'protrusion'
                    for v in defect_values_mm
                ]
            else:
                defect_rows, defect_point_indices = [], []
                defect_values_mm, defect_types = [], []

            # ---- 垂直度（模拟墙面法）：Huber-IRLS 鲁棒拟合局部窗口平面 ----
            # 物理模型：三维扫描逐点测量，以窗口内全部清洗后点鲁棒拟合局部
            # 窗口平面，其法向量相对铅垂线的倾斜即该窗口垂直度。
            win_pts3 = pts[ix_clean]

            verticality_mm = np.nan
            if len(win_pts3) >= 3:
                try:
                    n_win, _c3 = _fit_plane_huber(
                        win_pts3, iterations=3,
                        delta_m=max(gate_mm, 3.0) / 1000.0)
                    nn3 = np.linalg.norm(n_win)
                    if nn3 > 1e-12:
                        # 法向 180° 歧义消除：与窗口中心块基准面法向同向
                        blk = blocks[center_block_idx]
                        ref_n = (np.asarray(blk['plane_model'], float)[:3]
                                 if isinstance(blk, dict)
                                 else blk.plane_model[:3])
                        ref_n = ref_n / np.linalg.norm(ref_n)
                        if np.dot(n_win, ref_n) < 0:
                            n_win = -n_win
                        cos_ng = abs(float(np.dot(n_win, gravity)))
                        theta_win = float(np.arcsin(min(cos_ng, 1.0)))
                        verticality_mm = float(
                            np.tan(theta_win) * length_m * 1000.0)
                except np.linalg.LinAlgError:
                    verticality_mm = np.nan

            vp = (np.isfinite(verticality_mm)
                  and verticality_mm <= verticality_limit_mm)

            # ---- 窗口底部 0 基准：外凸/内陷判定与倾斜归因 ----
            band_h = min(0.1, length_m * 0.1)
            bot_band = v_clean <= (float(v_clean.min()) + band_h)
            if np.count_nonzero(bot_band) >= 1:
                d0 = float(dist_clean[bot_band].max())
            else:
                d0 = float(dist_clean.max())
            rel_mm = dist_clean - d0

            verticality_defect_point_indices = []
            verticality_defect_point_rows = []
            verticality_defect_values_mm = []
            verticality_defect_types = []
            if (np.isfinite(verticality_mm)
                    and verticality_mm > verticality_limit_mm):
                # 存储导致窗口倾斜的两个极值点：最外凸点与最内陷点
                i_out = int(np.argmax(rel_mm))
                i_in = int(np.argmin(rel_mm))
                verticality_defect_point_indices = [
                    int(ids[ix_clean[i_out]]), int(ids[ix_clean[i_in]])]
                verticality_defect_point_rows = [
                    int(ix_clean[i_out]), int(ix_clean[i_in])]
                verticality_defect_values_mm = [
                    float(rel_mm[i_out]), float(rel_mm[i_in])]
                verticality_defect_types = ['outward_lean', 'inward_sag']

            clipped_u = (u1 - u0) < width_m
            clipped_v = (v1 - v0) < length_m
            area = max(0.0, min(width_m, u1 - u0) * min(length_m, v1 - v0))
            center_xyz = (origin + u_ax * uc + v_ax * vc).tolist()

            windows.append({
                'grid_u': a,
                'grid_v': b,
                'point_count': int(len(ix_clean)),
                'actual_area_m2': float(area),
                'is_clipped': bool(clipped_u or clipped_v),
                'covered_source_ids': ids[ix_clean],
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
        if progress_cb is not None:
            progress_cb(int(100 * (a + 1) / len(u_centers)),
                        '正在计算模拟墙面质量')

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
# 全局平面质量评估（兼容旧调用；垂直度同模拟墙面法语义）
# =============================================================================

def compute_global_plane_quality(points, plane_model, origin, u_axis, v_axis,
                                 length_m=2., width_m=.055,
                                 flatness_limit_mm=8.,
                                 verticality_limit_mm=10.,
                                 min_points=3,
                                 uv_bounds=None, raw_ids=None,
                                 gravity_axis=(0., 0., 1.),
                                 local_normals=None):
    """Measure the facade with an overlapping I-ruler sweep.

    若传入 local_normals（(N,3) 点级局部法向量，通常来自
    compute_block_local_normals），垂直度按模拟墙面法语义计算
    （窗口内全部点拟合局部窗口平面）；否则使用全局平面倾角。返回字段与原版一致。
    """
    source = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    valid = np.all(np.isfinite(source), axis=1)
    pts = source[valid]
    full_ids = (np.asarray(raw_ids, np.int64).reshape(-1)[valid]
                if raw_ids is not None else np.arange(len(pts), dtype=np.int64))
    local_ids = np.arange(len(pts), dtype=np.int64)

    if local_normals is not None:
        ln_all = np.asarray(local_normals, dtype=np.float64)
        ln = ln_all[valid] if len(ln_all) == len(source) else ln_all
    else:
        ln = None

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

    pts, full_ids, local_ids, u, v, ln = (
        pts[inside], full_ids[inside], local_ids[inside], u[inside],
        v[inside], (ln[inside] if ln is not None else None))

    # Signed distance to plane in MILLIMETRES（外法向为正：正值=凸起）
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
                defect_rows = local_ids[ix][defect_mask].astype(np.int64).tolist()
                defect_point_indices = full_ids[ix][defect_mask].tolist()
                defect_values_mm = win_dist[defect_mask].tolist()
                defect_types = [
                    'depression' if val < 0 else 'protrusion'
                    for val in defect_values_mm
                ]
            else:
                defect_rows, defect_point_indices = [], []
                defect_values_mm, defect_types = [], []

            # 垂直度：无局部法向时使用全局倾角，否则拟合局部窗口平面。
            verticality_mm = plane_verticality_mm if ln is None else np.nan
            if ln is not None:
                win_n3 = ln[ix]
                finite_n3 = np.all(np.isfinite(win_n3), axis=1)
                if np.count_nonzero(finite_n3) >= 3:
                    win_pts3 = pts[ix]
                    ctr3 = win_pts3.mean(axis=0)
                    centered3 = win_pts3 - ctr3
                    try:
                        _, _, vh3 = np.linalg.svd(
                            centered3.T @ centered3, full_matrices=False)
                        n_win = vh3[-1]
                        nn3 = np.linalg.norm(n_win)
                        if nn3 > 1e-12:
                            n_win = n_win / nn3
                            if np.dot(n_win, normal) < 0:
                                n_win = -n_win
                            cos_ng = abs(float(np.dot(n_win, gravity)))
                            theta_win = float(np.arcsin(min(cos_ng, 1.0)))
                            verticality_mm = float(
                                np.tan(theta_win) * length_m * 1000.0)
                    except np.linalg.LinAlgError:
                        verticality_mm = np.nan

            vp = (np.isfinite(verticality_mm)
                  and verticality_mm <= verticality_limit_mm)

            verticality_defect_point_indices = []
            verticality_defect_point_rows = []
            verticality_defect_values_mm = []
            verticality_defect_types = []
            if (np.isfinite(verticality_mm)
                    and verticality_mm > verticality_limit_mm):
                # 窗口底部 0 基准：外凸(+)/内陷(−) 极值点存储
                win_v_l = v[ix]
                win_dist_l = distances_mm[ix]
                band_h_l = min(0.1, length_m * 0.1)
                bot_band_l = win_v_l <= (float(win_v_l.min()) + band_h_l)
                d0_l = (float(win_dist_l[bot_band_l].max())
                        if np.count_nonzero(bot_band_l) >= 1
                        else float(win_dist_l.max()))
                rel_l = win_dist_l - d0_l
                i_out_l = int(np.argmax(rel_l))
                i_in_l = int(np.argmin(rel_l))
                verticality_defect_point_indices = [
                    int(full_ids[ix][i_out_l]), int(full_ids[ix][i_in_l])]
                verticality_defect_point_rows = [
                    int(ix[i_out_l]), int(ix[i_in_l])]
                verticality_defect_values_mm = [
                    float(rel_l[i_out_l]), float(rel_l[i_in_l])]
                verticality_defect_types = ['outward_lean', 'inward_sag']

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