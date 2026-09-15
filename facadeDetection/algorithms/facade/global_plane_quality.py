"""Robust global facade plane fitting and batched window measurements."""
from __future__ import annotations

import numpy as np
from algorithms.geometry import cluster_depth_significant


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


def _plane_from_points(points, weights=None):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if weights is None:
        center = points.mean(axis=0)
        covariance = (points - center).T @ (points - center)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        total = max(float(weights.sum()), 1e-12)
        center = (points * weights[:, None]).sum(axis=0) / total
        centered = points - center
        covariance = (centered * weights[:, None]).T @ centered / total
    _, _, vh = np.linalg.svd(covariance, full_matrices=False)
    normal = _unit(vh[-1])
    return np.r_[normal, -float(normal @ center)]


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


def fit_global_plane(points, *, reference_plane, seed=42,
                     huber_delta_m=.015, max_iterations=500,
                     convergence_tol=1e-7, angle_limit_deg=3.,
                     outlier_sigma=3.0, final_gate_sigma=2.5,
                     min_inlier_ratio=0.30, max_p95_mm=100.0,
                     enable_partition_fallback=True,
                     partition_depth_gap_m=0.08,
                     partition_min_points_ratio=0.10,
                     partition_angle_limit_deg=5.0):
    """Huber M-estimator IRLS with prior-normal initialization.

    Robust partition fallback: when the global single-plane fit fails
    acceptance criteria, the point cloud is sliced along the reference
    normal into significant depth layers. The largest consistent layer
    (normal within ``partition_angle_limit_deg`` of reference) is
    promoted as the facade master plane, enabling reliable quality
    assessment on facades with balconies, openings, or decorations.
    """
    raw = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    finite_mask = np.all(np.isfinite(raw), axis=1)
    finite_ids = np.flatnonzero(finite_mask)
    pts = raw[finite_mask]
    if len(pts) < 3:
        raise ValueError('global plane requires at least three finite points')

    n0 = _unit(reference_plane[:3])
    d0 = -float(np.median(pts @ n0))
    plane = np.r_[n0, d0]

    r0 = pts @ n0 + d0
    med0 = float(np.median(r0))
    mad0 = max(float(np.median(np.abs(r0 - med0))) * 1.4826, 1e-6)
    w_init = np.where(np.abs(r0 - med0) <= outlier_sigma * mad0, 1.0, 0.0)

    delta = max(float(huber_delta_m), 1e-6)
    rng = np.random.default_rng(seed)
    angle_limit_rad = np.deg2rad(float(angle_limit_deg))
    cos_limit = np.cos(angle_limit_rad)

    n = plane[:3].copy()
    d = float(plane[3])

    for it in range(max_iterations):
        residuals = pts @ n + d

        abs_r = np.abs(residuals)
        w = np.where(abs_r <= delta, 1.0, delta / (abs_r + 1e-12))
        w *= w_init

        w_sum = float(w.sum())
        if w_sum < 3:
            break

        centroid = (pts * w[:, None]).sum(axis=0) / w_sum
        centered = pts - centroid
        cov = (centered * w[:, None]).T @ centered / w_sum

        _, _, vh = np.linalg.svd(cov, full_matrices=False)
        n_new = _unit(vh[-1])

        if n_new @ n0 < 0:
            n_new = -n_new

        cos_angle = float(n_new @ n0)
        if cos_angle < cos_limit:
            perp = n_new - cos_angle * n0
            perp_norm = np.linalg.norm(perp)
            if perp_norm > 1e-12:
                perp = perp / perp_norm
                n_new = np.sin(angle_limit_rad) * perp + np.cos(angle_limit_rad) * n0
                n_new = _unit(n_new)

        d_new = -float((pts * w[:, None]).sum(axis=0) @ n_new) / w_sum

        dn = float(np.linalg.norm(n_new - n))
        dd = abs(d_new - d)
        if dn < convergence_tol and dd < convergence_tol:
            break

        n, d = n_new, d_new

    plane = np.r_[n, d]

    residuals = pts @ plane[:3] + plane[3]
    med_r = float(np.median(residuals))
    mad_r = max(float(np.median(np.abs(residuals - med_r))) * 1.4826, 1e-6)

    gate = float(np.clip(final_gate_sigma * mad_r, 0.0025, 0.008))
    inliers = np.abs(residuals - med_r) <= gate

    if inliers.sum() >= 3:
        plane = _plane_from_points(pts[inliers])
        if plane[:3] @ n0 < 0:
            plane = -plane
        residuals = pts @ plane[:3] + plane[3]
        med_r = float(np.median(residuals[inliers]))

    support_mask = np.zeros(len(raw), dtype=bool)
    support_mask[finite_ids] = inliers

    abs_r = np.abs(residuals)
    inlier_ratio = float(inliers.mean())
    angle_to_ref = float(np.degrees(np.arccos(np.clip(abs(plane[:3] @ n0), 0, 1))))

    p95_mm = float(np.percentile(abs_r[inliers], 95) * 1000) if inliers.any() else np.inf
    fit_accepted = bool(
        inlier_ratio >= float(min_inlier_ratio)
        and p95_mm <= float(max_p95_mm)
        and angle_to_ref <= float(angle_limit_deg)
    )

    # ===== Partition Fallback =====
    if not fit_accepted and enable_partition_fallback:
        signed_all = pts @ n0 + d0
        min_points_partition = max(3, int(len(pts) * float(partition_min_points_ratio)))

        labels, medians = cluster_depth_significant(
            signed_all,
            tolerance=None,
            min_gap=float(partition_depth_gap_m),
            min_points=min_points_partition,
        )

        best_partition = None
        best_score = -1.0
        cos_part_limit = np.cos(np.deg2rad(float(partition_angle_limit_deg)))

        for seg_idx in range(len(medians)):
            seg_mask = labels == seg_idx
            seg_count = int(seg_mask.sum())
            if seg_count < min_points_partition:
                continue

            seg_pts = pts[seg_mask]
            part_plane = _plane_from_points(seg_pts)
            part_n = part_plane[:3]

            if abs(float(part_n @ n0)) < cos_part_limit:
                continue

            seg_residuals = seg_pts @ part_n + part_plane[3]
            seg_med_r = float(np.median(seg_residuals))
            seg_mad_r = max(float(np.median(np.abs(seg_residuals - seg_med_r))) * 1.4826, 1e-6)
            seg_gate = float(np.clip(final_gate_sigma * seg_mad_r, 0.0025, 0.008))
            seg_inliers = np.abs(seg_residuals - seg_med_r) <= seg_gate
            seg_inlier_ratio = float(seg_inliers.mean())

            if seg_inliers.sum() >= 3:
                part_plane = _plane_from_points(seg_pts[seg_inliers])
                part_n = part_plane[:3]
                seg_residuals = seg_pts @ part_n + part_plane[3]
                seg_med_r = float(np.median(seg_residuals[seg_inliers]))

            seg_abs_r = np.abs(seg_residuals)
            seg_p95_mm = float(np.percentile(seg_abs_r[seg_inliers], 95) * 1000) if seg_inliers.any() else np.inf
            seg_angle_to_ref = float(np.degrees(np.arccos(np.clip(abs(part_n @ n0), 0, 1))))

            relaxed_inlier_ratio = float(min_inlier_ratio) * 0.85
            relaxed_p95 = float(max_p95_mm) * 1.25

            if (seg_inlier_ratio >= relaxed_inlier_ratio and
                    seg_p95_mm <= relaxed_p95 and
                    seg_angle_to_ref <= float(angle_limit_deg)):
                score = seg_count * seg_inlier_ratio
                if score > best_score:
                    best_score = score
                    best_partition = {
                        'plane': part_plane,
                        'inliers': seg_inliers,
                        'seg_mask': seg_mask,
                        'inlier_ratio': seg_inlier_ratio,
                        'p95_mm': seg_p95_mm,
                        'angle_to_ref': seg_angle_to_ref,
                        'med_r': seg_med_r,
                        'mad_r': seg_mad_r,
                        'gate': seg_gate,
                    }

        if best_partition is not None:
            bp = best_partition
            plane = bp['plane']
            n = plane[:3]
            d = plane[3]

            residuals = pts @ n + d
            med_r = bp['med_r']
            mad_r = bp['mad_r']
            gate = bp['gate']
            inliers = np.zeros(len(pts), dtype=bool)
            inliers[bp['seg_mask']] = bp['inliers']

            if inliers.sum() >= 3:
                plane = _plane_from_points(pts[inliers])
                if plane[:3] @ n0 < 0:
                    plane = -plane
                residuals = pts @ plane[:3] + plane[3]
                med_r = float(np.median(residuals[inliers]))
                mad_r = max(float(np.median(np.abs(residuals[inliers] - med_r))) * 1.4826, 1e-6)
                gate = float(np.clip(final_gate_sigma * mad_r, 0.0025, 0.008))
                inliers = np.abs(residuals - med_r) <= gate

            support_mask = np.zeros(len(raw), dtype=bool)
            support_mask[finite_ids] = inliers

            abs_r = np.abs(residuals)
            inlier_ratio = float(inliers.mean())
            angle_to_ref = float(np.degrees(np.arccos(np.clip(abs(plane[:3] @ n0), 0, 1))))
            p95_mm = float(np.percentile(abs_r[inliers], 95) * 1000) if inliers.any() else np.inf

            return {
                'plane_model': plane.astype(float),
                'fit_accepted': True,
                'inlier_count': int(inliers.sum()),
                'point_count': int(len(pts)),
                'inlier_ratio': inlier_ratio,
                'residual_mad_mm': float(mad_r * 1000),
                'p50_abs_residual_mm': float(np.percentile(abs_r, 50) * 1000),
                'p95_abs_residual_mm': float(np.percentile(abs_r, 95) * 1000),
                'max_abs_residual_mm': float(abs_r.max() * 1000),
                'normal_angle_to_reference_deg': angle_to_ref,
                'support_limit_m': gate,
                'support_mask': support_mask,
                'iterations': it + 1,
                'partition_fallback': True,
            }

    return {
        'plane_model': plane.astype(float),
        'fit_accepted': fit_accepted,
        'inlier_count': int(inliers.sum()),
        'point_count': int(len(pts)),
        'inlier_ratio': inlier_ratio,
        'residual_mad_mm': float(mad_r * 1000),
        'p50_abs_residual_mm': float(np.percentile(abs_r, 50) * 1000),
        'p95_abs_residual_mm': float(np.percentile(abs_r, 95) * 1000),
        'max_abs_residual_mm': float(abs_r.max() * 1000),
        'normal_angle_to_reference_deg': angle_to_ref,
        'support_limit_m': gate,
        'support_mask': support_mask,
        'iterations': it + 1,
        'partition_fallback': False,
    }


def compute_global_plane_quality(points, plane_model, origin, u_axis, v_axis,
                                 length_m=2., width_m=.055,
                                 flatness_limit_mm=8., verticality_limit_mm=10.,
                                 min_points=3,
                                 uv_bounds=None, raw_ids=None,
                                 gravity_axis=(0., 0., 1.)):
    """Measure the facade with an overlapping I-ruler sweep.

    Verticality is computed independently per window as the absolute slope of
    distance-to-plane vs height, multiplied by `length_m` (mm). No fallback to
    global tilt is used; windows with insufficient vertical span are marked NaN.
    """
    source = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    valid = np.all(np.isfinite(source), axis=1)
    pts = source[valid]
    ids = (np.asarray(raw_ids, np.int64).reshape(-1)[valid]
           if raw_ids is not None else np.arange(len(pts)))

    plane = np.asarray(plane_model, float).reshape(4)
    plane[:3] = _unit(plane[:3])
    origin, u_axis, v_axis = np.asarray(origin, float), _unit(u_axis), _unit(v_axis)

    rel = pts - origin
    u, v = rel @ u_axis, rel @ v_axis

    u0, u1, v0, v1 = map(float, uv_bounds) if uv_bounds is not None else (
        *np.quantile(u, [.005, .995]), *np.quantile(v, [.005, .995])
    )
    inside = (u >= u0) & (u <= u1) & (v >= v0) & (v <= v1)
    if inside.sum() < min_points:
        inside = np.ones(len(pts), bool)
        u0, u1, v0, v1 = u.min(), u.max(), v.min(), v.max()

    pts, ids, u, v = pts[inside], ids[inside], u[inside], v[inside]

    # Signed distance to plane in MILLIMETRES
    distances_mm = (pts @ plane[:3] + plane[3]) * 1000.0

    # The global method uses the same physical sampling convention as an
    # I-ruler: width is the transverse pitch and length is the longitudinal
    # pitch.  The footprint itself remains width x length, therefore adjacent
    # windows touch for the normal case and the final position is explicitly
    # appended when arange cannot land on the edge.
    width_m = max(float(width_m), 1e-9)
    length_m = max(float(length_m), 1e-9)
    u_centers = _window_centers(u0, u1, width_m, width_m)
    v_centers = _window_centers(v0, v1, length_m, length_m)

    # Global tilt reference (only for logging, not used per‑window)
    gravity = _unit(gravity_axis)
    normal = plane[:3]
    normal_gravity = float(np.clip(abs(np.dot(normal, gravity)), 0.0, 1.0))
    plane_tilt_angle_rad = float(np.arcsin(normal_gravity))
    plane_verticality_mm = float(np.tan(plane_tilt_angle_rad) * length_m * 1000.0)

    windows = []
    for a, uc in enumerate(u_centers):
        for b, vc in enumerate(v_centers):
            # The rectangle is centred on the physical ruler centre.  This
            # is intentionally not a bin assignment: overlapping footprints
            # must be allowed to inspect the same original points.
            u_lo, u_hi = uc - width_m / 2.0, uc + width_m / 2.0
            v_lo, v_hi = vc - length_m / 2.0, vc + length_m / 2.0
            ix = np.flatnonzero((u >= u_lo) & (u <= u_hi) &
                                (v >= v_lo) & (v <= v_hi))
            if len(ix) < min_points:
                continue

            # --- flatness: max deviation from global plane ---
            win_dist = distances_mm[ix]
            depression = max(0.0, float(-win_dist.min()))
            protrusion = max(0.0, float(win_dist.max()))
            gap = max(depression, protrusion)

            # 逐点缺陷索引与凹凸分类（模拟墙面平整度专用）
            defect_mask = np.abs(win_dist) > flatness_limit_mm
            # Keep the render domain explicit.  ``ids`` may be raw database IDs,
            # while the renderer receives ``source``.  The old contract exposed
            # only ids and forced the renderer to guess the index space.
            defect_rows = ix[defect_mask].astype(np.int64).tolist() if np.any(defect_mask) else []
            defect_point_indices = ids[ix][defect_mask].tolist() if np.any(defect_mask) else []
            defect_values_mm = win_dist[defect_mask].tolist() if np.any(defect_mask) else []
            defect_types = []
            for val in defect_values_mm:
                if val < 0:
                    defect_types.append('depression')
                else:
                    defect_types.append('protrusion')

            # --- verticality: LOCAL trend of deviation vs height ---
            v_pts = v[ix]
            verticality_mm = np.nan

            if len(v_pts) >= 3 and np.ptp(v_pts) > 0.01:
                try:
                    slope, _ = _fit_line(v_pts, win_dist)
                    verticality_mm = float(abs(slope) * length_m)
                except (ValueError, np.linalg.LinAlgError):
                    pass

            vp = (np.isfinite(verticality_mm) and verticality_mm <= verticality_limit_mm)

            # Report the physical footprint, clipped only for dimensions smaller
            # than the ruler.  Rate calculation is performed by the service using
            # the effective-point mask, so overlapping areas are never summed.
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
                'covered_source_ids': ids[ix],
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
            })

    # ------------------------------------------------------------------
    # These raw rates are retained for standalone callers.  The facade service
    # replaces them with occupancy-mask rates so pass/fail areas are additive.
    # ------------------------------------------------------------------
    total_pts = len(pts)

    def _metric_rates(windows_list, pass_key):
        w_area = sum(w.get('actual_area_m2', 0.0) for w in windows_list)
        p_area = sum(w.get('actual_area_m2', 0.0) for w in windows_list if w.get(pass_key))
        w_pts = sum(w.get('point_count', 0) for w in windows_list)
        p_pts = sum(w.get('point_count', 0) for w in windows_list if w.get(pass_key))
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

    # Canonical point-level contract.  IDs are always the raw IDs supplied to
    # this function; values remain signed so consumers can distinguish
    # depression (negative) from protrusion (positive).
    sample_ids, sample_rows, sample_values, sample_types = [], [], [], []
    for window in windows:
        sample_ids.extend(window.get('defect_point_indices', []))
        sample_rows.extend(window.get('defect_point_rows', []))
        sample_values.extend(window.get('defect_values_mm', []))
        sample_types.extend(window.get('defect_types', []))

    # A standalone caller must receive the same physical-domain area contract
    # as the orchestration service: occupied projected cells only.  Never use
    # a window sum here because adjacent windows may overlap at their borders.
    area_resolution = min(width_m, length_m, 0.01)
    area_resolution = max(area_resolution, 1e-4)
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
            # Reference tilt (for information only)
            'global_tilt_verticality_mm': plane_verticality_mm,
        },
        'parameters': {
            'window_length_m': length_m,
            'window_width_m': width_m,
            'step_u_m': width_m,
            'step_v_m': length_m,
        },
        'defect_samples': {
            'raw_ids': np.asarray(sample_ids, dtype=np.int64),
            'source_rows': np.asarray(sample_rows, dtype=np.int64),
            'values_mm': np.asarray(sample_values, dtype=np.float64),
            'types': list(sample_types),
            'index_space': 'source_row',
        },
    }