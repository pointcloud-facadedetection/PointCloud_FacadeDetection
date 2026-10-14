"""Unified local baseline plane fitting for facade quality assessment.

Key design decisions
--------------------
1.  **No overlap** between blocks: each physical point belongs to exactly
    one block, eliminating the "which-plane-should-I-use" ambiguity.
2.  **Robust Huber IRLS** per block: identical fitting pipeline to the
    global plane fit, only confined to the local point set.
3.  **Signed-distance cache**: once a block plane is fitted, every point
    in that block gets its signed distance to the block plane (in mm)
    pre-computed.  Both downstream algorithms re-use this cache,
    avoiding redundant SVD inside tight loops.
4.  **V-direction strip assignment**: strips/windows query the block
    that owns their *centre v-coordinate*, guaranteeing a single,
    deterministic baseline for any given measurement location.
5.  **Per-point local normals**: each block also stores its fitted plane
    normal per contained point (模拟墙面法垂直度逐点法向使用）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class BaselineBlock:
    """A non-overlapping vertical slice of the facade with its own local plane."""

    v0: float
    v1: float
    plane_model: np.ndarray       # (4,)  ax+by+cz+d=0, |normal|=1
    point_indices: np.ndarray     # indices of points belonging to this block
    signed_dist_mm: np.ndarray    # pre-computed signed distance (mm) for those points
    local_normals: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3)))  # per-point local normals (N,3)


# =============================================================================
# Geometry helpers (self-contained, no external imports)
# =============================================================================

def _unit(value):
    value = np.asarray(value, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(value)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("zero direction vector")
    return value / norm


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


def _fit_constrained_normal_plane(points, reference_normal, *, delta_m,
                                   max_iterations=50):
    """以给定法向为硬约束，鲁棒加权估计平面截距。"""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = _unit(reference_normal)
    if len(pts) < 3:
        raise ValueError('constrained plane requires at least three points')

    heights = pts @ n
    d = -float(np.median(heights))
    delta = max(float(delta_m), 1e-6)
    for _ in range(max(1, int(max_iterations))):
        residuals = heights + d
        abs_r = np.abs(residuals)
        weights = np.where(abs_r <= delta, 1.0, delta / (abs_r + 1e-12))
        weight_sum = float(weights.sum())
        if weight_sum <= 1e-12:
            break
        d_new = -float(np.sum(weights * heights) / weight_sum)
        if abs(d_new - d) < 1e-9:
            d = d_new
            break
        d = d_new
    return np.r_[n, d]


# =============================================================================
# Global / local robust plane fitting (Huber IRLS)
# =============================================================================

def fit_global_plane(points, *, reference_plane, seed=42,
                     huber_delta_m=.015, max_iterations=500,
                     convergence_tol=1e-7, angle_limit_deg=3.,
                     outlier_sigma=3.0, final_gate_sigma=2.5,
                     min_inlier_ratio=0.30, max_p95_mm=100.0,
                     enable_partition_fallback=True,
                     partition_depth_gap_m=0.08,
                     partition_min_points_ratio=0.10,
                     partition_angle_limit_deg=5.0):
    """Huber M-estimator IRLS with prior-normal initialization."""
    from algorithms.geometry import cluster_depth_significant

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
    rng = np.random.default_rng(seed)  # noqa: F841
    angle_limit_rad = np.deg2rad(float(angle_limit_deg))
    cos_limit = np.cos(angle_limit_rad)

    n = plane[:3].copy()
    d = float(plane[3])
    it = 0

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
                n_new = (np.sin(angle_limit_rad) * perp
                         + np.cos(angle_limit_rad) * n0)
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
        plane = _fit_constrained_normal_plane(
            pts[inliers], n0, delta_m=delta,
            max_iterations=min(int(max_iterations), 50))
        residuals = pts @ plane[:3] + plane[3]
        med_r = float(np.median(residuals[inliers]))

    support_mask = np.zeros(len(raw), dtype=bool)
    support_mask[finite_ids] = inliers

    abs_r = np.abs(residuals)
    inlier_ratio = float(inliers.mean())
    angle_to_ref = float(np.degrees(np.arccos(
        np.clip(abs(plane[:3] @ n0), 0, 1))))

    p95_mm = (float(np.percentile(abs_r[inliers], 95) * 1000)
              if inliers.any() else np.inf)
    fit_accepted = bool(
        inlier_ratio >= float(min_inlier_ratio)
        and p95_mm <= float(max_p95_mm)
        and angle_to_ref <= float(angle_limit_deg)
    )

    # ===== Partition Fallback =====
    if not fit_accepted and enable_partition_fallback:
        signed_all = pts @ n0 + d0
        min_points_partition = max(
            3, int(len(pts) * float(partition_min_points_ratio)))

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
            seg_mad_r = max(float(np.median(np.abs(
                seg_residuals - seg_med_r))) * 1.4826, 1e-6)
            seg_gate = float(np.clip(final_gate_sigma * seg_mad_r,
                                     0.0025, 0.008))
            seg_inliers = np.abs(seg_residuals - seg_med_r) <= seg_gate
            seg_inlier_ratio = float(seg_inliers.mean())

            if seg_inliers.sum() >= 3:
                part_plane = _fit_constrained_normal_plane(
                    seg_pts[seg_inliers], n0, delta_m=delta,
                    max_iterations=min(int(max_iterations), 50))
                part_n = part_plane[:3]
                seg_residuals = seg_pts @ part_n + part_plane[3]
                seg_med_r = float(np.median(seg_residuals[seg_inliers]))
                seg_mad_r = max(float(np.median(np.abs(
                    seg_residuals[seg_inliers] - seg_med_r))) * 1.4826, 1e-6)
                seg_gate = float(np.clip(final_gate_sigma * seg_mad_r,
                                         0.0025, 0.008))
                seg_inliers = np.abs(seg_residuals - seg_med_r) <= seg_gate

            seg_abs_r = np.abs(seg_residuals)
            seg_p95_mm = (float(np.percentile(
                seg_abs_r[seg_inliers], 95) * 1000)
                if seg_inliers.any() else np.inf)
            seg_angle_to_ref = float(np.degrees(np.arccos(
                np.clip(abs(part_n @ n0), 0, 1))))

            relaxed_inlier_ratio = float(min_inlier_ratio) * 0.85
            relaxed_p95 = float(max_p95_mm) * 1.25

            if (seg_inlier_ratio >= relaxed_inlier_ratio
                    and seg_p95_mm <= relaxed_p95
                    and seg_angle_to_ref <= float(angle_limit_deg)):
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
                plane = _fit_constrained_normal_plane(
                    pts[inliers], n0, delta_m=delta,
                    max_iterations=min(int(max_iterations), 50))
                residuals = pts @ plane[:3] + plane[3]
                med_r = float(np.median(residuals[inliers]))
                mad_r = max(float(np.median(np.abs(
                    residuals[inliers] - med_r))) * 1.4826, 1e-6)
                gate = float(np.clip(final_gate_sigma * mad_r,
                                     0.0025, 0.008))
                inliers = np.abs(residuals - med_r) <= gate

            support_mask = np.zeros(len(raw), dtype=bool)
            support_mask[finite_ids] = inliers

            abs_r = np.abs(residuals)
            inlier_ratio = float(inliers.mean())
            angle_to_ref = float(np.degrees(np.arccos(
                np.clip(abs(plane[:3] @ n0), 0, 1))))
            p95_mm = (float(np.percentile(abs_r[inliers], 95) * 1000)
                      if inliers.any() else np.inf)

            return {
                'plane_model': plane.astype(float),
                'fit_accepted': True,
                'inlier_count': int(inliers.sum()),
                'point_count': int(len(pts)),
                'inlier_ratio': inlier_ratio,
                'residual_mad_mm': float(mad_r * 1000),
                'p50_abs_residual_mm': float(np.percentile(abs_r, 50) * 1000),
                'p95_abs_residual_mm': float(
                    np.percentile(abs_r, 95) * 1000),
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


# =============================================================================
# Interval baseline blocks (non-overlapping)
# =============================================================================

def fit_interval_baseline_planes(
    points,
    reference_plane,
    interval_size_m,
    origin,
    u_axis,
    v_axis,
    *,
    seed=42,
    huber_delta_m=0.002,
    max_iterations=500,
    convergence_tol=1e-7,
    angle_limit_deg=2.0,
    outlier_sigma=2.5,
    final_gate_sigma=2.5,
    min_inlier_ratio=0.20,
    max_p95_mm=50.0,
    enable_partition_fallback=True,
    partition_depth_gap_m=0.02,
    partition_min_points_ratio=0.15,
    continuity_constraint_mm=5.0,
) -> list[BaselineBlock]:
    """Fit robust local planes along the v-axis in non-overlapping intervals.

    Returns
    -------
    list[BaselineBlock]
        Blocks sorted by ascending v0.  Each block additionally stores its
        fitted plane normal per contained point (local_normals, (N,3)) for
        the simulated-wall verticality algorithm.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 3:
        raise ValueError("fit_interval_baseline_planes requires at least 3 points")

    u_ax = _unit(u_axis)
    v_ax = _unit(v_axis)
    origin = np.asarray(origin, dtype=float).reshape(3)

    rel = pts - origin
    v_all = rel @ v_ax

    v_min, v_max = float(v_all.min()), float(v_all.max())
    size = max(float(interval_size_m), 1e-6)

    n_blocks = max(1, int(np.ceil((v_max - v_min) / size)))

    blocks: list[BaselineBlock] = []
    prev_plane = np.asarray(reference_plane, dtype=np.float64).reshape(4)
    continuity_m = max(float(continuity_constraint_mm) / 1000.0, 1e-6)

    for i in range(n_blocks):
        v0 = v_min + i * size
        v1 = v_max if i == n_blocks - 1 else v_min + (i + 1) * size

        if i < n_blocks - 1:
            mask = (v_all >= v0) & (v_all < v1)
        else:
            mask = (v_all >= v0) & (v_all <= v1)

        indices = np.flatnonzero(mask)
        if len(indices) < 3:
            ref = np.asarray(reference_plane, dtype=np.float64).reshape(4)
            blocks.append(
                BaselineBlock(
                    v0=v0,
                    v1=v1,
                    plane_model=ref,
                    point_indices=indices,
                    signed_dist_mm=np.zeros(len(indices), dtype=np.float64),
                    local_normals=np.tile(_unit(ref[:3]), (len(indices), 1)),
                )
            )
            continue

        block_pts = pts[indices]
        try:
            fit = fit_global_plane(
                block_pts,
                reference_plane=reference_plane,
                seed=seed,
                huber_delta_m=huber_delta_m,
                max_iterations=max_iterations,
                convergence_tol=convergence_tol,
                angle_limit_deg=angle_limit_deg,
                outlier_sigma=outlier_sigma,
                final_gate_sigma=final_gate_sigma,
                min_inlier_ratio=min_inlier_ratio,
                max_p95_mm=max_p95_mm,
                enable_partition_fallback=enable_partition_fallback,
                partition_depth_gap_m=partition_depth_gap_m,
                partition_min_points_ratio=partition_min_points_ratio,
            )
        except Exception:
            fit = {
                "plane_model": np.asarray(reference_plane, dtype=np.float64).reshape(4),
                "fit_accepted": False,
            }

        plane = np.asarray(fit["plane_model"], dtype=np.float64).reshape(4)
        plane[:3] = _unit(plane[:3])

        # --- continuity constraint between adjacent blocks ---
        if i > 0:
            d_jump = abs(plane[3] - prev_plane[3])
            if d_jump > continuity_m:
                blend_plane = prev_plane.copy()
                blend_plane[:3] = _unit(reference_plane[:3])
                try:
                    fit = fit_global_plane(
                        block_pts,
                        reference_plane=blend_plane,
                        seed=seed,
                        huber_delta_m=huber_delta_m,
                        max_iterations=max_iterations,
                        convergence_tol=convergence_tol,
                        angle_limit_deg=angle_limit_deg,
                        outlier_sigma=outlier_sigma,
                        final_gate_sigma=final_gate_sigma,
                        min_inlier_ratio=min_inlier_ratio,
                        max_p95_mm=max_p95_mm,
                        enable_partition_fallback=enable_partition_fallback,
                        partition_depth_gap_m=partition_depth_gap_m,
                        partition_min_points_ratio=partition_min_points_ratio,
                    )
                    new_plane = np.asarray(fit["plane_model"], dtype=np.float64).reshape(4)
                    new_plane[:3] = _unit(new_plane[:3])
                    if abs(new_plane[3] - prev_plane[3]) < d_jump:
                        plane = new_plane
                except Exception:
                    pass

        prev_plane = plane.copy()

        dist_m = block_pts @ plane[:3] + plane[3]
        dist_mm = dist_m * 1000.0

        blocks.append(
            BaselineBlock(
                v0=v0,
                v1=v1,
                plane_model=plane,
                point_indices=indices,
                signed_dist_mm=dist_mm,
                local_normals=np.tile(plane[:3], (len(indices), 1)),
            )
        )

    return blocks


def compute_block_signed_distances_mm(points, blocks, origin, v_axis) -> np.ndarray:
    """Return a full-array (N,) signed-distance map in mm.

    For every input point the function finds the block that owns the
    point (by v-coordinate) and returns the signed distance to that
    block's plane.  Points outside all block ranges receive NaN.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = len(pts)
    result = np.full(n, np.nan, dtype=np.float64)

    origin = np.asarray(origin, dtype=float).reshape(3)
    v_ax = _unit(v_axis)
    v_all = (pts - origin) @ v_ax

    for block in blocks:
        mask = (v_all >= block.v0) & (v_all <= block.v1)
        if not np.any(mask):
            continue

        block_pts = pts[mask]
        dist_m = block_pts @ block.plane_model[:3] + block.plane_model[3]
        result[mask] = dist_m * 1000.0

    return result


def compute_block_local_normals(points, blocks, origin, v_axis) -> np.ndarray:
    """Return a full-array (N, 3) per-point local-normal map.

    Each point receives the normal of the block that owns it (by
    v-coordinate).  Points outside all block ranges receive NaN.
    Used by the simulated-wall verticality algorithm
    （逐点法向 vs 铅垂线，global_plane_quality.py 专用）。
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = len(pts)
    result = np.full((n, 3), np.nan, dtype=np.float64)

    origin = np.asarray(origin, dtype=float).reshape(3)
    v_ax = _unit(v_axis)
    v_all = (pts - origin) @ v_ax

    for block in blocks:
        mask = (v_all >= block.v0) & (v_all <= block.v1)
        if not np.any(mask):
            continue
        if len(block.local_normals) == int(mask.sum()):
            result[mask] = block.local_normals
        else:
            result[mask] = block.plane_model[:3]

    return result


def find_block_for_v(v_coord: float, blocks: list) -> int | None:
    """Return the index of the block that owns *v_coord*.

    Compatible with both BaselineBlock dataclass and legacy dict blocks.
    Because blocks are non-overlapping and gap-free, this is O(log N)
    in the number of blocks (blocks are kept sorted).
    """
    if not blocks:
        return None

    def _v0(b):
        return b.v0 if hasattr(b, 'v0') else b.get('v0', float('-inf'))

    def _v1(b):
        return b.v1 if hasattr(b, 'v1') else b.get('v1', float('inf'))

    lo, hi = 0, len(blocks)
    while lo < hi:
        mid = (lo + hi) // 2
        b = blocks[mid]
        bv0 = _v0(b)
        bv1 = _v1(b)
        if v_coord < bv0:
            hi = mid
        elif v_coord > bv1:
            lo = mid + 1
        else:
            return mid

    first_v0 = _v0(blocks[0])
    last_v1 = _v1(blocks[-1])
    if v_coord < first_v0:
        return 0
    if v_coord > last_v1:
        return len(blocks) - 1
    return None