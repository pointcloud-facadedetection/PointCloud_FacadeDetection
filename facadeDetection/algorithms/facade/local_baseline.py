"""用于立面质量评估的统一局部基准平面拟合。

Key design decisions
--------------------
1.  **块之间无重叠**：每个物理点仅属于一个块，从而消除了“应使用哪个平面”的歧义。
2.  每个块采用**鲁棒的Huber IRLS**：仅限于局部点集。
3.  **带符号距离缓存**：一旦块平面拟合完成，该块内每个点到块平面的带符号距离（单位：毫米）
    即被预先计算出来。后续两个算法均复用此缓存，
    从而避免在紧循环中进行冗余的奇异值分解（SVD）。
4.  **V 方向条带分配**：各条带/窗口查询拥有其
    *中心 V 坐标*的块，从而确保针对任何给定测量位置，均采用单一、确定性的基线。
5.  **逐点局部法线**：每个块还存储其所含各点的拟合平面
    法线（用于模拟墙面法垂直度的逐点法向计算）。
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
                     partition_angle_limit_deg=5.0,
                     enable_p95_refit=True,
                     p95_confidence=0.50):
    """Huber M-estimator IRLS with prior-normal initialization.

    p95_confidence：可信核心保留比例（默认 50%），不是把均值±1.96σ
    误用于通常非高斯、重尾的点云残差。可信核心先由 median/MAD 定位，
    再按中心残差分布截断，随后重新估计完整平面法向和截距。
    """
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

    # ===== 可信核心重拟合（保留比例由 p95_confidence 决定） =====
    p95_ci_refit = False
    p95_threshold_mm = np.nan
    p95_core_count = 0
    p95_core_ratio = 0.0
    if enable_p95_refit and len(pts) >= 3:
        confidence = float(np.clip(p95_confidence, 0.50, 0.999))
        signed_r_all = pts @ plane[:3] + plane[3]
        center_r = float(np.median(signed_r_all))
        centered_abs = np.abs(signed_r_all - center_r)
        robust_sigma = max(
            1.4826 * float(np.median(centered_abs)), 1e-6)
        percentile_gate = float(np.percentile(
            centered_abs, confidence * 100.0))
        # P95 是主筛选；MAD 上限防止大量远离墙面的第二表面把 P95 本身抬高。
        p95_threshold = min(percentile_gate,
                            max(float(outlier_sigma), 2.5) * robust_sigma)
        p95_mask = centered_abs <= max(p95_threshold, 1e-6)
        p95_core_count = int(p95_mask.sum())
        p95_core_ratio = float(p95_core_count / len(pts))
        if p95_core_count >= max(3, int(len(pts) * min_inlier_ratio)):
            try:
                candidate = _plane_from_points(pts[p95_mask])
                if candidate[:3] @ n0 < 0:
                    candidate *= -1.0
                candidate_angle = float(np.degrees(np.arccos(
                    np.clip(candidate[:3] @ n0, -1.0, 1.0))))
                if candidate_angle <= float(angle_limit_deg):
                    # 在可信核心上再做一次 Huber 截距估计，法向来自核心 SVD。
                    plane = _fit_constrained_normal_plane(
                        pts[p95_mask], candidate[:3], delta_m=delta,
                        max_iterations=min(int(max_iterations), 50))
                    p95_ci_refit = True
                    p95_threshold_mm = p95_threshold * 1000.0
            except Exception:
                pass

    residuals = pts @ plane[:3] + plane[3]
    med_r = float(np.median(residuals))
    mad_r = max(float(np.median(np.abs(residuals - med_r))) * 1.4826, 1e-6)

    gate = float(np.clip(final_gate_sigma * mad_r, 0.0025, 0.008))
    inliers = np.abs(residuals - med_r) <= gate

    if inliers.sum() >= 3:
        # 保留可信核心重拟合得到的法向；这里只在最终支持集上稳健更新截距。
        # 若仍使用 n0，会把上面的 P95 法向改进完全覆盖掉。
        plane = _fit_constrained_normal_plane(
            pts[inliers], plane[:3], delta_m=delta,
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

    dominant_plane_ratio = 1.0
    baseline_ambiguous = False
    if (p95_confidence <= 0.55
            and mad_r >= max(0.005, partition_depth_gap_m / 8)):
        signed_all = pts @ n0 + d0
        labels, medians = cluster_depth_significant(
            signed_all, tolerance=None,
            min_gap=float(partition_depth_gap_m),
            min_points=max(3, int(len(pts) * float(min_inlier_ratio))))
        counts = np.bincount(labels[labels >= 0], minlength=len(medians))
        if len(medians) > 1 and counts.sum():
            dominant_plane_ratio = float(counts.max() / len(pts))
            baseline_ambiguous = dominant_plane_ratio < 0.65
            if baseline_ambiguous:
                fit_accepted = False

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
                'fit_accepted': not baseline_ambiguous,
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
                'p95_ci_refit': p95_ci_refit,
                'p95_threshold_mm': float(p95_threshold_mm),
                'p95_core_count': p95_core_count,
                'p95_core_ratio': p95_core_ratio,
                'dominant_plane_ratio': dominant_plane_ratio,
                'baseline_ambiguous': baseline_ambiguous,
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
        'p95_ci_refit': p95_ci_refit,
        'p95_threshold_mm': float(p95_threshold_mm),
        'p95_core_count': p95_core_count,
        'p95_core_ratio': p95_core_ratio,
                'dominant_plane_ratio': dominant_plane_ratio,
                'baseline_ambiguous': baseline_ambiguous,
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
    enable_p95_refit=False,
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

    # v 只排序一次；各 block 通过 searchsorted 取得连续区间，避免每个
    # block 都构造一个 N 长布尔掩码（原实现为 O(blocks * N) 扫描）。
    v_order = np.argsort(v_all, kind='stable')
    v_sorted = v_all[v_order]
    edges = v_min + np.arange(n_blocks + 1, dtype=np.float64) * size
    edges[-1] = v_max
    edge_pos = np.searchsorted(v_sorted, edges, side='left')
    edge_pos[-1] = len(v_sorted)

    blocks: list[BaselineBlock] = []
    prev_plane = np.asarray(reference_plane, dtype=np.float64).reshape(4)
    continuity_m = max(float(continuity_constraint_mm) / 1000.0, 1e-6)

    for i in range(n_blocks):
        v0 = v_min + i * size
        v1 = v_max if i == n_blocks - 1 else v_min + (i + 1) * size

        indices = v_order[edge_pos[i]:edge_pos[i + 1]]
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
                enable_p95_refit=enable_p95_refit,
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
                        enable_p95_refit=enable_p95_refit,
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

    # fit_interval_baseline_planes 已经保存每个 block 的精确 owner 索引和
    # 对应距离。直接散写缓存即可，避免再次投影以及 blocks 次全量扫描。
    for block in blocks:
        indices = np.asarray(block.point_indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            continue
        cached = np.asarray(block.signed_dist_mm, dtype=np.float64).reshape(-1)
        if cached.size != indices.size:
            raise ValueError('BaselineBlock signed_dist_mm 与 point_indices 长度不一致')
        result[indices] = cached

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

    for block in blocks:
        indices = np.asarray(block.point_indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            continue
        if len(block.local_normals) == len(indices):
            result[indices] = block.local_normals
        else:
            result[indices] = block.plane_model[:3]

    return result


def find_block_for_v(v_coord: float, blocks: list) -> int | None:
    """Return the index of the block that owns *v_coord*.

    Because blocks are non-overlapping and gap-free, this is O(log N)
    in the number of blocks (blocks are kept sorted).
    """
    if not blocks:
        return None

    lo, hi = 0, len(blocks)
    while lo < hi:
        mid = (lo + hi) // 2
        b = blocks[mid]
        bv0 = b.v0
        bv1 = b.v1
        if v_coord < bv0:
            hi = mid
        elif v_coord > bv1:
            lo = mid + 1
        else:
            return mid

    first_v0 = blocks[0].v0
    last_v1 = blocks[-1].v1
    if v_coord < first_v0:
        return 0
    if v_coord > last_v1:
        return len(blocks) - 1
    return None