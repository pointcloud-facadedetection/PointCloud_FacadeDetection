"""Local facade patches from spatial, normal and shared-depth consistency."""
from __future__ import annotations

import numpy as np
import open3d as o3d

from algorithms.geometry import fit_plane_weighted
from utils.logging_utils import trace
from config.settings import Config


class _DisjointSet:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.uint8)

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[item] != item:
            nxt = int(self.parent[item])
            self.parent[item] = root
            item = nxt
        return root

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def _canonical_normals(normals):
    """Return unoriented normals in a deterministic hemisphere."""
    out = np.asarray(normals, dtype=np.float64).copy()
    lengths = np.linalg.norm(out, axis=1, keepdims=True)
    out /= np.maximum(lengths, 1e-12)
    major = np.argmax(np.abs(out), axis=1)
    signs = out[np.arange(len(out)), major]
    out[signs < 0.0] *= -1.0
    return out


def _feature_summary(ids, normals, normal, points, model,
                     intensities=None, density_field=None):
    """Return compact patch diagnostics without changing segmentation gates."""
    ids = np.asarray(ids, dtype=np.int64)
    agreement = np.clip(np.abs(np.asarray(normals)[ids] @ normal), 0.0, 1.0)
    angles = np.degrees(np.arccos(agreement))
    signed = np.asarray(points)[ids] @ normal + float(model[3])
    summary = {
        'normal_angle_median_deg': float(np.median(angles)),
        'normal_angle_p95_deg': float(np.percentile(angles, 95)),
        'depth_median_m': float(np.median(signed)),
        'depth_mad_m': float(np.median(np.abs(signed - np.median(signed)))),
        'depth_p95_abs_m': float(np.percentile(np.abs(signed), 95)),
    }
    if intensities is not None:
        values = np.asarray(intensities, dtype=float).reshape(-1)[ids]
        finite = values[np.isfinite(values)]
        summary['intensity_valid_ratio'] = float(len(finite) / max(len(values), 1))
        if len(finite):
            q10, q25, q50, q75, q90 = np.percentile(
                finite, [10, 25, 50, 75, 90])
            summary.update({
                'intensity_median': float(q50),
                'intensity_iqr': float(q75 - q25),
                'intensity_p10': float(q10),
                'intensity_p90': float(q90),
            })
    if density_field is not None:
        density = np.asarray(density_field.point_density, dtype=float)[ids]
        levels = np.asarray(density_field.point_level, dtype=np.int8)[ids]
        summary.update({
            'density_median_m2': float(np.median(density)),
            'density_p10_m2': float(np.percentile(density, 10)),
            'density_dense_ratio': float(np.mean(levels == 0)),
            'density_sparse_ratio': float(np.mean(levels == 1)),
            'density_skip_ratio': float(np.mean(levels == 2)),
        })
    return summary


def _trace_patch(seed_id, ids, summary, accepted, reason, enabled):
    if not enabled:
        return
    trace('facade.patch.features', seed_id=int(seed_id), points=int(len(ids)),
          accepted=bool(accepted), reason=str(reason),
          normal_median_deg=f"{summary.get('normal_angle_median_deg', np.nan):.3f}",
          normal_p95_deg=f"{summary.get('normal_angle_p95_deg', np.nan):.3f}",
          depth_mad_m=f"{summary.get('depth_mad_m', np.nan):.5f}",
          depth_p95_m=f"{summary.get('depth_p95_abs_m', np.nan):.5f}",
          intensity_valid=f"{summary.get('intensity_valid_ratio', 0.0):.3f}",
          intensity_median=f"{summary.get('intensity_median', np.nan):.4f}",
          intensity_iqr=f"{summary.get('intensity_iqr', np.nan):.4f}",
          density_median=f"{summary.get('density_median_m2', np.nan):.2f}",
          dense_ratio=f"{summary.get('density_dense_ratio', 0.0):.3f}",
          sparse_ratio=f"{summary.get('density_sparse_ratio', 0.0):.3f}",
          skip_ratio=f"{summary.get('density_skip_ratio', 0.0):.3f}")


def _large_patch_seeds(pts, nrms, valid, voxel_size, min_count,
                       depth_hard_m, normal_angle_deg, neighbour_radius,
                       intensities=None, density_field=None,
                       debug_features=False):
    """Bounded direction/depth seeds for production-size ROIs.

    This path intentionally builds no point KNN and no Python adjacency graph.
    Direction discovery uses at most 100k deterministic samples.  Every point
    is then assigned once to an unoriented azimuth bin and a depth bin computed
    against that bin's *shared* normal.  Work is O(N), apart from a stable sort
    of compact integer labels; the number of expensive plane fits is capped.

    Spatial opening separation remains the responsibility of the existing
    extraction/UV component stage.  This function only supplies conservative
    plane seeds and therefore cannot bridge points through neighbour edges.
    """
    local_pts = np.asarray(pts[valid], dtype=np.float64)
    local_nrms = _canonical_normals(nrms[valid])
    vertical = np.abs(local_nrms[:, 2]) <= 0.35
    if np.count_nonzero(vertical) < max(3, int(min_count)):
        return []

    # Unoriented facade azimuth lies in [0, pi).  Estimate supported directions
    # on a hard-capped representative set so this stage cannot scale with all
    # pairwise relations in a million-point ROI.
    vert_ids = np.flatnonzero(vertical)
    # 大 ROI 动态采样：保证至少有 20 万样本用于方向估计，
    # 对 121 万点 ROI 方向估计更稳健，召回率提升
    max_samples = min(300000, max(100000, int(len(vert_ids) * 0.25)))
    stride = max(1, int(np.ceil(len(vert_ids) / max_samples)))
    sample = vert_ids[::stride]
    sample_az = np.mod(np.arctan2(local_nrms[sample, 1],
                                  local_nrms[sample, 0]), np.pi)
    bin_width = max(np.deg2rad(float(normal_angle_deg)), np.deg2rad(5.0))
    n_bins = max(18, int(np.ceil(np.pi / bin_width)))
    counts, edges = np.histogram(sample_az, bins=n_bins, range=(0.0, np.pi))
    min_sample_support = max(20, int(len(sample) * 0.002))
    ranked = np.argsort(-counts, kind='stable')
    selected = [int(i) for i in ranked[:16] if counts[i] >= min_sample_support]
    if not selected:
        return []

    # Do not use histogram-bin centres as physical normals.  Even a 5° centre
    # error makes rho drift by metres across a long wall and falsely creates
    # many 5cm depth layers.  Refine every selected direction from the actual
    # sampled normals assigned to that bin.
    sample_bins = np.clip(np.searchsorted(edges, sample_az, side='right') - 1,
                          0, n_bins - 1)
    refined = []
    refined_selected = []
    for bin_id in selected:
        members = sample[sample_bins == bin_id]
        if not len(members):
            continue
        direction = np.median(local_nrms[members], axis=0)
        direction[2] = 0.0
        length = float(np.linalg.norm(direction))
        if length <= 1e-12:
            continue
        refined.append(direction / length)
        refined_selected.append(bin_id)
    if not refined:
        return []
    directions = np.asarray(refined, dtype=np.float64)
    selected = refined_selected
    # Assign each vertical point to one selected shared direction.  Absolute
    # dot makes the normal orientation irrelevant.
    agreement = np.abs(local_nrms[vert_ids] @ directions.T)
    best = np.argmax(agreement, axis=1)
    best_dot = agreement[np.arange(len(vert_ids)), best]
    accepted = best_dot >= float(np.cos(np.deg2rad(normal_angle_deg)))
    accepted_ids = vert_ids[accepted]
    direction_ids = best[accepted].astype(np.int32, copy=False)
    if len(accepted_ids) < max(3, int(min_count)):
        return []

    # Crucial invariant: rho uses a shared bin direction, never each point's
    # noisy normal (which causes coordinate-dependent depth drift).
    rho = np.einsum('ij,ij->i', local_pts[accepted_ids],
                    directions[direction_ids])
    # Sorting + gap cutting avoids the absolute-origin phase error of
    # floor(rho / 5cm): one physical wall can straddle a bin boundary even when
    # its robust depth spread is only a few millimetres.  A new layer starts
    # only when adjacent ordered depths exceed the hard 5cm limit.
    depth_step = max(float(depth_hard_m), 0.01)
    depth_ids = np.empty(len(rho), dtype=np.int64)
    next_layer = 0
    for direction_id in range(len(directions)):
        members = np.flatnonzero(direction_ids == direction_id)
        if not len(members):
            continue
        order_rho = members[np.argsort(rho[members], kind='stable')]
        gaps = np.diff(rho[order_rho])
        local_layers = np.r_[0, np.cumsum(gaps > depth_step)]
        depth_ids[order_rho] = next_layer + local_layers
        next_layer += int(local_layers[-1]) + 1

    # Compact pair labels; unique operates on only two integer columns rather
    # than the former 10-column point-specific supervoxel key.
    keys = np.column_stack((direction_ids.astype(np.int64), depth_ids))
    _, inverse, group_counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True)
    patch_min = max(30, int(min_count) // 4)
    eligible = np.flatnonzero(group_counts >= patch_min)
    if not len(eligible):
        return []
    # 限制最大 eligible groups：大 ROI 中 Top-48 已足够覆盖主要立面
    max_eligible = int(getattr(Config, 'FACADE_PRECLUSTER_MAX_ELIGIBLE', 48))
    eligible = eligible[np.argsort(-group_counts[eligible], kind='stable')[:max_eligible]]

    order = np.argsort(inverse, kind='stable')
    starts = np.r_[0, np.cumsum(group_counts[:-1])]
    cos_limit = float(np.cos(np.deg2rad(normal_angle_deg)))
    seeds = []
    for seed_id, group in enumerate(eligible):
        local_group = order[starts[group]:starts[group] + group_counts[group]]
        global_ids = valid[accepted_ids[local_group]]
        patch_pts = pts[global_ids]
        try:
            model = fit_plane_weighted(patch_pts, irls_iters=2)
        except Exception:
            continue
        normal = model[:3] / (np.linalg.norm(model[:3]) + 1e-12)
        residual = np.abs(patch_pts @ normal + float(model[3]))
        # 生产环境跳过详细诊断以减少开销；仅在 debug 模式计算
        reason = 'accepted'
        if abs(float(normal[2])) > 0.35:
            reason = 'non_vertical'
        elif float(np.median(np.abs(nrms[global_ids] @ normal))) < cos_limit:
            reason = 'normal_inconsistent'
        elif float(np.percentile(residual, 95)) > float(depth_hard_m):
            reason = 'depth_hard_violation'
        elif density_field is not None and density_field.gate_indices(global_ids) == 'skip':
            reason = 'density_skip'
        accepted_seed = reason == 'accepted'
        if debug_features:
            summary = _feature_summary(
                global_ids, nrms, normal, pts, model,
                intensities=intensities, density_field=density_field)
            _trace_patch(seed_id, global_ids, summary, accepted_seed, reason,
                         debug_features=True)
        else:
            summary = {}
        if not accepted_seed:
            continue
        seeds.append({
            'direction': normal,
            'rho': float(np.median(patch_pts @ normal)),
            'spatial_ids': global_ids,
            'support': int(len(global_ids)),
            'plane_model': model,
            'layer': -1,
            'dir_bin': int(direction_ids[local_group[0]]),
            'precluster': True,
            'bounded_grid': True,
            'feature_summary': summary,
        })
    return seeds


def build_depth_normal_patch_seeds(points, normals, mask, voxel_size,
                                   min_count, depth_hard_m=0.05,
                                   normal_angle_deg=10.0,
                                   neighbour_radius=None,
                                   intensities=None, density_field=None,
                                   debug_features=False):
    """Build deterministic local patch seeds without bridging facade openings.

    Two spatial neighbours are connected only when their unoriented normals
    agree and their displacement along the shared normal is within the hard
    depth limit. Openings therefore split local patches, but those patches can
    later be merged by the constrained cross-opening candidate graph.
    """
    pts = np.asarray(points, dtype=np.float64)
    nrms = np.asarray(normals, dtype=np.float64)
    valid = np.flatnonzero(np.asarray(mask, dtype=bool))
    if len(valid) < max(3, int(min_count)):
        return []

    radius = float(neighbour_radius or max(float(voxel_size) * 3.0, 0.12))
    # Exact point graph is useful only for small fixtures.  Production ROIs use
    # the bounded direction/depth path above and never query one neighbourhood
    # per point.
    if len(valid) > 100000:
        return _large_patch_seeds(
            pts, nrms, valid, voxel_size, min_count, depth_hard_m,
            normal_angle_deg, radius, intensities=intensities,
            density_field=density_field, debug_features=debug_features)

    local_pts = pts[valid]
    local_nrms = nrms[valid]
    lengths = np.linalg.norm(local_nrms, axis=1, keepdims=True)
    local_nrms = local_nrms / np.maximum(lengths, 1e-12)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(local_pts))
    tree = o3d.geometry.KDTreeFlann(cloud)
    cos_limit = float(np.cos(np.deg2rad(normal_angle_deg)))
    dsu = _DisjointSet(len(valid))

    for i, point in enumerate(local_pts):
        _, neighbours, _ = tree.search_radius_vector_3d(point, radius)
        ni = local_nrms[i]
        for j in neighbours:
            j = int(j)
            if j <= i:
                continue
            nj = local_nrms[j]
            dot = float(ni @ nj)
            if abs(dot) < cos_limit:
                continue
            if dot < 0.0:
                nj = -nj
            shared = ni + nj
            shared /= np.linalg.norm(shared) + 1e-12
            shared_depth = abs(float((local_pts[j] - point) @ shared))
            if shared_depth <= float(depth_hard_m):
                dsu.union(i, j)

    groups = {}
    for i in range(len(valid)):
        groups.setdefault(dsu.find(i), []).append(i)

    patch_min = max(30, int(min_count) // 4)
    seeds = []
    for seed_id, members in enumerate(sorted(
            groups.values(), key=lambda ids: (-len(ids), ids[0]))):
        if len(members) < patch_min:
            continue
        local_ids = np.asarray(members, dtype=np.int64)
        global_ids = valid[local_ids]
        patch_pts = pts[global_ids]
        try:
            model = fit_plane_weighted(patch_pts, irls_iters=2)
        except Exception:
            continue
        normal = model[:3] / (np.linalg.norm(model[:3]) + 1e-12)
        summary = _feature_summary(
            global_ids, nrms, normal, pts, model,
            intensities=intensities, density_field=density_field)
        reason = 'accepted'
        if abs(float(normal[2])) > 0.35:
            reason = 'non_vertical'
        agreement = np.abs(local_nrms[local_ids] @ normal)
        if reason == 'accepted' and float(np.median(agreement)) < cos_limit:
            reason = 'normal_inconsistent'
        if (reason == 'accepted' and density_field is not None and
                density_field.gate_indices(global_ids) == 'skip'):
            reason = 'density_skip'
        accepted_seed = reason == 'accepted'
        _trace_patch(seed_id, global_ids, summary, accepted_seed, reason,
                     debug_features)
        if not accepted_seed:
            continue
        seeds.append({
            'direction': normal,
            'rho': float(np.median(patch_pts @ normal)),
            'spatial_ids': global_ids,
            'support': int(len(global_ids)),
            'plane_model': model,
            'layer': -1,
            'dir_bin': -1,
            'precluster': True,
            'feature_summary': summary,
        })
    return seeds