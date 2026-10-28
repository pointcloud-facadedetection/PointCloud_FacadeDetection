"""距离自适应立面检测 v3：5cm 深度硬隔离收口 + 密度门控 + 深度种子×Hough 融合。


1. 【硬性约束】全局唯一深度硬隔离常量 DEPTH_HARD_M（= 5cm）。所有深度方向
   的桥接/合并/去重/回收容差一律不得超过它.
2. 【密度门控】接入 density_estimator.DensityField：桶级/层级/连通域级
   'skip' 区域直接跳过；'sparse' 区域放宽最小点数（召回稀疏窗台立面）。
3. 【种子融合】build_bucket_seeds v2：深度层由「排序+间隙切割」产生（无
   箱界漂移），层内 2cm 细箱双峰切割（解决噪声带重叠漏切），边界 ε 双
   归属真正生效；Hough ρ 投票校验层中心并对断裂丢失的稀疏层补种。
4. 合并调用收口：detect_facades_adaptive 末尾跨层裁决后执行深度硬隔离
   终审，再走完形回收。
"""
from __future__ import annotations

import time
import numpy as np
import open3d as o3d

from algorithms.geometry import (
    ensure_normals, classify_plane, project_to_uv,
    cluster_normals_direction, connected_components_3d_grid, connected_components_uv_depth_grid,
    cluster_depth_significant,
    estimate_plane_area, plane_axes,
    estimate_point_ranges, adaptive_plane_tolerance, fit_plane_weighted
)
from config.settings import Config
from algorithms.facade.hough_facade import normal_hough_peaks, rho_hough_peaks
from algorithms.facade.direction_depth_cluster import build_bucket_seeds

try:
    from algorithms.facade.density_estimator import (
        build_density_field, adaptive_min_count, adaptive_tol_scale)
    _HAS_DENSITY = True
except Exception:  # 密度模块缺失时优雅降级
    _HAS_DENSITY = False

from utils.logging_utils import trace


# ==================== 全局硬约束常量 ====================
# 硬性要求：同一方向相邻空间域的候选立面，深度差超过该值必须分割为不同立面。
# 全模块任何位置不得使用大于它的深度桥接/合并/去重容差。
DEPTH_HARD_M = float(getattr(Config, 'FACADE_DEPTH_LAYER_HARD_M', 0.05))

# 去重深度差阈值：严格小于 DEPTH_HARD_M（留 5mm 数值余量）
DEDUP_DEPTH_THR_M = float(getattr(Config, 'FACADE_DEDUP_DEPTH_THRESH_M',
                                  DEPTH_HARD_M - 0.005))


# ==================== 内部工具函数 ====================

def _canonicalize_plane(model):
    """平面法向半球规范化：|n| 最大分量恒为正。

    同一面墙被相反法向各拟合一次时（粗层回映射全程 abs()，不区分符号），
    规范化后两者 n 相同、ρ=p·n 相同，去重/合并/终审才能正确工作。
    日志实锤：规范化前终审误报 depth_delta=19.4m、共享 5 万点
    （实为 2×|ρ|，ρ≈9.7m 的同一面墙法向反向）。
    """
    m = np.asarray(model, dtype=float).copy()
    norm = np.linalg.norm(m[:3]) + 1e-12
    n = m[:3] / norm
    d = m[3] / norm
    k = int(np.argmax(np.abs(n)))
    if n[k] < 0.0:
        n, d = -n, -d
    return np.r_[n, d]


def _info(fid, model, points, indices, ranges):
    model = _canonicalize_plane(model)
    n = model[:3]
    center = np.mean(points, axis=0)
    typ, label, vert, horiz = classify_plane(n)
    area, bbox = estimate_plane_area(points, n, center, typ)
    dist = np.abs(points @ n + model[3])
    return {
        'id': int(fid), 'type': typ, 'type_label': label,
        'plane_model': model.tolist(), 'normal': n.tolist(),
        'center': center.tolist(), 'area': float(area),
        'point_count': int(len(points)), 'inlier_indices': indices.astype(int).tolist(),
        'support_indices': indices.astype(int).tolist(), 'verticality': vert,
        'horizontality': horiz, 'flatness': float(np.std(dist)),
        'flatness_mean': float(np.mean(dist)), 'flatness_max': float(np.max(dist)),
        'bbox_2d': bbox, 'mean_range': float(np.mean(ranges)) if len(ranges) else 0.,
        'max_range': float(np.max(ranges)) if len(ranges) else 0.,
    }


def _uv_descriptor(f, points, grid, normal=None, origin=None):
    """使用共享平面坐标生成 occupancy 与 bbox，禁止各面独立平移原点。"""
    pm = np.asarray(f['plane_model'], dtype=float)
    n = np.asarray(normal if normal is not None else pm[:3], dtype=float)
    n /= np.linalg.norm(n) + 1e-12
    typ, _, _, _ = classify_plane(n)
    idx = np.asarray(f['inlier_indices'], dtype=int)
    u, v = plane_axes(n, typ)
    o = np.asarray(origin if origin is not None else np.zeros(3), dtype=float)
    uv = np.column_stack(((points[idx] - o) @ u, (points[idx] - o) @ v))
    cells = set(zip(np.floor(uv[:, 0] / grid).astype(np.int64),
                    np.floor(uv[:, 1] / grid).astype(np.int64)))
    return cells, (float(uv[:, 0].min()), float(uv[:, 0].max()),
                   float(uv[:, 1].min()), float(uv[:, 1].max()))


def _uv_cells_for_facade(f, points, grid):
    """兼容旧调用；检测合并阶段使用 _uv_descriptor 的共享坐标。"""
    return _uv_descriptor(f, points, grid)[0]


def _dilate_cells(cells, rad):
    out = set()
    for (a, b) in cells:
        for da in range(-rad, rad + 1):
            for db in range(-rad, rad + 1):
                out.add((a + da, b + db))
    return out


def _fill_ratio(info):
    """立面填充率：凸包面积 / UV bbox 面积。合并与完形回收共用。"""
    bb = info.get('bbox_2d') or {}
    span = max((bb.get('u_max', 0) - bb.get('u_min', 0)) *
               (bb.get('v_max', 0) - bb.get('v_min', 0)), 1e-6)
    return float(info.get('area', 0.0)) / span


def _check_depth_single_peak(points, plane_model, max_plane_dist, tolerance=None):
    """检查点集在平面深度方向是否单峰，避免合并两个平行立面。

    【v4 修正】cluster_depth_significant 的切层阈值 = max(2×tol, min_gap)。
    tolerance=None 时内部默认 tol=5cm → 实际阈值 10cm，6~9cm 双峰被误判
    单峰；tol 为距离自适应值（可达 base×4=8cm）时阈值更达 16cm。
    因此必须把 tol 截断到 min_gap/2，保证有效切层阈值恒等于 min_gap。
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 100:
        return True
    pm = np.asarray(plane_model, dtype=float)
    n = pm[:3] / (np.linalg.norm(pm[:3]) + 1e-12)
    d = float(pm[3]) / (np.linalg.norm(pm[:3]) + 1e-12)
    signed = pts @ n + d

    min_gap_eff = min(DEPTH_HARD_M, max(float(max_plane_dist) * 1.5, 0.03))
    if tolerance is not None:
        tol_arr = np.minimum(np.asarray(tolerance, dtype=np.float64).reshape(-1),
                             min_gap_eff / 2.0)
        tol_arg = float(np.median(tol_arr))
    else:
        tol_arg = min_gap_eff / 2.0     # 截断：2×tol = min_gap
    seg_labels, seg_medians = cluster_depth_significant(
        signed, tol_arg,
        min_gap=min_gap_eff,
        min_points=max(50, len(pts) // 10))
    if len(seg_medians) > 1:
        floor_pts = max(50, int(len(pts) * 0.10))
        strong = [m for k, m in enumerate(seg_medians)
                  if int(np.count_nonzero(seg_labels == k)) >= floor_pts]
        if len(strong) > 1 and (max(strong) - min(strong)) > max_plane_dist * 1.5:
            return False

    n_bins = max(20, int(np.sqrt(len(pts)) / 3))
    counts, edges = np.histogram(signed, bins=n_bins)
    for k in range(2, len(counts) - 2):
        if counts[k] < counts[k - 1] and counts[k] < counts[k + 1]:
            left_peak = max(counts[:k])
            right_peak = max(counts[k + 1:])
            if left_peak > counts[k] * 2.0 and right_peak > counts[k] * 2.0:
                left_mask = signed <= edges[k + 1]
                right_mask = signed > edges[k + 1]
                if (np.sum(left_mask) > 50 and np.sum(right_mask) > 50 and
                        abs(np.median(signed[left_mask]) - np.median(signed[right_mask])) > max_plane_dist * 1.5):
                    return False
    return True


def _aabb_gap_m(points_a, points_b):
    """计算两组点 3D AABB 的最小间距（0 表示相交）。"""
    if len(points_a) == 0 or len(points_b) == 0:
        return float('inf')
    lo_a, hi_a = points_a.min(0), points_a.max(0)
    lo_b, hi_b = points_b.min(0), points_b.max(0)
    gap = np.maximum(np.maximum(lo_a - hi_b, lo_b - hi_a), 0.0)
    return float(np.linalg.norm(gap))


# ==================== Bucket 种子非竞争提取 ====================

def _expand_single_seed(seed_plane, seed_spatial_ids, points, normals, ranges,
                        tol, cos_tol, spacing, voxel_size, base, min_count,
                        min_facade_area, irls_iters, signed_dist_tolerance,
                        fid_start=0):
    """从单个种子扩展立面。扩展逻辑与 Hough 路径完全一致。

    只在 seed_spatial_ids 范围内做初始筛选，然后 UV 连通域扩展。
    不修改任何全局状态。
    【v3】UV+Depth 连通域深度方向零容忍桥接：depth_bin = depth_gap =
    DEPTH_HARD_M（旧 depth_gap=0.12 会把深度差 12cm 的两个面闭运算桥接）。
    """
    facades = []
    fid = fid_start

    if len(seed_spatial_ids) < min_count:
        return facades, fid

    seed_plane = np.asarray(seed_plane, dtype=np.float64)
    sn = seed_plane[:3] / (np.linalg.norm(seed_plane[:3]) + 1e-12)

    ids = np.asarray(seed_spatial_ids, dtype=int)
    signed = points[ids] @ sn + seed_plane[3]
    candidate = np.abs(signed) <= tol[ids] * 1.5
    candidate &= np.abs(normals[ids] @ sn) >= cos_tol[ids]
    cids = ids[candidate]
    if len(cids) < min_count:
        return facades, fid

    tight = np.abs(points[cids] @ sn + seed_plane[3]) <= tol[cids] * signed_dist_tolerance
    cids = cids[tight]
    if len(cids) < min_count:
        return facades, fid

    ftype, _, _, _ = classify_plane(sn)
    uv, _, _, _ = project_to_uv(points[cids], seed_plane, ftype)

    local_spacing = float(np.median(spacing[cids])) if len(cids) else voxel_size
    uv_grid = max(float(voxel_size) * 2.0, local_spacing * 1.5, 0.05)

    min_opening_m = float(getattr(Config, 'FACADE_MIN_OPENING_M', 0.5))
    big_radius = max(3, int(np.ceil(min_opening_m * 2.5 / uv_grid)))
    small_radius = max(2, int(np.ceil(min_opening_m / uv_grid)))

    if getattr(Config, 'FACADE_CLOSE_RADIUS_BY_SPAN', True) and len(uv):
        span = max(float(np.ptp(uv[:, 0])), float(np.ptp(uv[:, 1])))
        close_radius = big_radius if span >= min_opening_m * 4.0 else small_radius
    else:
        close_radius = big_radius
    close_radius = min(close_radius, int(getattr(Config, 'FACADE_UV_CLOSE_MAX_CELLS', 12)))

    seed_norm = np.linalg.norm(seed_plane[:3]) + 1e-12
    n_seed = seed_plane[:3] / seed_norm
    d_seed = float(seed_plane[3]) / seed_norm
    depths = points[cids] @ n_seed + d_seed

    # 【v3/v4】深度方向桥接 = 0：bin 与 gap 均为硬约束 5cm
    depth_bin = DEPTH_HARD_M
    depth_gap = DEPTH_HARD_M
    # 【v4】tol 截断到 gap/2：否则 cluster_depth_significant 内部
    # max(2×tol, min_gap) 在远距点（tol 达 base×4=8cm）把有效切层
    # 阈值抬到 16cm，5cm 硬约束失效
    tol_hard = np.minimum(tol[cids], depth_gap / 2.0)

    comps = connected_components_uv_depth_grid(
        uv, depths, grid_size=uv_grid,
        depth_bin=depth_bin,
        min_cells=max(2, int(min_count * .01)),
        close_radius_cells=close_radius,
        depth_gap=depth_gap,
        connectivity=8,
        tolerances=tol_hard)

    # Do not fall back to the entire candidate set: that bypasses depth-layer
    # separation and can merge parallel facade planes.
    if not comps:
        return facades, fid

    def _comp_rank(m):
        n_pts = int(m.sum())
        if n_pts <= 0:
            return (0, 0)
        sub = uv[np.asarray(m, dtype=bool)]
        cells = np.unique(np.floor(sub / uv_grid).astype(np.int64), axis=0)
        return (int(len(cells)), n_pts)

    for mask in sorted(comps, key=_comp_rank, reverse=True):
        comp = cids[mask]
        if len(comp) < min_count:
            continue
        model = fit_plane_weighted(points[comp], irls_iters=irls_iters)
        dist = np.abs(points[comp] @ model[:3] + model[3])
        keep = dist <= tol[comp] * 1.8
        comp = comp[keep]
        if len(comp) < min_count:
            continue
        info = _info(fid, model, points[comp], comp, ranges[comp])
        if info['type'] == 'vertical_facade' and info['area'] >= min_facade_area:
            facades.append(info)
            fid += 1

    return facades, fid


def _deduplicate_facades(facades, points, iou_thresh=0.5,
                         depth_diff_thresh=None, angle_deg=5.0):
    """按 inlier 集合 IoU + 法向/深度差异去重。保留大的。

    【v3】depth_diff_thresh 默认收紧为 DEDUP_DEPTH_THR_M（<5cm）：
    两个真实平行立面（如窗台内凹 6cm）若 IoU 高，旧值 0.08 会把它们
    当重复面删除其一，违反深度硬隔离要求。
    """
    if not facades:
        return facades
    if depth_diff_thresh is None:
        depth_diff_thresh = DEDUP_DEPTH_THR_M

    facades.sort(key=lambda f: -f['point_count'])
    kept = []
    kept_masks = []
    n = len(points)

    for f in facades:
        mask = np.zeros(n, bool)
        inliers = np.asarray(f['inlier_indices'], dtype=int)
        mask[inliers] = True

        fn = np.asarray(f['normal'])
        fm = np.asarray(f['plane_model'])
        fd = float(fm[3]) / (np.linalg.norm(fm[:3]) + 1e-12)

        duplicate = False
        for i, km in enumerate(kept_masks):
            inter = np.count_nonzero(mask & km)
            if inter == 0:
                continue
            union = np.count_nonzero(mask | km)
            iou = inter / union if union > 0 else 0.0

            kn = np.asarray(kept[i]['normal'])
            if abs(fn @ kn) < np.cos(np.deg2rad(angle_deg)):
                continue

            km_model = np.asarray(kept[i]['plane_model'])
            kd = float(km_model[3]) / (np.linalg.norm(km_model[:3]) + 1e-12)
            depth_diff = abs(fd - kd)

            if iou > iou_thresh and depth_diff < depth_diff_thresh:
                duplicate = True
                break

        if not duplicate:
            kept.append(f)
            kept_masks.append(mask)

    return kept


def _extract_bucket_to_facades(bucket_seeds, points, normals, ranges, tol, cos_tol,
                               spacing, voxel_size, base, min_count, min_facade_area,
                               irls_iters, signed_dist_tolerance,
                               density_field=None):
    """非竞争提取：每个 bucket 种子独立扩展，最后全局去重。

    【v3】密度门控：种子 spatial_ids 内 skip 占比过半的点集直接跳过
    （过稀疏区域不产生立面，也防止 SVD 拟合出伪平面）。
    """
    all_facades = []
    fid = 0
    seed_min_pts = max(50, min_count // 10)

    for seed in bucket_seeds:
        spatial_ids = np.asarray(seed.get('spatial_ids', []), dtype=int)
        if len(spatial_ids) < seed_min_pts:
            continue

        # 【v3】密度门控：过于稀疏的种子区域直接跳过
        if density_field is not None and _HAS_DENSITY:
            if density_field.gate_indices(spatial_ids) == 'skip':
                trace('facade.bucket.skip_sparse_seed',
                      n=int(len(spatial_ids)),
                      seed_rho=round(float(seed.get('rho', 0)), 3))
                continue

        # 构造归一化 seed plane
        if 'plane_model' in seed:
            pm = np.asarray(seed['plane_model'], float)
            sn = pm[:3] / (np.linalg.norm(pm[:3]) + 1e-12)
            d = float(pm[3]) / (np.linalg.norm(pm[:3]) + 1e-12)
            seed_plane = np.r_[sn, d]
        else:
            sn = np.asarray(seed['direction'], float)
            rho = float(seed['rho'])
            seed_plane = np.r_[sn, -rho]

        facades, fid = _expand_single_seed(
            seed_plane, spatial_ids, points, normals, ranges,
            tol, cos_tol, spacing, voxel_size, base,
            max(min_count // 4, 50), min_facade_area, irls_iters,
            signed_dist_tolerance, fid_start=fid)
        all_facades.extend(facades)

    if len(all_facades) > 1:
        all_facades = _deduplicate_facades(all_facades, points)

    return all_facades


# ==================== E. 同层合并 ====================

def _merge_same_layer(facades, points, normals, d_thresh=0.03, angle_deg=5.0,
                      max_plane_dist=0.05, layer_hard_m=None):
    """同方向 + 同深度层 + 仅合并"大间隙"段（被门窗孔洞打断的同一面）。

    只对 gap ∈ [min_merge_gap, merge_gap_uv] 的配对尝试合并：
      - gap 过小（贴合）：说明并不被门窗打断，或已由扩展阶段归并；
      - gap 过大：本就属于两个独立立面，不合并。

    【v3】深度合并容差 = DEPTH_HARD_M（原 layer_hard_m + d_thresh = 0.08，
    会桥接深度差 8cm 的退台立面）。传入 layer_hard_m 参数仅作兼容，
    内部一律以 DEPTH_HARD_M 为准。
    """
    if not facades:
        return facades

    min_opening = float(getattr(Config, 'FACADE_MIN_OPENING_M', 0.5))
    merge_gap_uv = float(getattr(Config, 'FACADE_MERGE_SAME_LAYER_GAP_M', 1.0))
    min_merge_gap = float(getattr(
        Config, 'FACADE_MERGE_SAME_LAYER_MIN_GAP_M',
        max(min_opening * 0.4, 0.20)))

    # 【v3】硬约束：两段平面互嵌深度差上限 = 5cm，不加 d_thresh 裕量
    layer_hard_eff = DEPTH_HARD_M

    changed = True
    iter_count = 0
    max_iter = len(facades)

    while changed and iter_count < max_iter:
        changed = False
        iter_count += 1
        facades.sort(key=lambda f: (-f.get('area', 0), -f['point_count']))

        for i in range(len(facades)):
            if i >= len(facades):
                break
            a = facades[i]
            na = np.asarray(a['normal'])
            pa = np.asarray(a['plane_model'])
            ia = np.asarray(a['inlier_indices'], dtype=int)
            layer_a = a.get('layer', -1)
            dir_bin_a = a.get('dir_bin', -1)

            for j in range(i + 1, len(facades)):
                if j >= len(facades):
                    break
                b = facades[j]
                nb = np.asarray(b['normal'])

                if a['type'] != b['type'] or abs(na @ nb) < np.cos(np.deg2rad(angle_deg)):
                    continue

                dir_bin_b = b.get('dir_bin', -1)
                if dir_bin_a >= 0 and dir_bin_b >= 0 and dir_bin_a != dir_bin_b:
                    if abs(na @ nb) < np.cos(np.deg2rad(2.0)):
                        continue

                pb = np.asarray(b['plane_model'])
                ca, cb = np.asarray(a['center']), np.asarray(b['center'])
                off_a = abs(cb @ pa[:3] + pa[3]) / (np.linalg.norm(pa[:3]) + 1e-12)
                off_b = abs(ca @ pb[:3] + pb[3]) / (np.linalg.norm(pb[:3]) + 1e-12)
                if max(off_a, off_b) > layer_hard_eff:
                    continue

                ib = np.asarray(b['inlier_indices'], dtype=int)
                if _aabb_gap_m(points[ia], points[ib]) > merge_gap_uv * 2.0:
                    continue

                spacing_a = max(0.05, float(np.percentile(
                    np.linalg.norm(points[ia][1:] - points[ia][:-1], axis=1), 50))) if len(ia) > 1 else 0.05
                grid = max(0.10, spacing_a * 2.0)
                common_n = na + nb
                common_n /= np.linalg.norm(common_n) + 1e-12
                cells_a, box_a = _uv_descriptor(a, points, grid, common_n)
                cells_b, box_b = _uv_descriptor(b, points, grid, common_n)
                gap_u = max(box_a[0] - box_b[1], box_b[0] - box_a[1], 0.0)
                gap_v = max(box_a[2] - box_b[3], box_b[2] - box_a[3], 0.0)
                gap_uv_val = float(np.hypot(gap_u, gap_v))
                # 仅合并"大间隙"段：门/窗/孔洞造成的断裂。
                if gap_uv_val < min_merge_gap or gap_uv_val > merge_gap_uv:
                    continue

                merged_pts = points[np.unique(np.r_[ia, ib])]
                avg_d = (off_a + off_b) / 2.0
                test_model = np.array([na[0], na[1], na[2], avg_d])
                if not _check_depth_single_peak(merged_pts, test_model, max_plane_dist):
                    continue

                ids = np.unique(np.r_[ia, ib])
                model = fit_plane_weighted(points[ids], irls_iters=2)
                merged = _info(a['id'], model, points[ids], ids, np.zeros(len(ids)))
                if _fill_ratio(merged) < min(_fill_ratio(a), _fill_ratio(b)) - 0.08:
                    continue

                merged['dir_bin'] = dir_bin_a
                merged['layer'] = layer_a
                merged['attachments'] = []
                for src in (a, b):
                    merged['attachments'].extend(src.get('attachments', []))

                facades[i] = merged
                del facades[j]
                changed = True
                break
            if changed:
                break

    return facades


# ==================== F. 跨层裁决 ====================

def _cross_layer_arbitration(facades, points, d_thresh=0.03, angle_deg=5.0,
                             layer_hard_m=None, max_plane_dist=0.05):
    """同方向但不同深度层的硬隔离与条件合并。

    【v3】hard_thr 固定为 DEPTH_HARD_M（原 layer_hard_m + 2*d_thresh =
    0.11，会把深度差 5~11cm 的退台/窗台立面合并）。Δρ > 5cm 的配对
    无条件记为 setback_parallel，绝不进入合并分支；Δρ ≤ 5cm 且单峰 +
    AABB 邻近才允许合并。
    """
    if not facades:
        return facades, []

    adjacency = []
    n = len(facades)
    merged_into = list(range(n))

    groups = {}
    for idx, f in enumerate(facades):
        na = np.asarray(f['normal'])
        az = np.degrees(np.arctan2(na[1], na[0]))
        az_key = int(round(az)) % 180
        groups.setdefault(az_key, []).append(idx)

    for gidxs in groups.values():
        if len(gidxs) < 2:
            continue
        rhos = []
        for idx in gidxs:
            pm = np.asarray(facades[idx]['plane_model'])
            n_vec = pm[:3] / (np.linalg.norm(pm[:3]) + 1e-12)
            rho_c = float(np.median(points[np.asarray(facades[idx]['inlier_indices'], int)] @ n_vec))
            rhos.append((rho_c, idx))
        rhos.sort(key=lambda x: x[0])

        for k in range(len(rhos) - 1):
            rho_a, idx_a = rhos[k]
            rho_b, idx_b = rhos[k + 1]
            root_a = merged_into[idx_a]
            while root_a != merged_into[root_a]:
                root_a = merged_into[root_a]
            root_b = merged_into[idx_b]
            while root_b != merged_into[root_b]:
                root_b = merged_into[root_b]
            if root_a == root_b:
                continue

            delta = abs(rho_b - rho_a)
            # 【v3】硬约束 5cm：取消 +2*d_thresh 加成
            hard_thr = DEPTH_HARD_M

            fa = facades[root_a]
            fb = facades[root_b]
            na = np.asarray(fa['normal'])
            nb = np.asarray(fb['normal'])
            if abs(na @ nb) < np.cos(np.deg2rad(angle_deg)):
                continue

            ia = np.asarray(fa['inlier_indices'], dtype=int)
            ib = np.asarray(fb['inlier_indices'], dtype=int)

            if delta > hard_thr:
                # 【硬约束】Δρ > 5cm：无条件硬隔离，绝不合并
                adjacency.append({
                    'facade_a': fa['id'],
                    'facade_b': fb['id'],
                    'depth_delta_m': round(delta, 3),
                    'relation': 'setback_parallel',
                    'aabb_gap_m': round(_aabb_gap_m(points[ia], points[ib]), 3),
                })
            else:
                if _aabb_gap_m(points[ia], points[ib]) > 1.5:
                    continue
                merged_pts = points[np.unique(np.r_[ia, ib])]
                avg_d = (rho_a + rho_b) / 2.0
                test_model = np.array([na[0], na[1], na[2], avg_d])
                if not _check_depth_single_peak(merged_pts, test_model, max_plane_dist):
                    adjacency.append({
                        'facade_a': fa['id'],
                        'facade_b': fb['id'],
                        'depth_delta_m': round(delta, 3),
                        'relation': 'adjacent_shallow_step',
                        'aabb_gap_m': round(_aabb_gap_m(points[ia], points[ib]), 3),
                    })
                    continue

                ids = np.unique(np.r_[ia, ib])
                model = fit_plane_weighted(points[ids], irls_iters=2)
                merged = _info(fa['id'], model, points[ids], ids, np.zeros(len(ids)))
                merged['dir_bin'] = fa.get('dir_bin', -1)
                merged['layer'] = min(fa.get('layer', -1), fb.get('layer', -1))
                merged['attachments'] = fa.get('attachments', []) + fb.get('attachments', [])
                facades[root_a] = merged
                merged_into[root_b] = root_a

    final = []
    seen_root = set()
    for i in range(n):
        root = i
        while root != merged_into[root]:
            root = merged_into[root]
        if root not in seen_root:
            seen_root.add(root)
            final.append(facades[root])

    return final, adjacency


# ==================== F2. 深度硬隔离终审（v3 新增）====================

def _enforce_depth_hard_split(facades, points, angle_deg=5.0,
                              hard_m=DEPTH_HARD_M):
    """终审防御：同方向候选按 ρ 排序，凡相邻深度差 > hard_m 者不得共享
    inlier 点。若共享则说明上游某路径违规合并，按多数派归属切回并告警。

    返回 (facades, n_violations)。
    """
    if len(facades) < 2:
        return facades, 0

    groups = {}
    for i, f in enumerate(facades):
        na = np.asarray(f['normal'])
        key = int(round(np.degrees(np.arctan2(na[1], na[0])))) % 180
        groups.setdefault(key, []).append(i)

    n_violations = 0
    cos_lim = np.cos(np.deg2rad(angle_deg))
    for gidxs in groups.values():
        if len(gidxs) < 2:
            continue
        entries = []
        for i in gidxs:
            pm = np.asarray(facades[i]['plane_model'])
            nv = pm[:3] / (np.linalg.norm(pm[:3]) + 1e-12)
            ids = np.asarray(facades[i]['inlier_indices'], dtype=int)
            rho = float(np.median(points[ids] @ nv)) if len(ids) else 0.0
            entries.append((rho, i, nv))
        entries.sort(key=lambda x: x[0])

        for k in range(len(entries) - 1):
            rho_a, ia, na = entries[k]
            rho_b, ib, nb = entries[k + 1]
            delta = rho_b - rho_a
            if delta <= hard_m or abs(na @ nb) < cos_lim:
                continue
            ids_a = np.asarray(facades[ia]['inlier_indices'], dtype=int)
            ids_b = np.asarray(facades[ib]['inlier_indices'], dtype=int)
            shared = np.intersect1d(ids_a, ids_b)
            if len(shared) == 0:
                continue
            n_violations += 1
            trace('facade.depth_hard_split.violation',
                  facade_a=facades[ia]['id'], facade_b=facades[ib]['id'],
                  depth_delta_m=round(delta, 4), n_shared=int(len(shared)))
            pm_a = np.asarray(facades[ia]['plane_model'])
            pm_b = np.asarray(facades[ib]['plane_model'])
            na_u = pm_a[:3] / (np.linalg.norm(pm_a[:3]) + 1e-12)
            nb_u = pm_b[:3] / (np.linalg.norm(pm_b[:3]) + 1e-12)
            rho_shared_a = points[shared] @ na_u
            rho_shared_b = points[shared] @ nb_u
            to_a = shared[np.abs(rho_shared_a - rho_a)
                          <= np.abs(rho_shared_b - rho_b)]
            to_b = np.setdiff1d(shared, to_a)
            keep_a = np.setdiff1d(ids_a, to_b)
            keep_b = np.setdiff1d(ids_b, to_a)
            facades[ia]['inlier_indices'] = keep_a.astype(int).tolist()
            facades[ib]['inlier_indices'] = keep_b.astype(int).tolist()
            facades[ia]['point_count'] = int(len(keep_a))
            facades[ib]['point_count'] = int(len(keep_b))

    if n_violations:
        trace('facade.depth_hard_split.done', violations=n_violations)
    return facades, n_violations


# ==================== G. 水平面挂靠 — 已彻底删除 ====================
# v4：_attach_horizontal_patches 函数本体已删除（此前保留兼容旧调用）。
# 水平片（窗台/雨棚/檐口）注入竖立面会污染 depth/平整度/面积统计，
# 并干扰 5cm 深度硬隔离的 ρ 中位数估计，新流程任何路径不得启用。


# ==================== H. 完形回收（简化版）====================

def _postprocess_v2(facades, points, normals, tol, cos_tol, spacing, voxel_size,
                    min_opening=.5):
    """完形回收简化版：保留 UV/AABB/连通域/深度段/fill_ratio 门控。

    【v3】回收阶段深度段 min_gap 收紧到 DEPTH_HARD_M（原 0.12，会把
    深度差 12cm 内的退台点吸附回主立面，违反 5cm 硬约束）。
    """
    if not facades:
        return facades

    assigned = np.zeros(len(points), dtype=bool)
    for f in facades:
        assigned[np.asarray(f['inlier_indices'], dtype=int)] = True

    reclaim_mult = float(getattr(Config, 'FACADE_RECLAIM_TOL_MULT', 1.2))

    for f in facades:
        ids = np.asarray(f['inlier_indices'], dtype=int)
        if not len(ids):
            continue
        n = np.asarray(f['normal'], dtype=float)

        spacing_m = max(0.05, float(np.median(spacing[ids])))
        grid = max(0.10, spacing_m * 2.0)
        cells = _uv_cells_for_facade(f, points, grid)
        dilated = _dilate_cells(cells, min(2, max(1, int(round(0.20 / grid)))))

        cand = np.where(~assigned)[0]
        d = np.abs(points[cand] @ n + f['plane_model'][3])
        ok = (d <= tol[cand] * reclaim_mult) & (np.abs(normals[cand] @ n) >= cos_tol[cand])
        add = cand[ok]

        if len(add):
            lo = points[ids].min(0) - max(1.5 * np.median(spacing[ids]), .2)
            hi = points[ids].max(0) + max(1.5 * np.median(spacing[ids]), .2)
            add = add[np.all((points[add] >= lo) & (points[add] <= hi), axis=1)]

            if len(add):
                pm = np.asarray(f['plane_model'], dtype=float)
                typ, _, _, _ = classify_plane(n)
                uv_add, _, _, _ = project_to_uv(points[add], pm, typ)
                cells_add = list(zip(np.floor(uv_add[:, 0] / grid).astype(np.int64),
                                     np.floor(uv_add[:, 1] / grid).astype(np.int64)))
                add = add[np.asarray([c in dilated for c in cells_add], dtype=bool)]

            if len(add) >= 10:
                test_pts = np.vstack([points[ids], points[add]])
                test_comp = connected_components_3d_grid(
                    test_pts, grid_size=max(0.3, voxel_size * 6), min_points=10)
                if len(test_comp) > 1:
                    facade_label = None
                    for ci, mask in enumerate(test_comp):
                        if np.any(mask[:len(ids)]):
                            facade_label = ci
                            break
                    if facade_label is not None:
                        add_mask = test_comp[facade_label][len(ids):]
                        add = add[add_mask]
                    else:
                        add = np.array([], dtype=int)

        if len(add):
            base_signed = points[ids] @ n + f['plane_model'][3]
            add_signed = points[add] @ n + f['plane_model'][3]
            # 【v4】tol 截断到 gap/2，保证有效切层阈值 = 5cm
            tol_seg = min(max(float(np.median(tol[add])), 0.01),
                          DEPTH_HARD_M / 2.0)
            seg_labels, seg_medians = cluster_depth_significant(
                np.r_[base_signed, add_signed], tol_seg,
                # 【v3】回收深度段间距 = 5cm 硬约束
                min_gap=DEPTH_HARD_M,
                min_points=max(20, len(ids) // 50))
            if len(seg_medians) > 1:
                main_seg = int(np.argmin(np.abs(seg_medians)))
                add = add[seg_labels[len(ids):] == main_seg]

        max_add = max(len(ids) // 3, 5000)
        if len(add) > max_add:
            add = add[np.argsort(np.abs(points[add] @ n + f['plane_model'][3]))[:max_add]]

        if len(add) >= max(10, len(ids) // 50):
            all_ids = np.unique(np.r_[ids, add])
            new_model = fit_plane_weighted(points[all_ids], irls_iters=2)
            new_n = new_model[:3] / (np.linalg.norm(new_model[:3]) + 1e-12)
            if abs(new_n @ n) >= np.cos(np.deg2rad(5.0)):
                new_info = _info(f['id'], new_model, points[all_ids], all_ids,
                                 np.zeros(len(all_ids)))
                slack = float(getattr(Config, 'FACADE_RECLAIM_FILL_SLACK', 0.05))
                if _fill_ratio(new_info) >= _fill_ratio(f) - slack:
                    f.update(new_info)
                    assigned[add] = True

    for f in facades:
        fill = _fill_ratio(f)
        f['fill_ratio'] = round(float(fill), 3)
        f['wall_kind'] = 'fragment' if fill < .25 else ('fenestrated' if fill < .55 else 'solid')

    return facades


# ==================== 旧版兼容：Hough 种子路径 ====================

def _build_spatial_hough_seeds(points, normals, mask, ranges, tol, base,
                               min_count, voxel_size):
    """阶段 1：空间化 Hough 投票（保留旧路径做 A/B 回退）。"""
    valid = np.flatnonzero(mask)
    if len(valid) < min_count:
        return []

    max_vote = int(getattr(Config, 'HOUGH_MAX_VOTE_POINTS', 100000))
    hough_valid = valid[::max(1, len(valid) // max_vote)] if len(valid) > max_vote else valid

    peaks = normal_hough_peaks(
        normals[hough_valid],
        min_support=max(80, int(len(hough_valid) * 0.002)),
        bin_deg=float(getattr(Config, 'HOUGH_NORMAL_BIN_DEG', 2.0)),
        merge_deg=float(getattr(Config, 'HOUGH_NORMAL_SUPPORT_DEG', 6.0)) - 2.0,
        max_peaks=int(getattr(Config, 'HOUGH_MAX_DIRECTION_PEAKS', 8)),
        vertical_nz=float(getattr(Config, 'VERTICAL_NZ_THR', 0.30)) + 0.05)

    hough_seeds = []
    support_deg = float(getattr(Config, 'HOUGH_NORMAL_SUPPORT_DEG', 6.0))
    cos_support = np.cos(np.deg2rad(support_deg))

    density_adaptive = getattr(Config, 'HOUGH_DENSITY_ADAPTIVE', True)
    low_density_thresh = float(getattr(Config, 'HOUGH_LOW_DENSITY_THRESH', 5.0))
    low_density_bin_mult = float(getattr(Config, 'HOUGH_LOW_DENSITY_BIN_MULT', 2.0))
    low_density_min_support_ratio = float(getattr(Config, 'HOUGH_LOW_DENSITY_MIN_SUPPORT_RATIO', 0.2))
    low_density_min_peak_dist = float(getattr(Config, 'HOUGH_LOW_DENSITY_MIN_PEAK_DIST_M', 0.10))
    low_density_prominence_ratio = float(getattr(Config, 'HOUGH_LOW_DENSITY_PROMINENCE_RATIO', 0.10))

    for peak in peaks:
        direction = np.asarray(peak['normal'], dtype=float)
        direction_mask = np.abs(normals[valid] @ direction) >= cos_support
        direction_ids = valid[direction_mask]
        if len(direction_ids) < min_count:
            continue

        coarse_grid = max(2.0, voxel_size * 40)
        spatial_comps = connected_components_3d_grid(
            points[direction_ids],
            grid_size=coarse_grid,
            min_points=max(min_count // 2, 50))

        for sp_mask in spatial_comps:
            sp_ids = direction_ids[sp_mask]
            if len(sp_ids) < min_count // 2:
                continue

            if density_adaptive:
                block_pts = points[sp_ids]
                bbox_vol = np.ptp(block_pts, axis=0).prod() if len(block_pts) >= 3 else 0.0
                density = len(sp_ids) / max(bbox_vol, 1e-6)
                is_low_density = density < low_density_thresh
                if is_low_density:
                    adaptive_bin = max(base * low_density_bin_mult, 0.05)
                    min_support = max(20, int(min_count * low_density_min_support_ratio))
                    min_peak_dist = low_density_min_peak_dist
                    prominence_ratio = low_density_prominence_ratio
                else:
                    adaptive_bin = max(float(voxel_size), float(np.median(tol[sp_ids])) * 0.8)
                    min_support = max(50, min_count // 3)
                    min_peak_dist = max(0.30, base * 3)
                    prominence_ratio = 0.08
            else:
                adaptive_bin = max(float(voxel_size), float(np.median(tol[sp_ids])) * 0.8)
                min_support = max(50, min_count // 3)
                min_peak_dist = max(0.30, base * 3)
                prominence_ratio = 0.08

            signed = points[sp_ids] @ direction
            depth_seg_labels, depth_seg_medians = cluster_depth_significant(
                signed, tolerance=None,
                min_gap=max(base * 1.5, 0.03),
                min_points=max(50, len(sp_ids) // 20))

            for seg_idx in range(len(depth_seg_medians)):
                seg_mask = depth_seg_labels == seg_idx
                seg_ids = sp_ids[seg_mask]
                if len(seg_ids) < max(min_support, min_count // 4):
                    continue

                if density_adaptive and (not is_low_density or density >= low_density_thresh):
                    seg_peak_dist = max(0.08, base * 1.5)
                else:
                    seg_peak_dist = max(0.15, base * 2)

                seg_rho_peaks = rho_hough_peaks(
                    points[seg_ids], direction,
                    bin_size=adaptive_bin,
                    min_support=max(min_support // 2, 20),
                    min_peak_distance=seg_peak_dist,
                    prominence_ratio=prominence_ratio,
                    max_peaks=int(getattr(Config, 'HOUGH_MAX_RHO_PEAKS_PER_DIRECTION', 24)))

                block_key = (tuple(np.round(direction, 3)),
                             int(np.floor(float(np.median(points[seg_ids] @ direction)) /
                                          max(adaptive_bin, 1e-6))),
                             seg_idx)
                for rho_peak in seg_rho_peaks:
                    hough_seeds.append({
                        'direction': direction,
                        'rho': rho_peak.value,
                        'spatial_ids': seg_ids,
                        'support': rho_peak.support,
                        'block': block_key,
                        'bin': float(adaptive_bin),
                        'depth_seg': seg_idx,
                    })

    if not hough_seeds:
        return []
    max_candidates = int(getattr(Config, 'HOUGH_MAX_CANDIDATES', 16))
    per_block = max(1, int(getattr(Config, 'HOUGH_MAX_RHO_PEAKS_PER_DIRECTION', 4)))

    groups = {}
    for s in hough_seeds:
        groups.setdefault(s['block'], []).append(s)
    quota = []
    for members in groups.values():
        members.sort(key=lambda s: -s['support'])
        quota.extend(members[:per_block])

    quota.sort(key=lambda s: -s['support'])
    kept = []
    for s in quota:
        duplicate = False
        for t in kept:
            if t['block'] != s['block']:
                continue
            gap = abs(float(s['rho']) - float(t['rho']))
            if gap > max(4.0 * float(s['bin']), 0.06):
                continue
            smaller = min(float(s['support']), float(t['support']))
            larger = max(float(s['support']), float(t['support']))
            if larger > 0 and smaller / larger >= float(
                    getattr(Config, 'HOUGH_PEAK_MERGE_VALLEY_RATIO', 0.30)):
                continue
            duplicate = True
            break
        if not duplicate:
            kept.append(s)

    kept.sort(key=lambda s: -s['support'])
    selected = list(kept[:max_candidates])
    selected_ids = {id(s) for s in selected}
    weak_floor = int(getattr(Config, 'HOUGH_MAX_WEAK_SEEDS', 16))
    weak_min_support = max(20, int(min_count *
                                   float(getattr(Config, 'HOUGH_WEAK_SUPPORT_RATIO', 0.20))))
    if weak_floor > 0:
        rest = [s for s in kept if id(s) not in selected_ids]
        rest.sort(key=lambda s: -s['support'])
        added = 0
        for s in rest:
            if added >= weak_floor or s['support'] < weak_min_support:
                break
            isolated = True
            for t in kept:
                if t is s or t['block'] != s['block']:
                    continue
                if abs(float(t['rho']) - float(s['rho'])) < 1.5:
                    isolated = False
                    break
            if isolated:
                selected.append(s)
                added += 1
    return selected


def _extract_facades_from_seeds(seed_iter, points, normals, remaining, ranges,
                                tol, cos_tol, spacing, voxel_size, base,
                                min_count, min_facade_area, irls_iters,
                                signed_dist_tolerance):
    """核心提取循环（Hough/RANSAC 种子 → 立面）。保留旧路径兼容。

    【v3】depth_bin/depth_gap 同样收口到 DEPTH_HARD_M。
    """
    facades = []
    fid = 0
    consumed_global = set()

    for seed_kind, seed_info in seed_iter:
        ids = np.flatnonzero(remaining)
        if len(ids) < min_count:
            break

        if seed_kind == 'hough':
            sn = np.asarray(seed_info['direction'], dtype=float)
            rho = float(seed_info['rho'])
            spatial_ids = seed_info.get('spatial_ids', ids)

            seed = np.r_[sn, -rho]
            signed_sp = points[spatial_ids] @ sn - rho
            near_mask = np.abs(signed_sp) <= max(base * 3, 0.15)
            local_ids = spatial_ids[near_mask]

            if (getattr(Config, 'FACADE_RANSAC_DEPTH_ISOLATE', True)
                    and len(local_ids) >= 200):
                seg_labels, seg_medians = cluster_depth_significant(
                    signed_sp[near_mask], tol[local_ids],
                    min_gap=float(getattr(Config, 'FACADE_RANSAC_DEPTH_MIN_GAP_M', 0.04)),
                    min_points=max(50, len(local_ids) // 20))
                if len(seg_medians) > 1:
                    keep_seg = int(np.argmin(np.abs(seg_medians)))
                    seg_sel = seg_labels == keep_seg
                    if np.count_nonzero(seg_sel) >= max(150, min_count // 3):
                        local_ids = local_ids[seg_sel]

            if len(local_ids) >= 200:
                sub_pcd = o3d.geometry.PointCloud()
                sub_pcd.points = o3d.utility.Vector3dVector(points[local_ids])
                local_model, local_inl = sub_pcd.segment_plane(
                    distance_threshold=base * 0.6,
                    ransac_n=3,
                    num_iterations=500)
                if len(local_inl) >= max(150, min_count // 3):
                    seed = np.asarray(local_model, float)
                    sn = seed[:3] / (np.linalg.norm(seed[:3]) + 1e-12)

            effective_ids = np.intersect1d(ids, spatial_ids, assume_unique=True)
            if len(effective_ids) < min_count:
                continue
            ids = effective_ids

        else:  # ransac fallback
            ransac_ids = ids
            max_ransac = int(getattr(Config, 'HOUGH_MAX_VOTE_POINTS', 100000))
            if len(ransac_ids) > max_ransac:
                stride = max(1, len(ransac_ids) // max_ransac)
                ransac_ids = ransac_ids[::stride]
            sub = o3d.geometry.PointCloud()
            sub.points = o3d.utility.Vector3dVector(points[ransac_ids])
            seed_model, seed_local = sub.segment_plane(
                float(np.median(tol[ransac_ids])), 3, 500)
            if len(seed_local) < max(100, min_count // 10):
                break
            seed = np.asarray(seed_model, float)
            sn = seed[:3] / (np.linalg.norm(seed[:3]) + 1e-12)

        signed = points[ids] @ sn + seed[3]
        candidate = np.abs(signed) <= tol[ids] * 1.5
        candidate &= np.abs(normals[ids] @ sn) >= cos_tol[ids]
        cids = ids[candidate]
        if len(cids) < min_count:
            if seed_kind == 'hough':
                continue
            break

        tight = np.abs(points[cids] @ sn + seed[3]) <= tol[cids] * signed_dist_tolerance
        cids = cids[tight]
        if len(cids) < min_count:
            if seed_kind == 'hough':
                continue
            break

        ftype, _, _, _ = classify_plane(sn)
        uv, _, _, _ = project_to_uv(points[cids], seed, ftype)

        local_spacing = float(np.median(spacing[cids])) if len(cids) else voxel_size
        uv_grid = max(float(voxel_size) * 2.0, local_spacing * 1.5, 0.05)

        is_major = (seed_kind == 'hough' and
                    seed_info.get('support', 0) > min_count * 2)
        min_opening_m = float(getattr(Config, 'FACADE_MIN_OPENING_M', 0.5))
        big_radius = max(3, int(np.ceil(min_opening_m * 2.5 / uv_grid)))
        small_radius = max(2, int(np.ceil(min_opening_m / uv_grid)))

        if getattr(Config, 'FACADE_CLOSE_RADIUS_BY_SPAN', True) and len(uv):
            span = max(float(np.ptp(uv[:, 0])), float(np.ptp(uv[:, 1])))
            close_radius = big_radius if span >= min_opening_m * 4.0 else small_radius
        else:
            close_radius = big_radius if is_major else small_radius
        close_radius = min(close_radius,
                           int(getattr(Config, 'FACADE_UV_CLOSE_MAX_CELLS', 12)))

        seed_norm = np.linalg.norm(seed[:3]) + 1e-12
        n_seed = seed[:3] / seed_norm
        d_seed = float(seed[3]) / seed_norm
        depths = points[cids] @ n_seed + d_seed

        # 【v3/v4】深度方向桥接 = 0；tol 截断到 gap/2（同 _expand_single_seed）
        depth_bin = DEPTH_HARD_M
        depth_gap = DEPTH_HARD_M
        tol_hard = np.minimum(tol[cids], depth_gap / 2.0)

        comps = connected_components_uv_depth_grid(
            uv, depths, grid_size=uv_grid,
            depth_bin=depth_bin,
            min_cells=max(2, int(min_count * .01)),
            close_radius_cells=close_radius,
            depth_gap=depth_gap,
            connectivity=8,
            tolerances=tol_hard)
        # 不能以整组点作为兜底：该路径会绕过深度层隔离，把多个退台面
        # 重新混成一个候选。无法证明单层时宁可交给其它种子召回。
        if not comps:
            continue

        def _comp_rank(m):
            n_pts = int(m.sum())
            if n_pts <= 0:
                return (0, 0)
            sub = uv[np.asarray(m, dtype=bool)]
            cells = np.unique(np.floor(sub / uv_grid).astype(np.int64), axis=0)
            return (int(len(cells)), n_pts)

        consumed = set()
        for mask in sorted(comps, key=_comp_rank, reverse=True):
            comp = cids[mask]
            if len(comp) < min_count:
                continue
            model = fit_plane_weighted(points[comp], irls_iters=irls_iters)
            dist = np.abs(points[comp] @ model[:3] + model[3])
            keep = dist <= tol[comp] * 1.8
            comp = comp[keep]
            if len(comp) < min_count:
                continue
            info = _info(fid, model, points[comp], comp, ranges[comp])
            if info['type'] == 'vertical_facade' and info['area'] >= min_facade_area:
                facades.append(info)
                fid += 1
                consumed.update(comp.tolist())

        if not consumed:
            if seed_kind == 'hough':
                continue
            break
        remaining[list(consumed)] = False
        consumed_global.update(consumed)

    return facades, remaining


def _weak_boundary_recall(main_facades, points, normals, remaining, ranges,
                          tol, cos_tol, spacing, voxel_size, base,
                          min_count, min_facade_area, irls_iters,
                          signed_dist_tolerance):
    """弱边界召回（保留旧逻辑，新流程下触发量大幅减少）。"""
    if not main_facades or not getattr(Config, 'FACADE_ENABLE_WEAK_BOUNDARY_RECALL', False):
        return []

    band_m = float(getattr(Config, 'FACADE_WEAK_BOUNDARY_BAND_M', 0.50))
    weak_min_area = float(getattr(Config, 'FACADE_WEAK_MIN_AREA_M2', 8.0))
    weak_min_pts = int(getattr(Config, 'FACADE_WEAK_MIN_POINTS', 80))
    weak_min_angle = float(getattr(Config, 'FACADE_WEAK_MIN_ANGLE_DEG', 25.0))
    max_per_facade = int(getattr(Config, 'FACADE_WEAK_MAX_PER_FACADE', 4))
    min_density = float(getattr(Config, 'FACADE_WEAK_MIN_DENSITY', 50.0))
    min_compact = float(getattr(Config, 'FACADE_WEAK_MIN_COMPACT_RATIO', 0.60))
    max_components = int(getattr(Config, 'FACADE_WEAK_MAX_COMPONENTS', 3))
    extend_tol_mult = float(getattr(Config, 'FACADE_WEAK_EXTEND_TOL_MULT', 0.8))
    extend_angle_deg = float(getattr(Config, 'FACADE_WEAK_EXTEND_ANGLE_DEG', 5.0))
    corner_area_thr = float(getattr(Config, 'FACADE_WEAK_CORNER_AREA_M2', 5.0))

    recalled = []
    fid = len(main_facades)
    used = np.zeros(len(points), dtype=bool)
    for f in main_facades:
        used[np.asarray(f.get('inlier_indices', []), dtype=int)] = True

    for f in main_facades:
        if f.get('type') != 'vertical_facade':
            continue

        pm = np.asarray(f['plane_model'], dtype=float)
        n = np.asarray(f['normal'], dtype=float)
        center = np.asarray(f['center'], dtype=float)
        u, v = plane_axes(n, 'vertical_facade')

        main_ids = np.asarray(f.get('inlier_indices', []), dtype=int)
        if len(main_ids) == 0:
            continue

        local = np.column_stack([
            (points[main_ids] - center) @ u,
            (points[main_ids] - center) @ v
        ])
        u_min, u_max = float(local[:, 0].min()), float(local[:, 0].max())
        v_min, v_max = float(local[:, 1].min()), float(local[:, 1].max())

        rem_ids = np.flatnonzero(remaining & ~used)
        if len(rem_ids) < weak_min_pts:
            continue

        rem_local = np.column_stack([
            (points[rem_ids] - center) @ u,
            (points[rem_ids] - center) @ v
        ])
        in_band = (
            (rem_local[:, 0] >= u_min - band_m) & (rem_local[:, 0] <= u_max + band_m) &
            (rem_local[:, 1] >= v_min - band_m) & (rem_local[:, 1] <= v_max + band_m)
        )
        band_ids = rem_ids[in_band]
        if len(band_ids) < weak_min_pts:
            continue

        band_normals = normals[band_ids]
        cos_main = np.abs(band_normals @ n)
        cos_weak_thr = np.cos(np.deg2rad(weak_min_angle))
        weak_mask = cos_main <= cos_weak_thr
        weak_ids = band_ids[weak_mask]

        if len(weak_ids) < weak_min_pts:
            continue

        clusters = cluster_normals_direction(normals[weak_ids], angle_threshold_deg=weak_min_angle)
        candidate_list = []

        for mask in clusters:
            cands = weak_ids[mask]
            if len(cands) < weak_min_pts:
                continue

            sub_pcd = o3d.geometry.PointCloud()
            sub_pcd.points = o3d.utility.Vector3dVector(points[cands].astype(float))
            local_tol = max(float(np.median(tol[cands])) * 0.6, base * 0.3)
            local_model, local_inl = sub_pcd.segment_plane(
                distance_threshold=local_tol,
                ransac_n=3,
                num_iterations=200
            )
            if len(local_inl) < weak_min_pts:
                continue

            seed = np.asarray(local_model, float)
            sn = seed[:3] / (np.linalg.norm(seed[:3]) + 1e-12)

            band_signed = points[band_ids] @ sn + float(seed[3]) / (np.linalg.norm(seed[:3]) + 1e-12)
            band_cos = np.abs(normals[band_ids] @ sn)
            tol_band = tol[band_ids] * extend_tol_mult
            cos_band = np.cos(np.deg2rad(extend_angle_deg))
            band_ok = (
                (np.abs(band_signed) <= tol_band) &
                (band_cos >= cos_band)
            )
            ext_ids = band_ids[band_ok]

            if len(ext_ids) < weak_min_pts:
                continue

            info_temp = _info(0, seed, points[ext_ids], ext_ids, ranges[ext_ids])
            area = info_temp.get('area', 0.0)
            is_corner_candidate = False
            cos_sn_n = abs(float(sn @ n))
            if cos_sn_n < 0.3:
                ext_local = np.column_stack([
                    (points[ext_ids] - center) @ u,
                    (points[ext_ids] - center) @ v
                ])
                near_edge = (
                    (np.any(ext_local[:, 0] < u_min + 0.5) or
                     np.any(ext_local[:, 0] > u_max - 0.5)) or
                    (np.any(ext_local[:, 1] < v_min + 0.5) or
                     np.any(ext_local[:, 1] > v_max - 0.5))
                )
                if near_edge:
                    is_corner_candidate = True

            effective_min_area = corner_area_thr if is_corner_candidate else weak_min_area
            if area < effective_min_area:
                continue

            density = len(ext_ids) / area if area > 0 else 0.0
            effective_min_density = min_density * 0.5 if is_corner_candidate else min_density
            if density < effective_min_density:
                continue

            grid_size = max(voxel_size * 2, 0.1)
            comps = connected_components_3d_grid(
                points[ext_ids],
                grid_size=grid_size,
                min_points=max(weak_min_pts // 5, 5)
            )
            if not comps:
                continue
            sizes = [np.sum(c) for c in comps]
            total = len(ext_ids)
            largest_ratio = max(sizes) / total if total > 0 else 0.0
            if largest_ratio < min_compact or len(comps) > max_components:
                continue

            final_model = fit_plane_weighted(points[ext_ids], irls_iters=irls_iters)
            final_dist = np.abs(points[ext_ids] @ final_model[:3] + final_model[3])
            final_keep = final_dist <= tol[ext_ids] * 0.9
            final_ids = ext_ids[final_keep]
            if len(final_ids) < weak_min_pts:
                continue

            info = _info(fid, final_model, points[final_ids], final_ids, ranges[final_ids])
            info['weak'] = True
            candidate_list.append((info, len(final_ids), area))

        candidate_list.sort(key=lambda x: (x[2], x[1]), reverse=True)
        for info, _, _ in candidate_list[:max_per_facade]:
            recalled.append(info)
            fid += 1
            used[np.asarray(info['inlier_indices'], dtype=int)] = True

    return recalled


# ==================== 主入口 ====================

def detect_facades_adaptive(pcd, voxel_size=.05, min_facade_area=5., max_plane_dist=None,
                            min_points_ratio=.003, roi_bounds=None, roi_indices=None,
                            enable_merge=True, enable_grow=False, signed_dist_tolerance=1.,
                            range_adaptive=True, scan_origin=None, range_coeff=.0012,
                            normal_relax_deg_per_m=.15, normal_angle_max_deg=15.,
                            irls_iters=2, metadata=None, normals=None, elevations=None,
                            **_kwargs):
    """主入口：多分辨率立面检测，支持 Hough / Bucket 双种子模式切换。

    当传入 ``normals``（Nx3 ndarray，长度与 pcd 一致且全部有限）时，
    直接复用该法向量，完全跳过内部 ``ensure_normals`` 的 KD-tree 估计。
    若 ``pcd`` 本身已带法向量，也优先复用；代理点直接使用 ``pcd.points``。

    FACADE_SEED_MODE 配置：
      - 'bucket'（默认）：方向桶 × 深度层确定性枚举 + 非竞争扩展 + 全局去重。
      - 'hough'  ：保留原有空间化 Hough 投票路径（回退兼容）。
      - 'hybrid' ：先 bucket，若输出为空则 fallback 到 hough。

    【v3 新增】elevations 参数：高度角数组（度），供密度场高/低分层；
    None 时密度模块以 z 中位数近似分层。
    """
    started = time.perf_counter()
    points = np.asarray(pcd.points, float)
    n = len(points)
    if not n:
        return {'facades': [], 'remaining': pcd, 'total_points': 0}

    # 读取预存法向量：外部 normals > pcd 自带 normals > 重新估计
    work = pcd
    normals_arr = None
    normals_src = None

    if normals is not None:
        ext = np.asarray(normals, float)
        if ext.shape == (n, 3) and np.all(np.isfinite(ext)):
            lens = np.linalg.norm(ext, axis=1, keepdims=True)
            if np.all(lens > 1e-8):
                normals_arr = ext / lens
                normals_src = 'external'

    if normals_arr is None and pcd.has_normals():
        pcd_normals = np.asarray(pcd.normals, float)
        if pcd_normals.shape == (n, 3) and np.all(np.isfinite(pcd_normals)):
            lens = np.linalg.norm(pcd_normals, axis=1, keepdims=True)
            if np.all(lens > 1e-8):
                normals_arr = pcd_normals / lens
                normals_src = 'pcd_precomputed'

    if normals_arr is None:
        work = ensure_normals(pcd, voxel_size, inplace=True)
        normals_arr = np.asarray(work.normals, float)
        normals_src = 'estimated'

    # 同步写回 work，后续阶段统一通过 work.normals 读取
    work.normals = o3d.utility.Vector3dVector(normals_arr)
    trace("facade.algo.normals", points=n, source=normals_src,
          seconds=f"{time.perf_counter()-started:.2f}")

    normals = normals_arr

    meta = metadata or {}
    cfg = meta.get('adaptive_detection', {})
    method = str(cfg.get('method', getattr(Config, 'FACADE_DETECT_METHOD', 'hybrid'))).lower()
    if method not in {'ransac', 'hough', 'hybrid'}:
        method = 'hybrid'

    seed_mode = str(getattr(Config, 'FACADE_SEED_MODE', 'bucket')).lower()

    range_adaptive = bool(range_adaptive or cfg.get('enabled', False))
    if range_adaptive:
        range_coeff = float(cfg.get('range_coeff', range_coeff))
        irls_iters = int(cfg.get('irls_iters', irls_iters))
        scan_origin = scan_origin if scan_origin is not None else meta.get('scan_origins')

    ranges = estimate_point_ranges(points, scan_origin) if range_adaptive else np.zeros(n)
    base = float(max_plane_dist or max(voxel_size * 1.5,
                                       float(getattr(Config, 'DETECT_DIST_TOL_MM', 20)) / 1000.))
    tol = adaptive_plane_tolerance(ranges, base, range_coeff, base * 4.) if range_adaptive else np.full(n, base)
    theta = np.clip(float(getattr(Config, 'FACADE_NORMAL_ANGLE_DEG', 8.0)) +
                    normal_relax_deg_per_m * ranges, 8., normal_angle_max_deg)
    cos_tol = np.cos(np.deg2rad(theta))

    active = np.ones(n, bool)
    if roi_bounds is not None:
        active &= np.all((points >= roi_bounds[0]) & (points <= roi_bounds[1]), axis=1)
    if roi_indices is not None:
        m = np.zeros(n, bool)
        valid = np.asarray(roi_indices, int)
        m[valid[(valid >= 0) & (valid < n)]] = True
        active &= m

    min_count = max(80 if roi_indices is not None or roi_bounds is not None else 100,
                    min(int(n * min_points_ratio),
                        int(getattr(Config, 'FACADE_MIN_COUNT_CAP', 2000))),
                    int(getattr(Config, 'CLUSTER_MIN_POINTS', 200)))

    if n <= 100000:
        spacing = np.asarray(work.compute_nearest_neighbor_distance()) if n > 1 else np.zeros(n)
    else:
        spacing = np.full(n, max(float(voxel_size), 0.05), dtype=np.float32)

    # ===== 【v3 新增】密度场：构建一次，透传给种子/提取/区域跳过 =====
    density_field = None
    if _HAS_DENSITY:
        try:
            density_field = build_density_field(
                points, ranges,
                elevations=elevations,
                voxel_size=float(voxel_size))
            trace('facade.algo.density',
                  dense=int(np.count_nonzero(density_field.point_level == 0)),
                  sparse=int(np.count_nonzero(density_field.point_level == 1)),
                  skip=int(np.count_nonzero(density_field.point_level == 2)),
                  seconds=f"{time.perf_counter()-started:.2f}")
        except Exception:
            density_field = None
            trace('facade.algo.density', failed=True)

    remaining = active.copy()
    facades = []

    # ===== 阶段 4：粗到细双速提取 =====
    coarse_voxel = voxel_size * 4
    enable_coarse = (n > 100000 and method == 'hybrid' and
                     getattr(Config, 'FACADE_ENABLE_COARSE_LAYER', True))

    if enable_coarse:
        coarse_pcd = work.voxel_down_sample(coarse_voxel)
        coarse_points = np.asarray(coarse_pcd.points, float)
        coarse_normals = np.asarray(coarse_pcd.normals, float) if coarse_pcd.has_normals() else np.zeros_like(coarse_points)
        coarse_n = len(coarse_points)

        if coarse_n >= max(min_count // 2, 50):
            coarse_ranges = estimate_point_ranges(coarse_points, scan_origin) if range_adaptive else np.zeros(coarse_n)
            coarse_tol = adaptive_plane_tolerance(coarse_ranges, base * 2.0, range_coeff, base * 6.0)
            coarse_theta = np.clip(float(getattr(Config, 'FACADE_NORMAL_ANGLE_DEG', 8.0)) +
                                   normal_relax_deg_per_m * coarse_ranges, 8., normal_angle_max_deg)
            coarse_cos_tol = np.cos(np.deg2rad(coarse_theta))
            coarse_spacing = np.full(coarse_n, coarse_voxel, dtype=np.float32)

            # 粗层 bucket 种子 + 非竞争扩展
            if seed_mode in ('bucket', 'hybrid'):
                coarse_seeds = build_bucket_seeds(
                    coarse_points, coarse_normals, np.ones(coarse_n, bool),
                    coarse_ranges, coarse_tol, base, max(min_count // 2, 50), coarse_voxel)
                if coarse_seeds:
                    coarse_facades = _extract_bucket_to_facades(
                        coarse_seeds, coarse_points, coarse_normals, coarse_ranges,
                        coarse_tol, coarse_cos_tol, coarse_spacing, coarse_voxel,
                        base * 2.0, max(min_count // 2, 50), min_facade_area * 0.5,
                        irls_iters, signed_dist_tolerance=2.0)
                else:
                    coarse_facades = []
            else:
                coarse_seeds = _build_spatial_hough_seeds(
                    coarse_points, coarse_normals, np.ones(coarse_n, bool),
                    coarse_ranges, coarse_tol, base, max(min_count // 2, 50), coarse_voxel)
                coarse_facades, _ = _extract_facades_from_seeds(
                    [('hough', s) for s in coarse_seeds],
                    coarse_points, coarse_normals, np.ones(coarse_n, bool),
                    coarse_ranges, coarse_tol, coarse_cos_tol, coarse_spacing,
                    coarse_voxel, base * 2.0,
                    max(min_count // 2, 50), min_facade_area * 0.5,
                    irls_iters, 2.0)

            # 粗层 facade 映射回原始点云
            if coarse_facades:
                covered = np.zeros(n, dtype=bool)
                for cf in coarse_facades:
                    cn = np.asarray(cf['normal'])
                    cd = float(np.asarray(cf['plane_model'])[3])
                    on_plane = (np.abs(points @ cn + cd) <= tol * 1.5) & \
                               (np.abs(normals @ cn) >= np.cos(np.deg2rad(5.0)))
                    covered |= on_plane

                remaining &= ~covered
                for cf in coarse_facades:
                    cn = np.asarray(cf['normal'])
                    cd = float(np.asarray(cf['plane_model'])[3])
                    all_signed = points @ cn + cd
                    all_near = np.abs(all_signed) <= tol * 0.9
                    all_near &= np.abs(normals @ cn) >= cos_tol
                    if all_near.sum() >= min_count:
                        model = fit_plane_weighted(points[all_near], irls_iters=irls_iters)
                        facades.append(_info(len(facades), model, points[all_near],
                                            np.flatnonzero(all_near), ranges[all_near]))

                for f in facades:
                    remaining[np.asarray(f['inlier_indices'], int)] = False

            trace("facade.algo.coarse", coarse_points=coarse_n,
                  facades=len(facades),
                  seconds=f"{time.perf_counter()-started:.2f}")

    # ===== 细层：Bucket 或 Hough =====
    fine_facades = []

    if seed_mode in ('bucket', 'hybrid') and remaining.sum() >= min_count:
        bucket_seeds = build_bucket_seeds(
            points, normals, remaining, ranges, tol, base, min_count, voxel_size,
            elevations=elevations, density_field=density_field)
        if bucket_seeds:
            fine_facades = _extract_bucket_to_facades(
                bucket_seeds, points, normals, ranges,
                tol, cos_tol, spacing, voxel_size, base,
                min_count, min_facade_area, irls_iters, signed_dist_tolerance,
                density_field=density_field)
            # 更新 remaining
            if fine_facades:
                used = np.zeros(n, bool)
                for f in fine_facades:
                    used[np.asarray(f['inlier_indices'], int)] = True
                remaining &= ~used

    # hybrid fallback：bucket 为空或极少时切 hough
    if seed_mode == 'hybrid' and len(fine_facades) < 2 and remaining.sum() >= min_count:
        hough_seeds = _build_spatial_hough_seeds(
            points, normals, remaining, ranges, tol, base, min_count, voxel_size)
        seed_iter = [('hough', s) for s in hough_seeds]
        seed_iter.extend([('ransac', None)] * 8)
        hough_facades, remaining = _extract_facades_from_seeds(
            seed_iter, points, normals, remaining, ranges,
            tol, cos_tol, spacing, voxel_size, base,
            min_count, min_facade_area, irls_iters, signed_dist_tolerance)
        fine_facades.extend(hough_facades)

    if seed_mode == 'hough' and remaining.sum() >= min_count:
        hough_seeds = _build_spatial_hough_seeds(
            points, normals, remaining, ranges, tol, base, min_count, voxel_size)
        seed_iter = [('hough', s) for s in hough_seeds]
        if method == 'ransac' or not seed_iter:
            seed_iter = [('ransac', None)] * 30
        elif method == 'hybrid':
            seed_iter.extend([('ransac', None)] * 8)
        hough_facades, remaining = _extract_facades_from_seeds(
            seed_iter, points, normals, remaining, ranges,
            tol, cos_tol, spacing, voxel_size, base,
            min_count, min_facade_area, irls_iters, signed_dist_tolerance)
        fine_facades.extend(hough_facades)

    trace("facade.algo.seeds", seed_mode=seed_mode,
          fine_facades=len(fine_facades),
          seconds=f"{time.perf_counter()-started:.2f}")

    facades.extend(fine_facades)

    # ===== 弱边界召回 =====
    if getattr(Config, 'FACADE_ENABLE_WEAK_BOUNDARY_RECALL', False):
        weak_facades = _weak_boundary_recall(
            facades, points, normals, remaining, ranges,
            tol, cos_tol, spacing, voxel_size, base,
            min_count, min_facade_area, irls_iters, signed_dist_tolerance)
        if weak_facades:
            facades.extend(weak_facades)
            trace("facade.algo.weak_recall", recalled=len(weak_facades),
                  seconds=f"{time.perf_counter()-started:.2f}")

    # ===== E. 同层合并（仅合并大间隙段）=====
    if enable_merge:
        facades = _merge_same_layer(
            facades, points, normals,
            d_thresh=float(getattr(Config, 'MERGE_D_THRESH_M', 0.08)),
            angle_deg=float(getattr(Config, 'MERGE_ANGLE_DEG', 5.0)),
            max_plane_dist=base,
            layer_hard_m=DEPTH_HARD_M)
        trace("facade.algo.merge_same_layer", facades=len(facades),
              seconds=f"{time.perf_counter()-started:.2f}")

    # ===== F. 跨层裁决 =====
    if enable_merge:
        facades, adjacency = _cross_layer_arbitration(
            facades, points,
            d_thresh=float(getattr(Config, 'MERGE_D_THRESH_M', 0.08)),
            angle_deg=float(getattr(Config, 'MERGE_ANGLE_DEG', 5.0)),
            layer_hard_m=DEPTH_HARD_M,
            max_plane_dist=base)
        trace("facade.algo.cross_layer", facades=len(facades),
              adjacency_pairs=len(adjacency),
              seconds=f"{time.perf_counter()-started:.2f}")
        for f in facades:
            f['adjacency'] = [a for a in adjacency
                              if a['facade_a'] == f['id'] or a['facade_b'] == f['id']]

    # ===== F2. 深度硬隔离终审（v3 新增，5cm 硬性要求的防御性收口）=====
    if enable_merge:
        facades, n_viol = _enforce_depth_hard_split(facades, points)
        trace("facade.algo.depth_hard_split", facades=len(facades),
              violations=n_viol,
              seconds=f"{time.perf_counter()-started:.2f}")

    # ===== G. 水平面挂靠 — 已彻底删除（v4） =====

    # ===== H. 完形回收 =====
    facades = _postprocess_v2(facades, points, normals, tol, cos_tol, spacing, voxel_size)

    # ===== 【v3】完形回收后再次终审：回收可能重新引入跨层点 =====
    facades, n_viol2 = _enforce_depth_hard_split(facades, points)
    if n_viol2:
        trace("facade.algo.depth_hard_split.post_reclaim",
              violations=n_viol2,
              seconds=f"{time.perf_counter()-started:.2f}")

    trace("facade.algo.postprocess", facades=len(facades),
          seconds=f"{time.perf_counter()-started:.2f}")

    # 构建剩余点云
    used = np.zeros(n, bool)
    for f in facades:
        used[np.asarray(f['inlier_indices'], int)] = True
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(points[~used])

    facades.sort(key=lambda f: -f['point_count'])
    for i, f in enumerate(facades):
        f['id'] = i
    trace("facade.algo.done", facades=len(facades), total_points=n,
          seconds=f"{time.perf_counter()-started:.2f}")

    return {'facades': facades, 'remaining': out, 'total_points': n}