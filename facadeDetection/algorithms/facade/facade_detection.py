"""距离自适应立面检测 v4：5cm 深度硬隔离收口 + 密度门控 + 深度种子×Hough 校验
+ 双通道全域非竞争提取 + 同层跨洞合并（楼层尺度桥接）。


1. 【硬性约束】全局唯一深度硬隔离常量 DEPTH_HARD_M（= 5cm）。所有深度方向
   的桥接/合并/去重/回收容差一律不得超过它.
2. 【密度门控】接入 density_estimator.DensityField：桶级/层级/连通域级
   'skip' 区域直接跳过；'sparse' 区域放宽最小点数（召回稀疏窗台立面）。
3. build_bucket_seeds：深度层由「排序+间隙切割」产生（无
   箱界漂移），层内 2cm 细箱双峰切割（解决噪声带重叠漏切），边界 ε 双
   归属真正生效；ρ-Hough 校验层中心；残余方向（bin_idx=-2）补充通道。
4. precluster 与 bucket 两个种子通道都在完整 active 域上运行。
5. 合并调用收口：detect_facades_adaptive 末尾跨层裁决后执行深度硬隔离
   终审，再走完形回收。
"""
from __future__ import annotations

import time
import numpy as np
import open3d as o3d

from algorithms.geometry import (
    ensure_normals, classify_plane, project_to_uv,
    connected_components_uv_depth_grid,
    estimate_plane_area, plane_axes, cluster_depth_significant,
    estimate_point_ranges, adaptive_plane_tolerance, fit_plane_weighted
)
from config.settings import Config
from algorithms.facade.direction_depth_cluster import build_bucket_seeds
from algorithms.facade.facade_precluster import build_depth_normal_patch_seeds
from algorithms.preprocess import finite_sparse_keep_indices

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
    pm = np.asarray(plane_model, dtype=np.float64)
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
                        fid_start=0, stats=None):
    """从单个种子扩展立面。扩展逻辑与 Hough 路径完全一致。

    只在 seed_spatial_ids 范围内做初始筛选，然后 UV 连通域扩展。
    不修改任何全局状态。
    UV+Depth 连通域深度方向零容忍桥接：depth_bin = depth_gap =
    DEPTH_HARD_M（旧 depth_gap=0.12 会把深度差 12cm 的两个面闭运算桥接）。

    stats: 可选 dict，记录各拒绝原因的计数（facade.seed.rejected 观测）。
    """
    facades = []
    fid = fid_start

    if len(seed_spatial_ids) < min_count:
        if stats is not None:
            stats['below_min'] += 1
        return facades, fid

    seed_plane = np.asarray(seed_plane, dtype=np.float64)
    sn = seed_plane[:3] / (np.linalg.norm(seed_plane[:3]) + 1e-12)

    ids = np.asarray(seed_spatial_ids, dtype=int)
    signed = points[ids] @ sn + seed_plane[3]
    candidate = np.abs(signed) <= tol[ids] * 1.5
    candidate &= np.abs(normals[ids] @ sn) >= cos_tol[ids]
    cids = ids[candidate]
    if len(cids) < min_count:
        if stats is not None:
            stats['below_min'] += 1
        return facades, fid

    tight = np.abs(points[cids] @ sn + seed_plane[3]) <= tol[cids] * signed_dist_tolerance
    cids = cids[tight]
    if len(cids) < min_count:
        if stats is not None:
            stats['below_min'] += 1
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
        if stats is not None:
            stats['no_comp'] += 1
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
            if stats is not None:
                stats['below_min'] += 1
            continue
        model = fit_plane_weighted(points[comp], irls_iters=irls_iters)
        dist = np.abs(points[comp] @ model[:3] + model[3])
        keep = dist <= tol[comp] * 1.8
        comp = comp[keep]
        if len(comp) < min_count:
            if stats is not None:
                stats['below_min'] += 1
            continue
        info = _info(fid, model, points[comp], comp, ranges[comp])
        if info['type'] == 'vertical_facade' and info['area'] >= min_facade_area:
            facades.append(info)
            fid += 1
        elif stats is not None:
            if info['type'] != 'vertical_facade':
                stats['bad_type'] += 1
            else:
                stats['small_area'] += 1

    return facades, fid


def _deduplicate_facades(facades, points, iou_thresh=0.5,
                         depth_diff_thresh=None, angle_deg=5.0):
    """按 inlier 集合 IoU / 包含率 + 法向/深度差异去重。保留大的。

    【v3】depth_diff_thresh 默认收紧为 DEDUP_DEPTH_THR_M（<5cm）：
    两个真实平行立面（如窗台内凹 6cm）若 IoU 高，旧值 0.08 会把它们
    当重复面删除其一，违反深度硬隔离要求。

    【优化1】新增包含率规则：同面（深度差 < 4.5cm）且小片段 inlier 的
    ≥90% 已落在大片段内时，无论 IoU 是否过半都判重。修复同一面墙被两个
    种子（含边界 ε 双归属）各拟合一次、小片落入大片洞内区域时 IoU 低
    导致去重漏杀的问题。

    【优化2】inlier 集合改用排序 int64 数组 + intersect1d（O(k log k)），
    替代原来的 O(F×N) 全点布尔掩码（120 万点 × 276 面）。
    """
    if not facades:
        return facades
    if depth_diff_thresh is None:
        depth_diff_thresh = DEDUP_DEPTH_THR_M

    facades.sort(key=lambda f: -f['point_count'])
    kept = []          # facade dicts
    kept_ids = []      # sorted int64 arrays，与 kept 对齐

    for f in facades:
        ids = np.sort(np.asarray(f['inlier_indices'], dtype=np.int64))
        fn = np.asarray(f['normal'])
        fm = np.asarray(f['plane_model'])
        fd = float(fm[3]) / (np.linalg.norm(fm[:3]) + 1e-12)

        duplicate = False
        for i in range(len(kept)):
            inter = np.intersect1d(ids, kept_ids[i])   # 输入已排序
            if len(inter) == 0:
                continue
            union = len(ids) + len(kept_ids[i]) - len(inter)
            iou = len(inter) / union if union > 0 else 0.0
            contain = len(inter) / max(1, min(len(ids), len(kept_ids[i])))

            kn = np.asarray(kept[i]['normal'])
            if abs(fn @ kn) < np.cos(np.deg2rad(angle_deg)):
                continue

            km = np.asarray(kept[i]['plane_model'])
            kd = float(km[3]) / (np.linalg.norm(km[:3]) + 1e-12)
            if abs(fd - kd) >= depth_diff_thresh:
                continue

            if iou > iou_thresh or contain >= 0.90:
                duplicate = True
                break

        if not duplicate:
            kept.append(f)
            kept_ids.append(ids)

    return kept


def _extract_bucket_to_facades(bucket_seeds, points, normals, ranges, tol, cos_tol,
                               spacing, voxel_size, base, min_count, min_facade_area,
                               irls_iters, signed_dist_tolerance,
                               density_field=None):
    """非竞争提取：每个种子独立扩展，最后全局去重。

    【v3】密度门控：种子 spatial_ids 内 skip 占比过半的点集直接跳过
    （过稀疏区域不产生立面，也防止 SVD 拟合出伪平面）。
    【三期】扩展最小点数密度自适应：dense = max(min_count//4, 50)，
    sparse ×0.4（窗台等稀疏立面碎片可成活，交给后续合并与裁决）。
    """
    all_facades = []
    fid = 0
    seed_min_pts = max(50, min_count // 10)
    expand_min_base = max(min_count // 4, 50)
    stats = {'below_min': 0, 'no_comp': 0, 'bad_type': 0, 'small_area': 0}

    for seed in bucket_seeds:
        spatial_ids = np.asarray(seed.get('spatial_ids', []), dtype=int)
        if len(spatial_ids) < seed_min_pts:
            continue

        # 密度门控：过于稀疏的种子区域直接跳过；等级复用给 min_count
        seed_level = 'dense'
        if density_field is not None and _HAS_DENSITY:
            seed_level = density_field.gate_indices(spatial_ids)
            if seed_level == 'skip':
                trace('facade.bucket.skip_sparse_seed',
                      n=int(len(spatial_ids)),
                      seed_rho=round(float(seed.get('rho', 0)), 3))
                continue
        if _HAS_DENSITY:
            eff_expand_min = adaptive_min_count(expand_min_base, seed_level)
        else:
            eff_expand_min = expand_min_base

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
            eff_expand_min, min_facade_area, irls_iters,
            signed_dist_tolerance, fid_start=fid, stats=stats)
        all_facades.extend(facades)

    if any(stats.values()):
        trace('facade.seed.rejected', **stats)

    if len(all_facades) > 1:
        all_facades = _deduplicate_facades(all_facades, points)

    return all_facades


# ==================== E. 同层合并 ====================

def _merge_same_layer(facades, points, normals, d_thresh=0.03, angle_deg=5.0,
                      max_plane_dist=0.05, layer_hard_m=None,
                      intensities=None):
    """用确定性候选图合并被门窗/楼层空洞切开的同层片段。

    候选边沿用双向平面偏移、法向、UV/AABB、强度和深度单峰门控；每次
    union 前对整个拟议组件重新拟合，并复验 P95、MAD、5cm 深度跨度和
    单峰性，防止 A-B、B-C 的传递闭包错误地把 A-C 合并。
    """
    if not facades:
        return facades

    min_opening = float(getattr(Config, 'FACADE_MIN_OPENING_M', 0.5))
    merge_gap_uv = float(getattr(Config, 'FACADE_MERGE_SAME_LAYER_GAP_M', 3.0))
    min_merge_gap = float(getattr(
        Config, 'FACADE_MERGE_SAME_LAYER_MIN_GAP_M',
        max(min_opening * 0.2, 0.10)))

    # 桥接窗自适应参数
    gap_span_ratio = float(getattr(Config, 'FACADE_MERGE_GAP_SPAN_RATIO', 0.5))
    gap_max_m = float(getattr(Config, 'FACADE_MERGE_GAP_MAX_M', 12.0))
    confident_cos = np.cos(np.deg2rad(float(getattr(
        Config, 'FACADE_MERGE_CONFIDENT_ANGLE_DEG', 2.0))))
    confident_fill_slack = float(getattr(
        Config, 'FACADE_MERGE_CONFIDENT_FILL_SLACK', 0.15))
    opening_overlap_min_m = float(getattr(
        Config, 'FACADE_MERGE_OPENING_MIN_OVERLAP_M', 0.30))
    opening_overlap_ratio = float(getattr(
        Config, 'FACADE_MERGE_OPENING_MIN_OVERLAP_RATIO', 0.08))

    # 硬约束：两段平面互嵌深度差上限 = 5cm，不加 d_thresh 裕量
    layer_hard_eff = DEPTH_HARD_M
    intensity = None
    if intensities is not None:
        candidate_intensity = np.asarray(intensities, dtype=float).reshape(-1)
        if len(candidate_intensity) == len(points):
            intensity = candidate_intensity
    intensity_delta = float(getattr(
        Config, 'FACADE_MERGE_INTENSITY_DELTA', 0.10))
    use_intensity = bool(getattr(
        Config, 'FACADE_ENABLE_INTENSITY_FEATURE', True)) and intensity is not None

    def _span2d(f):
        """立面自身 UV 外接框长边（米），作桥接窗的自适应基准。"""
        bb = f.get('bbox_2d') or {}
        u = max(float(bb.get('u_max', 0.0)) - float(bb.get('u_min', 0.0)), 0.0)
        v = max(float(bb.get('v_max', 0.0)) - float(bb.get('v_min', 0.0)), 0.0)
        return max(u, v)

    ordered = sorted(
        facades,
        key=lambda f: (int(f.get('id', 0)),
                       min(f.get('inlier_indices', [0])),
                       -int(f.get('point_count', 0))))
    count = len(ordered)
    parent = list(range(count))
    members = {i: [i] for i in range(count)}

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def pair_candidate(i, j):
        a = ordered[i]
        na = np.asarray(a['normal'])
        pa = np.asarray(a['plane_model'])
        ia = np.asarray(a['inlier_indices'], dtype=int)
        b = ordered[j]
        nb = np.asarray(b['normal'])
        if a['type'] != b['type'] or abs(na @ nb) < np.cos(np.deg2rad(angle_deg)):
            return None
        pb = np.asarray(b['plane_model'])
        ca, cb = np.asarray(a['center']), np.asarray(b['center'])
        off_a = abs(cb @ pa[:3] + pa[3]) / (np.linalg.norm(pa[:3]) + 1e-12)
        off_b = abs(ca @ pb[:3] + pb[3]) / (np.linalg.norm(pb[:3]) + 1e-12)
        if max(off_a, off_b) > layer_hard_eff:
            return None
        ib = np.asarray(b['inlier_indices'], dtype=int)

        # 桥接窗按立面尺度自适应；AABB 窗随动（AABB 间隙 ≥ UV 间隙）
        pair_gap_cap = min(max(merge_gap_uv,
                               gap_span_ratio * max(_span2d(a), _span2d(b))),
                           gap_max_m)
        aabb_gap = _aabb_gap_m(points[ia], points[ib])
        if aabb_gap > pair_gap_cap * 2.0:
            return None

        spacing_a = (max(0.05, float(np.median(np.linalg.norm(
            points[ia][1:] - points[ia][:-1], axis=1)))) if len(ia) > 1 else 0.05)
        grid = max(0.10, spacing_a * 2.0)
        common_n = na + np.sign(float(na @ nb) or 1.0) * nb
        common_n /= np.linalg.norm(common_n) + 1e-12
        _, box_a = _uv_descriptor(a, points, grid, common_n)
        _, box_b = _uv_descriptor(b, points, grid, common_n)
        gap_u = max(box_a[0] - box_b[1], box_b[0] - box_a[1], 0.0)
        gap_v = max(box_a[2] - box_b[3], box_b[2] - box_a[3], 0.0)
        gap_uv = float(np.hypot(gap_u, gap_v))
        if gap_uv < min_merge_gap or gap_uv > pair_gap_cap:
            return None

        # 门窗洞口通常只在一个 UV 方向形成间隙，另一方向仍有明显投影重叠。
        # 若两个方向都分离，仅凭“共面 + 距离近”容易把相邻楼栋/转角外的独立
        # 墙片误并。这里要求至少一个方向具备与较小片段尺度相称的重叠证据。
        span_au, span_av = box_a[1] - box_a[0], box_a[3] - box_a[2]
        span_bu, span_bv = box_b[1] - box_b[0], box_b[3] - box_b[2]
        overlap_u = max(min(box_a[1], box_b[1]) - max(box_a[0], box_b[0]), 0.0)
        overlap_v = max(min(box_a[3], box_b[3]) - max(box_a[2], box_b[2]), 0.0)
        required_u = min(opening_overlap_min_m,
                         opening_overlap_ratio * max(min(span_au, span_bu), 0.0))
        required_v = min(opening_overlap_min_m,
                         opening_overlap_ratio * max(min(span_av, span_bv), 0.0))
        opening_bridge = ((gap_u > 0.0 and gap_v == 0.0 and overlap_v >= required_v)
                          or (gap_v > 0.0 and gap_u == 0.0 and overlap_u >= required_u))
        if not opening_bridge:
            return None

        # 【优化】高置信判定：共面证据极强时对填充率跌幅放宽
        confident = (abs(float(na @ nb)) >= confident_cos and
                     max(off_a, off_b) <= max(max_plane_dist * 0.5, 0.02))

        intensity_penalty = 0.0
        if use_intensity:
            va, vb = intensity[ia], intensity[ib]
            va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
            if len(va) >= 20 and len(vb) >= 20:
                med_delta = abs(float(np.median(va) - np.median(vb)))
                spread = max(float(np.ptp(np.percentile(va, [25, 75]))),
                             float(np.ptp(np.percentile(vb, [25, 75]))), 0.02)
                intensity_penalty = min(
                    med_delta / max(intensity_delta, 1.5 * spread, 1e-6),
                    4.0)
        ids = np.unique(np.r_[ia, ib])
        model = fit_plane_weighted(points[ids], irls_iters=2)
        if not _check_depth_single_peak(points[ids], model, max_plane_dist):
            return None
        score = (intensity_penalty, gap_uv, max(off_a, off_b), aabb_gap,
                 min(int(a.get('id', i)), int(b.get('id', j))), i, j)
        return score, gap_u, gap_v, confident

    edges = []
    for i in range(count):
        for j in range(i + 1, count):
            candidate = pair_candidate(i, j)
            if candidate is not None:
                edges.append((candidate[0], i, j, candidate[1], candidate[2],
                              candidate[3]))
    edges.sort(key=lambda edge: edge[0])
    trace('facade.merge_same_layer.candidates',
          pairs=len(edges), fragments=count)

    def component_fit(component):
        # 在重新拟合前先用各源片段的共同法向复验层深跨度。仅检查拟合后残差
        # 不足以阻止 A-B-C 阶梯链：最小二乘可把空间上依次升高的片段拟合成
        # 一张倾斜平面，从而掩盖首尾超过 5cm 的真实深度差。
        reference = np.asarray(ordered[component[0]]['normal'], dtype=float)
        reference /= np.linalg.norm(reference) + 1e-12
        aligned_normals = []
        component_rhos = []
        for k in component:
            source_normal = np.asarray(ordered[k]['normal'], dtype=float)
            source_normal /= np.linalg.norm(source_normal) + 1e-12
            if float(source_normal @ reference) < 0.0:
                source_normal = -source_normal
            aligned_normals.append(source_normal)
        shared_normal = np.mean(aligned_normals, axis=0)
        shared_normal /= np.linalg.norm(shared_normal) + 1e-12
        for k in component:
            source_ids = np.asarray(ordered[k]['inlier_indices'], dtype=int)
            component_rhos.append(float(np.median(points[source_ids] @ shared_normal)))
        if (max(component_rhos) - min(component_rhos)) > DEPTH_HARD_M:
            return None

        ids = np.unique(np.concatenate([
            np.asarray(ordered[k]['inlier_indices'], dtype=int) for k in component]))
        model = fit_plane_weighted(points[ids], irls_iters=3)
        model = _canonicalize_plane(model)
        residual = np.abs(points[ids] @ model[:3] + model[3])
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        p95 = float(np.percentile(residual, 95))
        # Robust whole-component validation. The absolute hard bound remains 5cm.
        if p95 > min(DEPTH_HARD_M, max(max_plane_dist * 1.5, 0.02)):
            return None
        if mad > min(DEPTH_HARD_M / 3.0, max(max_plane_dist * 0.75, 0.008)):
            return None
        signed = points[ids] @ model[:3] + model[3]
        if float(np.percentile(signed, 95) - np.percentile(signed, 5)) > 2.0 * DEPTH_HARD_M:
            return None
        if not _check_depth_single_peak(points[ids], model, max_plane_dist):
            return None
        return ids, model

    for _, left, right, gap_u, gap_v, confident in edges:
        root_l, root_r = find(left), find(right)
        if root_l == root_r:
            continue
        proposed = sorted(members[root_l] + members[root_r])
        fit = component_fit(proposed)
        if fit is None:
            continue
        if gap_u > 0.0 and gap_v > 0.0:
            ids, model = fit
            merged_info = _info(ordered[proposed[0]]['id'], model, points[ids], ids,
                                np.zeros(len(ids)))
            # 【优化】跨大洞合并联合 bbox 必然变大、填充率下跌；
            # 高置信共面对只放宽跌幅，非高置信对保留原严格门槛。
            slack = confident_fill_slack if confident else 0.08
            if _fill_ratio(merged_info) < min(_fill_ratio(ordered[k]) for k in proposed) - slack:
                continue
        keep, drop = (root_l, root_r) if root_l < root_r else (root_r, root_l)
        parent[drop] = keep
        members[keep] = proposed
        del members[drop]

    result = []
    for component in sorted(members.values(), key=lambda group: group[0]):
        fit = component_fit(component) if len(component) > 1 else None
        if fit is None:
            if len(component) == 1:
                result.append(ordered[component[0]])
            else:  # Defensive: never discard source fragments on a failed final fit.
                result.extend(ordered[k] for k in component)
            continue
        ids, model = fit
        sources = [ordered[k] for k in component]
        merged = _info(min(int(f.get('id', 0)) for f in sources), model,
                       points[ids], ids, np.zeros(len(ids)))
        merged['dir_bin'] = sources[0].get('dir_bin', -1)
        merged['layer'] = min(f.get('layer', -1) for f in sources)
        merged['attachments'] = [item for f in sources for item in f.get('attachments', [])]
        result.append(merged)
    return result


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

    cross_aabb_max = float(getattr(Config, 'FACADE_CROSS_LAYER_MAX_AABB_GAP_M', 3.0))

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
                if _aabb_gap_m(points[ia], points[ib]) > cross_aabb_max:
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


# ==================== H. 完形回收====================
def _postprocess_v2(facades, points, normals, tol, cos_tol, spacing, voxel_size,
                    min_opening=.5, intensities=None):
    """以标准化联合代价对未归属点执行一次全局唯一归属。

    每个点先收集通过 UV 膨胀、AABB、平面残差和法向门控的所有立面，
    intensity 存在时仅增加软代价。最优与次优过近的点保持未归属，避免
    facade 遍历顺序决定结果；最终仍对每个拟议立面重新拟合并复验形状。
    """
    if not facades:
        return facades

    assigned = np.zeros(len(points), dtype=bool)
    for f in facades:
        assigned[np.asarray(f['inlier_indices'], dtype=int)] = True

    reclaim_mult = float(getattr(Config, 'FACADE_RECLAIM_TOL_MULT', 1.2))
    ambiguity_margin = float(getattr(Config, 'FACADE_RECLAIM_AMBIGUITY_MARGIN', 0.12))
    reclaim_dilate_m = float(getattr(Config, 'FACADE_RECLAIM_UV_DILATE_M', 0.30))
    intensity_arr = None
    if intensities is not None:
        candidate = np.asarray(intensities, dtype=float).reshape(-1)
        if len(candidate) == len(points):
            intensity_arr = candidate

    unassigned = np.flatnonzero(~assigned)
    candidates = [[] for _ in range(len(points))]
    facade_context = []

    for facade_idx, f in enumerate(facades):
        ids = np.asarray(f['inlier_indices'], dtype=int)
        if not len(ids):
            facade_context.append(None)
            continue
        n = np.asarray(f['normal'], dtype=float)
        n /= np.linalg.norm(n) + 1e-12

        spacing_m = max(0.05, float(np.median(spacing[ids])))
        grid = max(0.10, spacing_m * 2.0)
        cells, _ = _uv_descriptor(f, points, grid, normal=n)
        dilated = _dilate_cells(cells, min(2, max(1, int(round(reclaim_dilate_m / grid)))))
        pad = max(1.5 * float(np.median(spacing[ids])), .2)
        lo, hi = points[ids].min(0) - pad, points[ids].max(0) + pad
        pm = _canonicalize_plane(f['plane_model'])
        u, v = plane_axes(n, f['type'])
        intensity_med = np.nan
        intensity_scale = 0.02
        if intensity_arr is not None:
            values = intensity_arr[ids]
            values = values[np.isfinite(values)]
            if len(values) >= 20:
                intensity_med = float(np.median(values))
                intensity_scale = max(float(np.subtract(*np.percentile(values, [75, 25]))), 0.02)
        facade_context.append((ids, pm, intensity_med, intensity_scale))

        cand = unassigned[np.all((points[unassigned] >= lo) &
                                 (points[unassigned] <= hi), axis=1)]
        if not len(cand):
            continue
        residual = np.abs(points[cand] @ pm[:3] + pm[3])
        agreement = np.abs(normals[cand] @ n)
        ok = ((residual <= np.minimum(tol[cand] * reclaim_mult, DEPTH_HARD_M)) &
              (agreement >= cos_tol[cand]))
        cand, residual, agreement = cand[ok], residual[ok], agreement[ok]
        if not len(cand):
            continue
        uv = np.column_stack((points[cand] @ u, points[cand] @ v))
        uv_cells = list(zip(np.floor(uv[:, 0] / grid).astype(np.int64),
                            np.floor(uv[:, 1] / grid).astype(np.int64)))
        uv_ok = np.asarray([cell in dilated for cell in uv_cells], dtype=bool)
        cand, residual, agreement = cand[uv_ok], residual[uv_ok], agreement[uv_ok]
        for point_idx, dist, normal_cos in zip(cand, residual, agreement):
            dist_cost = float(dist / max(min(tol[point_idx] * reclaim_mult,
                                             DEPTH_HARD_M), 1e-6))
            normal_cost = float((1.0 - normal_cos) /
                                max(1.0 - cos_tol[point_idx], 1e-6))
            cost = 0.70 * dist_cost + 0.30 * normal_cost
            if intensity_arr is not None and np.isfinite(intensity_med):
                value = intensity_arr[point_idx]
                if np.isfinite(value):
                    # Soft evidence only: geometry can still win when intensity differs.
                    cost += 0.10 * min(abs(float(value) - intensity_med) /
                                       max(3.0 * intensity_scale, 0.05), 1.0)
            candidates[int(point_idx)].append((cost, facade_idx))

    additions = [[] for _ in facades]
    ambiguous_count = 0
    no_candidate_count = 0
    for point_idx in unassigned:
        choices = sorted(candidates[int(point_idx)], key=lambda item: (item[0], item[1]))
        if not choices:
            no_candidate_count += 1
            continue
        if len(choices) > 1 and choices[1][0] - choices[0][0] < ambiguity_margin:
            ambiguous_count += 1
            continue
        additions[choices[0][1]].append(int(point_idx))

    for facade_idx, f in enumerate(facades):
        ids = np.asarray(f['inlier_indices'], dtype=int)
        add = np.asarray(additions[facade_idx], dtype=int)
        max_add = max(len(ids) // 3, 5000)
        if len(add) > max_add:
            pm = np.asarray(f['plane_model'], dtype=float)
            add = add[np.argsort(np.abs(points[add] @ pm[:3] + pm[3]))[:max_add]]
        if len(add) >= max(10, len(ids) // 50):
            all_ids = np.unique(np.r_[ids, add])
            new_model = fit_plane_weighted(points[all_ids], irls_iters=2)
            residual = np.abs(points[all_ids] @ new_model[:3] + new_model[3])
            new_n = new_model[:3] / (np.linalg.norm(new_model[:3]) + 1e-12)
            old_n = np.asarray(f['normal'], dtype=float)
            if (abs(new_n @ old_n) >= np.cos(np.deg2rad(5.0)) and
                    float(np.percentile(residual, 95)) <= DEPTH_HARD_M and
                    _check_depth_single_peak(points[all_ids], new_model, DEPTH_HARD_M)):
                new_info = _info(f['id'], new_model, points[all_ids], all_ids,
                                 np.zeros(len(all_ids)))
                slack = float(getattr(Config, 'FACADE_RECLAIM_FILL_SLACK', 0.05))
                if _fill_ratio(new_info) >= _fill_ratio(f) - slack:
                    f.update(new_info)

    for f in facades:
        fill = _fill_ratio(f)
        f['fill_ratio'] = round(float(fill), 3)
        f['wall_kind'] = 'fragment' if fill < .25 else ('fenestrated' if fill < .55 else 'solid')

    proposed_count = int(sum(len(ids) for ids in additions))
    trace('facade.assignment', input_points=len(points),
          seed_assigned=int(np.count_nonzero(assigned)),
          unassigned_input=int(len(unassigned)), proposed=proposed_count,
          ambiguous=int(ambiguous_count), no_candidate=int(no_candidate_count))

    return facades


# ==================== 主入口 ====================

def detect_facades_adaptive(pcd, voxel_size=.05, min_facade_area=5., max_plane_dist=None,
                            min_points_ratio=.003, roi_bounds=None, roi_indices=None,
                            enable_merge=True, enable_grow=False, signed_dist_tolerance=1.,
                            range_adaptive=True, scan_origin=None, range_coeff=.0012,
                            normal_relax_deg_per_m=.15, normal_angle_max_deg=15.,
                            irls_iters=2, metadata=None, normals=None, elevations=None,
                            intensities=None,
                            **_kwargs):
    """主入口：多分辨率立面检测，支持 precluster / bucket / hybrid 种子模式。

    当传入 ``normals``（Nx3 ndarray，长度与 pcd 一致且全部有限）时，
    直接复用该法向量，完全跳过内部 ``ensure_normals`` 的 KD-tree 估计。
    若 ``pcd`` 本身已带法向量，也优先复用；代理点直接使用 ``pcd.points``。

    FACADE_SEED_MODE 配置：
      - 'precluster'：空间邻接 × 法向一致 × 共享深度一致局部图预聚类。
      - 'bucket'（默认）：方向桶 × 深度层确定性枚举 + 非竞争扩展 + 全局去重。
      - 'hybrid'     ：precluster + bucket 双通道。

    【v3 新增】elevations 参数：高度角数组（度），供密度场高/低分层；
    None 时密度模块以 z 中位数近似分层。
    """
    started = time.perf_counter()
    points = np.asarray(pcd.points, float)
    n = len(points)
    if not n:
        return {'facades': [], 'remaining': pcd, 'total_points': 0}

    # Remove non-finite samples and isolated ROI noise before normal estimation.
    # All point-aligned inputs are sliced together, while returned facade indices
    # are mapped back to the original detection-input index space.
    input_count = n
    meta = metadata or {}
    # PointCloudService adaptive denoise already rebuilt every point-aligned
    # proxy array.  Repeating radius filtering inside detection is redundant
    # and used to scan million-point ROIs a second time.  Non-finite samples
    # are still removed unconditionally by finite_sparse_keep_indices.
    already_denoised = bool(meta.get('denoise_history'))
    finite_sparse_keep = finite_sparse_keep_indices(
        points, voxel_size=float(voxel_size),
        radius=float(getattr(Config, 'FACADE_OUTLIER_RADIUS_M',
                             max(float(voxel_size) * 2.5, 0.08))),
        min_neighbors=int(getattr(Config, 'FACADE_OUTLIER_MIN_NEIGHBORS', 2)),
        enabled=(bool(getattr(Config, 'FACADE_ENABLE_SPARSE_OUTLIER_FILTER', True))
                 and not already_denoised))
    source_indices = np.flatnonzero(finite_sparse_keep)
    if len(source_indices) != n:
        points = points[source_indices]
        n = len(points)
        if normals is not None and len(normals) == input_count:
            normals = np.asarray(normals)[source_indices]
        if intensities is not None and len(intensities) == input_count:
            intensities = np.asarray(intensities)[source_indices]
        if elevations is not None and len(elevations) == input_count:
            elevations = np.asarray(elevations)[source_indices]
        if roi_indices is not None:
            inverse = np.full(input_count, -1, dtype=np.int64)
            inverse[source_indices] = np.arange(n)
            roi_indices = inverse[np.asarray(roi_indices, dtype=np.int64)]
            roi_indices = roi_indices[roi_indices >= 0]
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(points)
        pcd = filtered_pcd
        trace('facade.algo.preprocess', input_points=input_count,
              retained_points=n, removed_points=input_count - n)
    else:
        source_indices = np.arange(n, dtype=np.int64)

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

    intensity_arr = None
    if intensities is not None:
        candidate = np.asarray(intensities, float).reshape(-1)
        if len(candidate) == n:
            intensity_arr = candidate

    seed_mode = str(getattr(Config, 'FACADE_SEED_MODE', 'bucket')).lower()

    cfg = meta.get('adaptive_detection', {})
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

    facades = []

    # ===== 种子阶段：precluster + bucket 双通道全域非竞争提取 =====
    fine_facades = []

    # 1) 预聚类种子：快速发现高置信候选
    if seed_mode in ('precluster', 'hybrid') and active.sum() >= min_count:
        patch_seeds = build_depth_normal_patch_seeds(
            points, normals, active, voxel_size=float(voxel_size),
            min_count=min_count, depth_hard_m=DEPTH_HARD_M,
            normal_angle_deg=float(getattr(
                Config, 'FACADE_PRECLUSTER_NORMAL_ANGLE_DEG', 10.0)),
            neighbour_radius=float(getattr(
                Config, 'FACADE_PRECLUSTER_RADIUS_M',
                max(float(voxel_size) * 3.0, 0.12))),
            intensities=intensity_arr,
            density_field=density_field,
            debug_features=bool(getattr(
                Config, 'FACADE_DEBUG_PATCH_FEATURES', False)))
        if patch_seeds:
            fine_facades = _extract_bucket_to_facades(
                patch_seeds, points, normals, ranges,
                tol, cos_tol, spacing, voxel_size, base,
                min_count, min_facade_area, irls_iters, signed_dist_tolerance,
                density_field=density_field)
            trace('facade.algo.precluster', seeds=len(patch_seeds),
                  facades=len(fine_facades),
                  seconds=f"{time.perf_counter()-started:.2f}")

    # 2) Bucket 种子（含残余方向补充通道）：全域运行，补充召回
    if ((seed_mode == 'bucket' or seed_mode == 'hybrid')
            and active.sum() >= min_count):
        bucket_seeds = build_bucket_seeds(
            points, normals, active, ranges, tol, base, min_count, voxel_size,
            elevations=elevations, density_field=density_field)
        if bucket_seeds:
            bucket_facades = _extract_bucket_to_facades(
                bucket_seeds, points, normals, ranges,
                tol, cos_tol, spacing, voxel_size, base,
                min_count, min_facade_area, irls_iters, signed_dist_tolerance,
                density_field=density_field)
            fine_facades.extend(bucket_facades)

    trace("facade.algo.seeds", seed_mode=seed_mode,
          fine_facades=len(fine_facades),
          seconds=f"{time.perf_counter()-started:.2f}")

    facades.extend(fine_facades)

    # ===== E. 同层合并（跨板带/洞口桥接，几何复验保精度）=====
    if enable_merge:
        facades = _merge_same_layer(
            facades, points, normals,
            d_thresh=float(getattr(Config, 'MERGE_D_THRESH_M', 0.08)),
            angle_deg=float(getattr(Config, 'MERGE_ANGLE_DEG', 5.0)),
            max_plane_dist=base,
            layer_hard_m=DEPTH_HARD_M,
            intensities=intensity_arr)
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

    # ===== H. 完形回收 =====
    facades = _postprocess_v2(
        facades, points, normals, tol, cos_tol, spacing, voxel_size,
        intensities=intensity_arr)

    # ===== 【v3】完形回收后再次终审：回收可能重新引入跨层点 =====
    facades, n_viol2 = _enforce_depth_hard_split(facades, points)
    if n_viol2:
        trace('facade.algo.depth_hard_split.post_reclaim',
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
    assigned_count = int(np.count_nonzero(used))
    trace("facade.algo.done", facades=len(facades), total_points=n,
          assigned_points=assigned_count, unassigned_points=int(n - assigned_count),
          coverage=f"{assigned_count / max(n, 1):.4f}",
          seconds=f"{time.perf_counter()-started:.2f}")

    for facade in facades:
        ids = np.asarray(facade['inlier_indices'], dtype=np.int64)
        mapped = source_indices[ids]
        facade['inlier_indices'] = mapped.astype(int).tolist()
        facade['support_indices'] = mapped.astype(int).tolist()
        facade.pop('weak', None)
        facade['layer'] = int(facade.get('layer', -1))
        facade['dir_bin'] = int(facade.get('dir_bin', -1))
        facade['attachments'] = list(facade.get('attachments') or [])
        facade['adjacency'] = list(facade.get('adjacency') or [])
    return {'facades': facades, 'remaining': out, 'total_points': input_count}