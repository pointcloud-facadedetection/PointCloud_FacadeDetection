"""距离自适应立面检测：多分辨率 Hough 种子 → 局部 RANSAC 精化 → 
UV 连通域（自适应孔洞闭合）→ 3D 空间约束合并 → 完形回收。

该模块只输出当前输入点云空间的局部索引，FacadeService 负责把它们归一化
到 proxy_global，因此不会改变 VoxelCascadeIndex 的映射契约。

五阶段优化：
  1. 空间化 Hough：3D 粗网格隔离，方向+rho 投票仅在空间块内进行
  2. 局部 RANSAC 精化：Hough 提供方向先验，局部 RANSAC 定标平面参数
  3. 自适应孔洞闭合：主立面大闭合半径（跨门窗），细部保守
  4. 粗到细双速：粗层(20cm)快速版图 + 细层(原分辨率)边缘精化
  5. 3D 空间硬约束：AABB 快速排斥 + 体素连通域兜底
"""
from __future__ import annotations

import copy
import numpy as np
import open3d as o3d

from algorithms.geometry import (
    ensure_normals, classify_plane, project_to_uv,
    connected_components_2d_grid, connected_components_3d_grid,
    estimate_plane_area, plane_axes,
    estimate_point_ranges, adaptive_plane_tolerance, fit_plane_weighted
)
from config.settings import Config
from algorithms.facade.hough_facade import normal_hough_peaks, rho_hough_peaks


# ==================== 内部工具函数 ====================

def _info(fid, model, points, indices, ranges):
    model = np.asarray(model, dtype=float)
    n = model[:3] / (np.linalg.norm(model[:3]) + 1e-12)
    model[:3] = n
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


def _check_depth_single_peak(points, plane_model, max_plane_dist):
    """检查点集在平面深度方向是否单峰，避免合并两个平行立面。"""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 100:
        return True
    pm = np.asarray(plane_model, dtype=float)
    n = pm[:3] / (np.linalg.norm(pm[:3]) + 1e-12)
    d = float(pm[3] / (np.linalg.norm(pm[:3]) + 1e-12))
    signed = pts @ n + d
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


def _merge(facades, points, gap_m=3.0, angle_deg=5.0, d_thresh=.08, max_plane_dist=0.05):
    """共面且 UV 邻接合并；增加 3D AABB 快速排斥，禁止合并空间分离或深度双峰的立面。"""
    # 每个 facade 只建立一次 KD-tree；合并循环可能反复访问同一 facade。
    spacing_cache = {}
    for facade in facades:
        ids = np.asarray(facade.get('inlier_indices', []), dtype=int)
        if len(ids) <= 1:
            spacing_cache[id(facade)] = 0.05
            continue
        nn = np.asarray(o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(points[ids])
        ).compute_nearest_neighbor_distance(), dtype=float)
        positive = nn[nn > 0]
        spacing_cache[id(facade)] = max(0.05, float(np.median(positive))) if len(positive) else 0.05
    
    changed = True
    while changed:
        changed = False
        facades.sort(key=lambda f: -f['point_count'])
        for i in range(len(facades)):
            for j in range(i + 1, len(facades)):
                a, b = facades[i], facades[j]
                na, nb = np.asarray(a['normal']), np.asarray(b['normal'])
                if a['type'] != b['type'] or abs(na @ nb) < np.cos(np.deg2rad(angle_deg)):
                    continue
                pa, pb = np.asarray(a['plane_model']), np.asarray(b['plane_model'])
                ca, cb = np.asarray(a['center']), np.asarray(b['center'])
                off_a = abs(cb @ pa[:3] + pa[3]) / (np.linalg.norm(pa[:3]) + 1e-12)
                off_b = abs(ca @ pb[:3] + pb[3]) / (np.linalg.norm(pb[:3]) + 1e-12)
                if max(off_a, off_b) > d_thresh:
                    continue
                
                ia = np.asarray(a['inlier_indices'], dtype=int)
                ib = np.asarray(b['inlier_indices'], dtype=int)
                
                # ===== 阶段 5：3D AABB 快速排斥 =====
                aabb_gap = _aabb_gap_m(points[ia], points[ib])
                if aabb_gap > gap_m:
                    continue  # 空间相距超过 gap_m，绝不合并
                
                spacing = spacing_cache.get(id(a), 0.05)
                grid = max(0.10, spacing * 2.0)
                common_n = na + nb
                common_n /= np.linalg.norm(common_n) + 1e-12
                cells_a, box_a = _uv_descriptor(a, points, grid, common_n)
                cells_b, box_b = _uv_descriptor(b, points, grid, common_n)
                gap_u = max(box_a[0] - box_b[1], box_b[0] - box_a[1], 0.0)
                gap_v = max(box_a[2] - box_b[3], box_b[2] - box_a[3], 0.0)
                if np.hypot(gap_u, gap_v) > gap_m:
                    continue
                
                # 深度单峰校验
                merged_pts = points[np.unique(np.r_[ia, ib])]
                avg_d = (off_a + off_b) / 2.0
                test_model = np.array([na[0], na[1], na[2], avg_d])
                if not _check_depth_single_peak(merged_pts, test_model, max_plane_dist):
                    continue
                
                ids = np.unique(np.r_[ia, ib])
                model = fit_plane_weighted(points[ids], irls_iters=2)
                merged = _info(a['id'], model, points[ids], ids, np.zeros(len(ids)))
                spacing_cache[id(merged)] = max(
                    spacing_cache.get(id(a), 0.05), spacing_cache.get(id(b), 0.05))
                facades[i] = merged
                del facades[j]
                changed = True
                break
            if changed:
                break
    return facades


def _postprocess(facades, points, normals, tol, cos_tol, spacing, voxel_size, min_opening=.5):
    """完形回收：增加 3D 体素连通域约束，防止回收远处平行面点。"""
    if not facades:
        return facades
    
    assigned = np.zeros(len(points), dtype=bool)
    for f in facades:
        assigned[np.asarray(f['inlier_indices'], dtype=int)] = True
    
    for f in facades:
        ids = np.asarray(f['inlier_indices'], dtype=int)
        if not len(ids):
            continue
        n = np.asarray(f['normal'], dtype=float)
        
        spacing_m = max(0.05, float(np.median(spacing[ids])))
        grid = max(0.10, spacing_m * 2.0)
        cells = _uv_cells_for_facade(f, points, grid)
        # 完形回收用较小膨胀（只补边缘），避免吞并
        dilated = _dilate_cells(cells, min(2, max(1, int(round(0.20 / grid)))))
        
        cand = np.where(~assigned)[0]
        d = np.abs(points[cand] @ n + f['plane_model'][3])
        ok = (d <= tol[cand] * 1.5) & (np.abs(normals[cand] @ n) >= cos_tol[cand])
        add = cand[ok]
        
        if len(add):
            # AABB 膨胀约束
            lo = points[ids].min(0) - max(1.5 * np.median(spacing[ids]), .2)
            hi = points[ids].max(0) + max(1.5 * np.median(spacing[ids]), .2)
            add = add[np.all((points[add] >= lo) & (points[add] <= hi), axis=1)]
            
            # UV 邻接约束
            if len(add):
                pm = np.asarray(f['plane_model'], dtype=float)
                typ, _, _, _ = classify_plane(n)
                uv_add, _, _, _ = project_to_uv(points[add], pm, typ)
                cells_add = list(zip(np.floor(uv_add[:, 0] / grid).astype(np.int64),
                                     np.floor(uv_add[:, 1] / grid).astype(np.int64)))
                add = add[np.asarray([c in dilated for c in cells_add], dtype=bool)]
            
            # ===== 阶段 5：3D 体素连通域约束 =====
            if len(add) >= 10:
                test_pts = np.vstack([points[ids], points[add]])
                test_comp = connected_components_3d_grid(
                    test_pts, grid_size=max(0.3, voxel_size * 6), min_points=10)
                if len(test_comp) > 1:
                    # 找到包含立面主体点的连通域标签
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
        
        # 限制回收比例
        max_add = max(len(ids) // 3, 5000)
        if len(add) > max_add:
            add = add[np.argsort(np.abs(points[add] @ n + f['plane_model'][3]))[:max_add]]
        
        if len(add) >= max(10, len(ids) // 50):
            all_ids = np.unique(np.r_[ids, add])
            new_model = fit_plane_weighted(points[all_ids], irls_iters=2)
            new_n = new_model[:3] / (np.linalg.norm(new_model[:3]) + 1e-12)
            if abs(new_n @ n) >= np.cos(np.deg2rad(5.0)):
                f.update(_info(f['id'], new_model, points[all_ids], all_ids, np.zeros(len(all_ids))))
                assigned[add] = True
    
    # 结果分类字段
    for f in facades:
        bb = f.get('bbox_2d') or {}
        span = max((bb.get('u_max', 0) - bb.get('u_min', 0)) * 
                   (bb.get('v_max', 0) - bb.get('v_min', 0)), 1e-6)
        fill = f['area'] / span
        f['fill_ratio'] = round(float(fill), 3)
        f['wall_kind'] = 'fragment' if fill < .25 else ('fenestrated' if fill < .55 else 'solid')
    
    return facades


def _extract_facades_from_seeds(seed_iter, points, normals, remaining, ranges,
                                 tol, cos_tol, spacing, voxel_size, base,
                                 min_count, min_facade_area, irls_iters,
                                 signed_dist_tolerance):
    """核心提取循环：从种子序列中提取立面。与 detect_facades_adaptive 解耦，
    支持粗层和细层复用同一逻辑。"""
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
            
            # ===== 阶段 2：局部 RANSAC 精化 Hough Seed =====
            seed = np.r_[sn, -rho]
            # 在 Hough seed 附近取局部点做 RANSAC 精化
            signed_sp = points[spatial_ids] @ sn - rho
            near_mask = np.abs(signed_sp) <= max(base * 3, 0.15)
            local_ids = spatial_ids[near_mask]
            
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
            
            # 扩展范围限制在 spatial_ids 与 remaining 的交集
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
        
        # 有符号距离扩展
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
        
        # ===== 阶段 3：自适应孔洞闭合 =====
        # 判断是否为"主立面候选"：Hough seed 支持度高
        is_major = (seed_kind == 'hough' and 
                    seed_info.get('support', 0) > min_count * 2)
        min_opening_m = float(getattr(Config, 'FACADE_MIN_OPENING_M', 0.5))
        
        if is_major:
            # 主立面：允许跨越 min_opening_m * 2.5 的缝隙（大玻璃幕墙）
            close_radius = max(3, int(np.ceil(min_opening_m * 2.5 / uv_grid)))
        else:
            # 细部/小面：保守闭合
            close_radius = max(2, int(np.ceil(min_opening_m / uv_grid)))
        # 上限保护
        close_radius = min(close_radius, 12)
        
        comps = connected_components_2d_grid(
            uv, grid_size=uv_grid,
            min_cells=max(2, int(min_count * .01)),
            close_radius_cells=close_radius)
        
        if not comps and len(cids) <= min_count * 20:
            comps = [np.ones(len(cids), dtype=bool)]
        elif not comps:
            continue
        
        consumed = set()
        for mask in sorted(comps, key=lambda x: int(x.sum()), reverse=True):
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


def detect_facades_adaptive(pcd, voxel_size=.05, min_facade_area=5., max_plane_dist=None,
                            min_points_ratio=.003, roi_bounds=None, roi_indices=None,
                            enable_merge=True, enable_grow=False, signed_dist_tolerance=1.,
                            range_adaptive=True, scan_origin=None, range_coeff=.0012,
                            normal_relax_deg_per_m=.15, normal_angle_max_deg=15.,
                            irls_iters=2, metadata=None, **_kwargs):
    """主入口：多分辨率 Hough-IRLS 立面检测。
    
    保持与原有 facade_detection.py 完全相同的函数签名和返回结构。
    """
    work = ensure_normals(copy.deepcopy(pcd), voxel_size)
    points, normals = np.asarray(work.points, float), np.asarray(work.normals, float)
    n = len(points)
    if not n:
        return {'facades': [], 'remaining': work, 'total_points': 0}
    
    meta = metadata or {}
    cfg = meta.get('adaptive_detection', {})
    method = str(cfg.get('method', getattr(Config, 'FACADE_DETECT_METHOD', 'hybrid'))).lower()
    if method not in {'ransac', 'hough', 'hybrid'}:
        method = 'hybrid'
    
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
    
    # 间距估计：大数据集用体素尺寸作为稳健下界
    if n <= 100000:
        spacing = np.asarray(work.compute_nearest_neighbor_distance()) if n > 1 else np.zeros(n)
    else:
        spacing = np.full(n, max(float(voxel_size), 0.05), dtype=np.float32)
    
    remaining = active.copy()
    facades = []
    
    # ===== 阶段 4：粗到细双速提取 =====
    coarse_voxel = voxel_size * 4  # 20cm 粗层
    enable_coarse = (n > 100000 and method == 'hybrid' and 
                     getattr(Config, 'FACADE_ENABLE_COARSE_LAYER', True))
    
    if enable_coarse:
        # 粗层：20cm 体素下采样
        coarse_pcd = work.voxel_down_sample(coarse_voxel)
        coarse_points = np.asarray(coarse_pcd.points, float)
        coarse_normals = np.asarray(coarse_pcd.normals, float) if coarse_pcd.has_normals() else np.zeros_like(coarse_points)
        coarse_n = len(coarse_points)
        
        if coarse_n >= min_count:
            # 粗层参数
            coarse_ranges = estimate_point_ranges(coarse_points, scan_origin) if range_adaptive else np.zeros(coarse_n)
            coarse_tol = adaptive_plane_tolerance(coarse_ranges, base * 2.0, range_coeff, base * 6.0)
            coarse_theta = np.clip(float(getattr(Config, 'FACADE_NORMAL_ANGLE_DEG', 8.0)) + 
                                   normal_relax_deg_per_m * coarse_ranges, 8., normal_angle_max_deg)
            coarse_cos_tol = np.cos(np.deg2rad(coarse_theta))
            coarse_spacing = np.full(coarse_n, coarse_voxel, dtype=np.float32)
            
            # 粗层 Hough 种子生成（空间化）
            coarse_seeds = _build_spatial_hough_seeds(
                coarse_points, coarse_normals, np.ones(coarse_n, bool),
                coarse_ranges, coarse_tol, base, min_count, coarse_voxel)
            
            # 粗层提取：大容差保完整
            coarse_facades, coarse_remaining = _extract_facades_from_seeds(
                [('hough', s) for s in coarse_seeds],
                coarse_points, coarse_normals, np.ones(coarse_n, bool),
                coarse_ranges, coarse_tol, coarse_cos_tol, coarse_spacing,
                coarse_voxel, base * 2.0,
                max(min_count // 2, 50),  # 粗层降低点数门槛
                min_facade_area * 0.5,    # 粗层降低面积门槛
                irls_iters, 2.0)          # 粗层大容差
            
            # 将粗层立面映射回原始点云：用粗层平面对原始点做距离筛选
            covered = np.zeros(n, dtype=bool)
            for cf in coarse_facades:
                cn = np.asarray(cf['normal'])
                cd = float(np.asarray(cf['plane_model'])[3])
                # 粗层容差映射回原始点
                on_plane = (np.abs(points @ cn + cd) <= tol * 3.0) & \
                           (np.abs(normals @ cn) >= np.cos(np.deg2rad(10.0)))
                covered |= on_plane
            
            remaining &= ~covered
            # 粗层 facade 信息需要映射回原始点索引，这里简化：只保留平面参数
            # 实际映射：粗层点是子采样，用空间近邻或平面距离找回原始点
            for cf in coarse_facades:
                cn = np.asarray(cf['normal'])
                cd = float(np.asarray(cf['plane_model'])[3])
                # 在原始 remaining 点中找该平面的精确内点
                all_signed = points @ cn + cd
                all_near = np.abs(all_signed) <= tol * 1.2
                all_near &= np.abs(normals @ cn) >= cos_tol
                if all_near.sum() >= min_count:
                    model = fit_plane_weighted(points[all_near], irls_iters=irls_iters)
                    facades.append(_info(len(facades), model, points[all_near],
                                        np.flatnonzero(all_near), ranges[all_near]))
            
            # 更新 remaining
            for f in facades:
                remaining[np.asarray(f['inlier_indices'], int)] = False
    
    # ===== 细层：在 remaining 点上执行原有逻辑 + 空间化 Hough =====
    hough_seeds = []
    if method in {'hough', 'hybrid'} and remaining.sum() >= min_count:
        hough_seeds = _build_spatial_hough_seeds(
            points, normals, remaining, ranges, tol, base, min_count, voxel_size)
    
    seed_iter = [('hough', s) for s in hough_seeds]
    if method == 'ransac' or (method == 'hybrid' and not seed_iter):
        seed_iter = [('ransac', None)] * 30
    elif method == 'hybrid':
        seed_iter.extend([('ransac', None)] * 8)
    
    fine_facades, remaining = _extract_facades_from_seeds(
        seed_iter, points, normals, remaining, ranges,
        tol, cos_tol, spacing, voxel_size, base,
        min_count, min_facade_area, irls_iters, signed_dist_tolerance)
    
    facades.extend(fine_facades)
    
    # 后处理
    if enable_merge:
        facades = _merge(facades, points,
                         gap_m=float(getattr(Config, 'MERGE_UV_DIST_M', 3.0)),
                         angle_deg=float(getattr(Config, 'MERGE_ANGLE_DEG', 5.0)),
                         d_thresh=float(getattr(Config, 'MERGE_D_THRESH_M', 0.10)),
                         max_plane_dist=base)
    
    facades = _postprocess(facades, points, normals, tol, cos_tol, spacing, voxel_size)
    
    # 构建剩余点云
    used = np.zeros(n, bool)
    for f in facades:
        used[np.asarray(f['inlier_indices'], int)] = True
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(points[~used])
    
    facades.sort(key=lambda f: -f['point_count'])
    for i, f in enumerate(facades):
        f['id'] = i
    
    return {'facades': facades, 'remaining': out, 'total_points': n}


def _build_spatial_hough_seeds(points, normals, mask, ranges, tol, base, 
                               min_count, voxel_size):
    """阶段 1：空间化 Hough 投票。3D 粗网格隔离后，每个空间块独立方向+rho投票。
    
    返回 seed 列表，每个 seed 包含 direction, rho, spatial_ids（空间块内点索引）。
    """
    valid = np.flatnonzero(mask)
    if len(valid) < min_count:
        return []
    
    # 方向投票采样
    max_vote = int(getattr(Config, 'HOUGH_MAX_VOTE_POINTS', 100000))
    hough_valid = valid[::max(1, len(valid) // max_vote)] if len(valid) > max_vote else valid
    
    peaks = normal_hough_peaks(
        normals[hough_valid],
        min_support=max(80, int(len(hough_valid) * 0.002)),
        bin_deg=float(getattr(Config, 'HOUGH_NORMAL_BIN_DEG', 2.0)),
        merge_deg=float(getattr(Config, 'HOUGH_NORMAL_SUPPORT_DEG', 6.0)) - 2.0,  # 收紧
        max_peaks=int(getattr(Config, 'HOUGH_MAX_DIRECTION_PEAKS', 8)),
        vertical_nz=float(getattr(Config, 'VERTICAL_NZ_THR', 0.30)) + 0.05)
    
    hough_seeds = []
    support_deg = float(getattr(Config, 'HOUGH_NORMAL_SUPPORT_DEG', 6.0))
    cos_support = np.cos(np.deg2rad(support_deg))
    
    for peak in peaks:
        direction = np.asarray(peak['normal'], dtype=float)
        
        # 全量法向筛选
        direction_mask = np.abs(normals[valid] @ direction) >= cos_support
        direction_ids = valid[direction_mask]
        
        if len(direction_ids) < min_count:
            continue
        
        # ===== 核心：3D 粗网格连通域分离 =====
        # 粗网格尺寸：max(2m, 40*voxel_size)，足够分离不同建筑/前后墙
        coarse_grid = max(2.0, voxel_size * 40)
        spatial_comps = connected_components_3d_grid(
            points[direction_ids],
            grid_size=coarse_grid,
            min_points=max(min_count // 2, 50))
        
        for sp_mask in spatial_comps:
            sp_ids = direction_ids[sp_mask]
            if len(sp_ids) < min_count // 2:
                continue
            
            # rho 投票：只在空间块内，bin_size 自适应
            sp_ranges = ranges[sp_ids] if len(ranges) else np.zeros(len(sp_ids))
            adaptive_bin = max(float(voxel_size),
                              float(np.median(tol[sp_ids])) * 0.8)
            
            rho_peaks = rho_hough_peaks(
                points[sp_ids], direction,
                bin_size=adaptive_bin,
                min_support=max(50, min_count // 3),
                min_peak_distance=max(0.30, base * 3),
                prominence_ratio=0.08,  # 放宽：大立面 rho 分布平缓
                max_peaks=int(getattr(Config, 'HOUGH_MAX_RHO_PEAKS_PER_DIRECTION', 16)))
            
            for rho_peak in rho_peaks:
                hough_seeds.append({
                    'direction': direction,
                    'rho': rho_peak.value,
                    'spatial_ids': sp_ids,
                    'support': rho_peak.support
                })
    
    # 按支持度降序，截断
    hough_seeds.sort(key=lambda s: -s['support'])
    max_candidates = int(getattr(Config, 'HOUGH_MAX_CANDIDATES', 16))
    return hough_seeds[:max_candidates]