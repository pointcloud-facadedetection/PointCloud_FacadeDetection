"""距离自适应立面检测：RANSAC 种子 → 有符号距离 → UV 连通域 → 后处理。

该模块只输出当前输入点云空间的局部索引，FacadeService 负责把它们归一化
到 proxy_global，因此不会改变 VoxelCascadeIndex 的映射契约。
"""
from __future__ import annotations

import copy
import numpy as np
import open3d as o3d

from algorithms.geometry import (ensure_normals, classify_plane, project_to_uv,
    connected_components_2d_grid, estimate_plane_area, plane_axes,
    estimate_point_ranges, adaptive_plane_tolerance, fit_plane_weighted)
from config.settings import Config


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


def _uv_cells_for_facade(f, points, grid):
    """返回立面在 UV 平面上的占用格集合。"""
    pm = np.asarray(f['plane_model'], dtype=float)
    n = pm[:3] / (np.linalg.norm(pm[:3]) + 1e-12)
    typ, _, _, _ = classify_plane(n)
    idx = np.asarray(f['inlier_indices'], dtype=int)
    uv, _, _, _ = project_to_uv(points[idx], pm, typ)
    return set(zip(np.floor(uv[:, 0] / grid).astype(np.int64),
                   np.floor(uv[:, 1] / grid).astype(np.int64)))


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


def _merge(facades, points, gap_m=3.0, angle_deg=5.0, d_thresh=.08, max_plane_dist=0.05):
    """共面且 UV 邻接合并；禁止合并空间分离或深度双峰的立面。"""
    changed = True
    while changed:
        changed = False
        # 按点数降序，大面优先
        facades.sort(key=lambda f: -f['point_count'])
        for i in range(len(facades)):
            for j in range(i + 1, len(facades)):
                a, b = facades[i], facades[j]
                na, nb = np.asarray(a['normal']), np.asarray(b['normal'])
                if a['type'] != b['type'] or abs(na @ nb) < np.cos(np.deg2rad(angle_deg)):
                    continue
                pa, pb = np.asarray(a['plane_model']), np.asarray(b['plane_model'])
                # 互相到对方平面的有符号距离
                ca = np.asarray(a['center']); cb = np.asarray(b['center'])
                off_a = abs(cb @ pa[:3] + pa[3]) / (np.linalg.norm(pa[:3]) + 1e-12)
                off_b = abs(ca @ pb[:3] + pb[3]) / (np.linalg.norm(pb[:3]) + 1e-12)
                if max(off_a, off_b) > d_thresh:
                    continue
                # UV 邻接判定：用立面局部点间距估计网格
                ia = np.asarray(a['inlier_indices'], dtype=int)
                ib = np.asarray(b['inlier_indices'], dtype=int)
                # 禁止构造 NxN 距离矩阵；23003 点会瞬间申请约 12 GiB。
                # spacing 已在检测阶段按最近邻计算，这里使用稳健的局部近邻估计。
                if len(ia) > 1:
                    local_nn = np.asarray(
                        o3d.geometry.PointCloud(
                            o3d.utility.Vector3dVector(points[ia])
                        ).compute_nearest_neighbor_distance(), dtype=float)
                    spacing = max(0.05, float(np.median(local_nn[local_nn > 0]))) if np.any(local_nn > 0) else 0.05
                else:
                    spacing = 0.05
                grid = max(0.10, spacing * 2.0)
                cells_a = _uv_cells_for_facade(a, points, grid)
                cells_b = _uv_cells_for_facade(b, points, grid)
                dilated_a = _dilate_cells(cells_a, max(1, int(round(gap_m / grid))))
                if not cells_b.intersection(dilated_a):
                    continue
                # 深度单峰校验
                merged_pts = points[np.unique(np.r_[ia, ib])]
                avg_d = (off_a + off_b) / 2.0
                test_model = np.array([na[0], na[1], na[2], avg_d])
                if not _check_depth_single_peak(merged_pts, test_model, max_plane_dist):
                    continue
                ids = np.unique(np.r_[ia, ib])
                model = fit_plane_weighted(points[ids], irls_iters=2)
                facades[i] = _info(a['id'], model, points[ids], ids, np.zeros(len(ids)))
                del facades[j]
                changed = True
                break
            if changed: break
    return facades


def _postprocess(facades, points, normals, tol, cos_tol, spacing, min_opening=.5):
    if not facades:
        return facades
    # 完形回收：只吸收同一平面容差带内、法向一致且 UV 邻接的点。
    assigned = np.zeros(len(points), dtype=bool)
    for f in facades: assigned[np.asarray(f['inlier_indices'], dtype=int)] = True
    for f in facades:
        ids = np.asarray(f['inlier_indices'], dtype=int)
        if not len(ids): continue
        n = np.asarray(f['normal'], dtype=float)
        # 计算当前立面 UV 占用格
        spacing_m = max(0.05, float(np.median(spacing[ids])))
        grid = max(0.10, spacing_m * 2.0)
        cells = _uv_cells_for_facade(f, points, grid)
        dilated = _dilate_cells(cells, max(1, int(round(2.0 / grid))))
        cand = np.where(~assigned)[0]
        d = np.abs(points[cand] @ n + f['plane_model'][3])
        ok = (d <= tol[cand] * 1.5) & (np.abs(normals[cand] @ n) >= cos_tol[cand])
        add = cand[ok]
        if len(add):
            # 用 AABB 膨胀约束邻接，避免回收整场景的平行面。
            lo, hi = points[ids].min(0)-max(1.5*np.median(spacing[ids]), .2), points[ids].max(0)+max(1.5*np.median(spacing[ids]), .2)
            add = add[np.all((points[add] >= lo) & (points[add] <= hi), axis=1)]
            # UV 邻接约束：只回收与当前立面 UV 占用格相邻的点
            if len(add):
                pm = np.asarray(f['plane_model'], dtype=float)
                typ, _, _, _ = classify_plane(n)
                uv_add, _, _, _ = project_to_uv(points[add], pm, typ)
                cells_add = list(zip(np.floor(uv_add[:, 0] / grid).astype(np.int64),
                                     np.floor(uv_add[:, 1] / grid).astype(np.int64)))
                # 不使用 np.fromiter(count=...)：部分 numpy/Python 组合在集合
                # 迭代器耗尽时会抛出 ``iterator too short``，而且此处本来就
                # 已由 cells_add 保证长度与 add 一致。
                add = add[np.asarray([c in dilated for c in cells_add], dtype=bool)]
        # 限制回收比例，防止把远处平行面整体吞并
        max_add = max(len(ids) // 3, 5000)
        if len(add) > max_add:
            add = add[np.argsort(np.abs(points[add] @ n + f['plane_model'][3]))[:max_add]]
        if len(add) >= max(10, len(ids)//50):
            all_ids = np.unique(np.r_[ids, add])
            # 回收后重新检查法向/平面一致性，若变化过大则放弃
            new_model = fit_plane_weighted(points[all_ids], irls_iters=2)
            new_n = new_model[:3] / (np.linalg.norm(new_model[:3]) + 1e-12)
            if abs(new_n @ n) >= np.cos(np.deg2rad(5.0)):
                f.update(_info(f['id'], new_model, points[all_ids], all_ids, np.zeros(len(all_ids))))
                assigned[add] = True
    # 结果分类字段，供现有 UI/报告使用。
    for f in facades:
        bb = f.get('bbox_2d') or {}
        span = max((bb.get('u_max', 0)-bb.get('u_min', 0))*(bb.get('v_max', 0)-bb.get('v_min', 0)), 1e-6)
        fill = f['area']/span
        f['fill_ratio'] = round(float(fill), 3)
        f['wall_kind'] = 'fragment' if fill < .25 else ('fenestrated' if fill < .55 else 'solid')
    return facades


def detect_facades_adaptive(pcd, voxel_size=.05, min_facade_area=5., max_plane_dist=None,
                            min_points_ratio=.003, roi_bounds=None, roi_indices=None,
                            enable_merge=True, enable_grow=False, signed_dist_tolerance=1.,
                            range_adaptive=True, scan_origin=None, range_coeff=.0012,
                            normal_relax_deg_per_m=.15, normal_angle_max_deg=15.,
                            irls_iters=2, metadata=None, **_kwargs):
    work = ensure_normals(copy.deepcopy(pcd), voxel_size)
    points, normals = np.asarray(work.points, float), np.asarray(work.normals, float)
    n = len(points)
    if not n: return {'facades': [], 'remaining': work, 'total_points': 0}
    meta = metadata or {}; cfg = meta.get('adaptive_detection', {})
    range_adaptive = bool(range_adaptive or cfg.get('enabled', False))
    if range_adaptive:
        range_coeff = float(cfg.get('range_coeff', range_coeff)); irls_iters = int(cfg.get('irls_iters', irls_iters))
        scan_origin = scan_origin if scan_origin is not None else meta.get('scan_origins')
    ranges = estimate_point_ranges(points, scan_origin) if range_adaptive else np.zeros(n)
    base = float(max_plane_dist or max(voxel_size*1.5, float(getattr(Config, 'DETECT_DIST_TOL_MM', 20))/1000.))
    tol = adaptive_plane_tolerance(ranges, base, range_coeff, base*4.) if range_adaptive else np.full(n, base)
    theta = np.clip(float(getattr(Config, 'FACADE_NORMAL_ANGLE_DEG', 8.))+normal_relax_deg_per_m*ranges, 8., normal_angle_max_deg)
    cos_tol = np.cos(np.deg2rad(theta))
    active = np.ones(n, bool)
    if roi_bounds is not None: active &= np.all((points >= roi_bounds[0]) & (points <= roi_bounds[1]), axis=1)
    if roi_indices is not None:
        m = np.zeros(n, bool); valid = np.asarray(roi_indices, int); m[valid[(valid>=0)&(valid<n)]] = True; active &= m
    min_count = max(80 if roi_indices is not None or roi_bounds is not None else 100, int(n*min_points_ratio), int(getattr(Config, 'CLUSTER_MIN_POINTS', 200)))
    remaining = active.copy(); facades = []; fid = 0
    spacing = np.asarray(work.compute_nearest_neighbor_distance()) if n > 1 else np.zeros(n)
    for _ in range(30):
        ids = np.flatnonzero(remaining)
        if len(ids) < min_count: break
        sub = work.select_by_index(ids.tolist())
        seed_model, seed_local = sub.segment_plane(float(np.median(tol[ids])), 3, 500)
        if len(seed_local) < min_count:
            break
        seed = np.asarray(seed_model, float); sn = seed[:3]/(np.linalg.norm(seed[:3])+1e-12)
        signed = points[ids] @ sn + seed[3]
        candidate = np.abs(signed) <= tol[ids]*1.5
        candidate &= np.abs(normals[ids] @ sn) >= cos_tol[ids]
        cids = ids[candidate]
        if len(cids) < min_count: break
        tight = np.abs(points[cids] @ sn + seed[3]) <= tol[cids]*signed_dist_tolerance
        cids = cids[tight]
        if len(cids) < min_count: break
        ftype, _, _, _ = classify_plane(sn)
        uv, _, _, _ = project_to_uv(points[cids], seed, ftype)
        # 代理点可能因距离分层而变稀，UV 网格必须跟随局部最近邻间距，
        # 否则固定 2*voxel_size 会把正常远场墙面切成大量小组件。
        local_spacing = float(np.median(spacing[cids])) if len(cids) else voxel_size
        uv_grid = max(float(voxel_size) * 2.0, local_spacing * 1.5, 0.05)
        comps = connected_components_2d_grid(
            uv, grid_size=uv_grid,
            min_cells=max(2, int(min_count * .01)),
            close_radius_cells=3)
        if not comps:
            # 网格组件仍不足时，保留整个 signed-distance 候选；它已经经过
            # 法向、平面距离和空间 seed 约束，不会回退到全场景混合。
            comps = [np.ones(len(cids), dtype=bool)]
        consumed = set()
        for mask in sorted(comps, key=lambda x: int(x.sum()), reverse=True):
            comp = cids[mask]
            if len(comp) < min_count: continue
            model = fit_plane_weighted(points[comp], irls_iters=irls_iters)
            dist = np.abs(points[comp] @ model[:3] + model[3])
            keep = dist <= tol[comp]*1.8
            comp = comp[keep]
            if len(comp) < min_count: continue
            info = _info(fid, model, points[comp], comp, ranges[comp])
            if info['type'] == 'vertical_facade' and info['area'] >= min_facade_area:
                facades.append(info); fid += 1; consumed.update(comp.tolist())
        if not consumed:
            break
        remaining[list(consumed)] = False
    if enable_merge: facades = _merge(facades, points, max_plane_dist=base)
    facades = _postprocess(facades, points, normals, tol, cos_tol, spacing)
    used = np.zeros(n, bool)
    for f in facades: used[np.asarray(f['inlier_indices'], int)] = True
    out = o3d.geometry.PointCloud(); out.points = o3d.utility.Vector3dVector(points[~used])
    facades.sort(key=lambda f: -f['point_count'])
    for i, f in enumerate(facades): f['id'] = i
    return {'facades': facades, 'remaining': out, 'total_points': n}