import copy
import numpy as np
import open3d as o3d

from algorithms.geometry import (
    fit_plane_svd, fit_plane_irls, classify_plane, connected_components_3d_grid,
    cluster_normals_direction, statistical_outlier_mask, plane_basis_from_normal,
    project_to_plane, estimate_uv_bbox_area, uv_bbox_distance,
    estimate_plane_area, ensure_normals,
    split_by_depth_histogram, connected_components_2d_grid,
)
from config.settings import Config


def _wall_core_mask(points, plane_model, max_dist=None, core_ratio=0.80):
    """从候选支撑点中保留墙面主深度层。

    不能用 IRLS 单独替代此步骤：窗框和装饰线往往点数足够，会影响基准面。
    用深度主峰的 MAD 做自适应阈值，并保留一个绝对安全上限。
    """
    if len(points) < 8:
        return np.ones(len(points), dtype=bool), {'median': 0.0, 'mad': 0.0}
    model = np.asarray(plane_model, dtype=float)
    n = model[:3] / (np.linalg.norm(model[:3]) + 1e-12)
    signed = points @ n + float(model[3])
    med = float(np.median(signed))
    mad = float(np.median(np.abs(signed - med)))
    robust = max(3.0 * 1.4826 * mad, 1e-6)
    limit = robust
    if max_dist is not None:
        limit = min(limit, float(max_dist) * float(core_ratio))
    # MAD 极小时使用 voxel/检测容差，避免完美平面被误删
    limit = max(limit, min(float(max_dist) * float(core_ratio), 0.002) if max_dist else 0.002)
    mask = np.abs(signed - med) <= limit
    if np.sum(mask) < max(20, int(len(points) * 0.25)):
        mask = np.abs(signed - med) <= max(robust * 1.5, limit)
    return mask, {'median': med, 'mad': mad, 'limit': float(limit),
                  'p95': float(np.percentile(np.abs(signed - med), 95))}


def build_facade_info(facade_id, plane_model, points, original_indices,
                      support_indices=None, extract_core=True, core_dist=None):
    plane_model = np.asarray(plane_model, dtype=float)
    normal = plane_model[:3]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    plane_model[:3] = normal
    plane_model[3] = float(plane_model[3])
    support_indices = np.asarray(original_indices if support_indices is None else support_indices, dtype=int)
    core_mask, depth_stats = _wall_core_mask(points, plane_model, core_dist) if extract_core else (np.ones(len(points), dtype=bool), {})
    core_points = points[core_mask]
    core_indices = np.asarray(original_indices, dtype=int)[core_mask]
    # 核心点不足时保留候选结果，避免小立面因鲁棒筛选消失
    if len(core_points) < 20:
        core_points, core_indices = points, np.asarray(original_indices, dtype=int)
    center = np.mean(core_points, axis=0)
    facade_type, type_label, verticality, horizontality = classify_plane(normal)
    area, bbox_2d = estimate_plane_area(core_points, normal, center, facade_type)

    distances = np.abs(core_points @ normal + plane_model[3])
    flatness_mean = float(np.mean(distances)) if len(distances) else 0.0
    flatness_max = float(np.max(distances)) if len(distances) else 0.0
    flatness_std = float(np.std(distances)) if len(distances) else 0.0

    return {
        'id': int(facade_id),
        'type': facade_type,
        'type_label': type_label,
        'plane_model': [float(x) for x in plane_model],
        'center': [float(x) for x in center],
        'normal': [float(x) for x in normal],
        'area': float(area),
        'point_count': int(len(core_points)),
        'inlier_indices': [int(x) for x in core_indices],
        'core_indices': [int(x) for x in core_indices],
        'measurement_indices': [int(x) for x in core_indices],
        'support_indices': [int(x) for x in support_indices],
        'support_point_count': int(len(support_indices)),
        'core_ratio': float(len(core_indices) / max(len(support_indices), 1)),
        'depth_stats': depth_stats,
        'verticality': verticality,
        'horizontality': horizontality,
        'flatness': flatness_std,
        'flatness_mean': flatness_mean,
        'flatness_max': flatness_max,
        'bbox_2d': bbox_2d
    }

def _merge_vertical_facades(facades: list, points: np.ndarray,
                            merge_angle_deg: float = 5.0,
                            merge_d_thresh: float = 0.08,
                            uv_dist_thresh: float = 2.5) -> list:
    """垂直立面智能合并：法向+d+UV-bbox邻近性"""
    if not facades or len(facades) <= 1:
        return facades

    cos_thr = np.cos(np.deg2rad(merge_angle_deg))
    n = len(facades)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    normals, ds = [], []
    for f in facades:
        nv = np.asarray(f.get('normal') or f['plane_model'][:3], dtype=float)
        nv = nv / (np.linalg.norm(nv) + 1e-12)
        normals.append(nv)
        ds.append(float(np.asarray(f['plane_model'])[3]))

    for i in range(n):
        for j in range(i + 1, n):
            if abs(float(np.dot(normals[i], normals[j]))) < cos_thr:
                continue
            if abs(ds[i] - ds[j]) > merge_d_thresh:
                continue
            if uv_bbox_distance(facades[i], facades[j]) < uv_dist_thresh:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged = []
    for group in groups.values():
        if len(group) == 1:
            merged.append(facades[group[0]])
            continue
        all_idx = []
        all_support = []
        for g in group:
            all_idx.extend(facades[g].get('measurement_indices', facades[g].get('inlier_indices', [])))
            all_support.extend(facades[g].get('support_indices', facades[g].get('inlier_indices', [])))
        idx_arr = np.unique(np.asarray(all_idx, dtype=int))
        idx_arr = idx_arr[(idx_arr >= 0) & (idx_arr < len(points))]
        if len(idx_arr) == 0:
            continue
        pts = points[idx_arr]
        refit = fit_plane_irls(pts, init_model=None, max_iters=3, huber_delta=0.05)
        support_arr = np.unique(np.asarray(all_support, dtype=int))
        support_arr = support_arr[(support_arr >= 0) & (support_arr < len(points))]
        merged.append(build_facade_info(-1, refit, pts, idx_arr,
                                        support_indices=support_arr,
                                        core_dist=merge_d_thresh))

    return merged


# ==================== 垂直立面检测 ====================

def detect_facades(pcd, voxel_size=0.05,
                   min_facade_area=10.0,
                   max_plane_dist=None,
                   min_points_ratio=0.003,
                   normal_angle_deg=None,
                   min_cluster_points=200,
                   roi_bounds=None,
                   roi_indices=None,
                   enable_merge=True,
                   enable_grow=True,
                   signed_dist_tolerance=2.0):
    """
    垂直立面检测

    流程：
        1. 法向预聚类（全局）→ 找出所有主方向
        2. 垂直筛选 → 只保留 |nz| <= 0.20 的簇
        3. 3D空间连通域 → 分离空间不相连的立面（解决误合并）
        4. 直接拟合 → SVD初值 + IRLS精修（无层层过滤）
        5. UV面积验证 → 排除碎片
        6. 智能合并 → 同一平面被洞口切开的碎片合并
        7. 二次生长 → 用最终平面在全局吸收遗漏点
        8. 最终精修 → 完整点集上统一IRLS
    """
    # ensure_normals 在需要时自行复制；这里不再提前 deepcopy，才能复用输入法向。
    pcd_work = ensure_normals(pcd, voxel_size)
    points = np.asarray(pcd_work.points)
    normals = np.asarray(pcd_work.normals)
    total_points = len(points)
    if total_points == 0:
        return {'facades': [], 'remaining': pcd_work, 'total_points': 0}

    # ==================== 参数自适应 ====================
    tol_m_from_mm = float(getattr(Config, 'DETECT_DIST_TOL_MM', 20.0)) / 1000.0
    max_plane_dist = max_plane_dist or max(float(voxel_size) * 2.0, tol_m_from_mm)

    if normal_angle_deg is None:
        try:
            normal_angle_deg = float(Config.FACADE_NORMAL_ANGLE_DEG)
        except Exception:
            normal_angle_deg = 10.0

    try:
        VERTICAL_NZ_THR = float(getattr(Config, 'VERTICAL_NZ_THR', 0.20))
    except Exception:
        VERTICAL_NZ_THR = 0.20  # |nz| <= 0.20 为垂直立面
    min_remaining = max(100, int(total_points * min_points_ratio))
    min_cluster_points = max(int(min_cluster_points), min_remaining)

    is_roi = (roi_bounds is not None) or (roi_indices is not None)
    if is_roi:
        # ROI 模式：适度放宽点数阈值，但保持面积硬下限 >= Config.MIN_FACADE_AREA
        try:
            min_area_conf = float(getattr(Config, 'MIN_FACADE_AREA', 10.0))
        except Exception:
            min_area_conf = 10.0
        min_facade_area = max(float(min_facade_area), min_area_conf)
        min_cluster_points = max(int(min_cluster_points * 0.6), 80)
        min_remaining = max(int(min_remaining * 0.6), 50)

    # ==================== ROI掩码 ====================
    active_mask = np.ones(total_points, dtype=bool)
    if roi_bounds is not None:
        try:
            bmin = np.asarray(roi_bounds[0], dtype=float)
            bmax = np.asarray(roi_bounds[1], dtype=float)
            active_mask &= np.all((points >= bmin) & (points <= bmax), axis=1)
        except Exception:
            pass
    if roi_indices is not None:
        m = np.zeros(total_points, dtype=bool)
        valid_roi = np.asarray(roi_indices, dtype=int)
        valid_roi = valid_roi[(valid_roi >= 0) & (valid_roi < total_points)]
        m[valid_roi] = True
        active_mask &= m

    # ROI统计滤波（仅ROI模式）
    if is_roi and np.sum(active_mask) > min_cluster_points:
        roi_pts = points[active_mask]
        roi_local_idx = np.where(active_mask)[0]
        clean_mask = statistical_outlier_mask(roi_pts, k=20, std_ratio=2.0)
        if np.sum(clean_mask) >= min_cluster_points:
            active_mask[:] = False
            active_mask[roi_local_idx[clean_mask]] = True

    # ==================== Step 1: 法向预聚类（全局） ====================
    # 只在active点中聚类，避免噪点干扰
    active_idx = np.where(active_mask)[0]
    if len(active_idx) < min_cluster_points:
        return {'facades': [], 'remaining': pcd_work, 'total_points': total_points}

    active_points = points[active_idx]
    active_normals = normals[active_idx]

    try:
        clusters = cluster_normals_direction(
            active_normals,
            angle_threshold_deg=max(normal_angle_deg, 8.0)
        )
    except Exception:
        clusters = []

    if not clusters:
        return {'facades': [], 'remaining': pcd_work, 'total_points': total_points}

    # ==================== Step 2: 垂直筛选 + 3D连通域 + 拟合 ====================
    facades = []
    facade_id = 0
    consumed_global = np.zeros(total_points, dtype=bool)  # 已被占用的全局索引

    for cl_mask in clusters:
        cl_count = int(np.sum(cl_mask))
        if cl_count < min_cluster_points:
            continue

        # 该簇的点（在active_idx坐标系下）
        cl_local_idx = np.where(cl_mask)[0]
        cl_points = active_points[cl_local_idx]
        cl_normals = active_normals[cl_local_idx]

        # 判断主方向是否垂直：用簇中法向均值
        mean_normal = np.mean(cl_normals, axis=0)
        mean_normal = mean_normal / (np.linalg.norm(mean_normal) + 1e-12)
        if abs(mean_normal[2]) > VERTICAL_NZ_THR:
            continue  # 非垂直方向，跳过

        # 3D连通域：分离空间上不相连的立面（不同建筑/房间）
        comp3d_list = connected_components_3d_grid(
            cl_points,
            grid_size=max(voxel_size * 2.0, 0.10),
            min_points=max(min_cluster_points // 2, 50)
        )
        if not comp3d_list:
            comp3d_list = [np.ones(len(cl_points), dtype=bool)]

        for comp3d_mask in sorted(comp3d_list, key=lambda m: np.sum(m), reverse=True):
            if np.sum(comp3d_mask) < min_cluster_points:
                continue

            comp_local = cl_local_idx[comp3d_mask]
            comp_pts = cl_points[comp3d_mask]
            comp_global = active_idx[comp_local]  # 映射回原始点云索引

            # 检查是否已被其他立面占用（避免重叠）
            overlap = int(np.count_nonzero(consumed_global[comp_global]))
            if overlap > len(comp_global) * 0.7:
                continue

            # ==================== Step 4: 直接拟合（无层层过滤）====================
            # SVD初值
            init_model = fit_plane_svd(comp_pts)
            init_normal = init_model[:3] / (np.linalg.norm(init_model[:3]) + 1e-12)

            # 垂直校验
            if abs(init_normal[2]) > VERTICAL_NZ_THR:
                continue

            # IRLS精修一次（Huber 0.6×）
            refined = fit_plane_irls(
                comp_pts,
                init_model=init_model,
                max_iters=3,
                huber_delta=max_plane_dist * 0.6
            )
            rn = refined[:3] / (np.linalg.norm(refined[:3]) + 1e-12)
            rd = float(refined[3])

            # 宽松内点：2.0×max_plane_dist（保留完整立面边缘）
            dists = np.abs(comp_pts @ rn + rd)
            inlier_mask = dists <= max_plane_dist * signed_dist_tolerance
            if np.sum(inlier_mask) < min_cluster_points:
                continue

            inlier_pts = comp_pts[inlier_mask]
            inlier_global = comp_global[inlier_mask]

            # 剥离凸出/凹进结构：沿法向深度直方图分割，选择 UV 面积最大的主墙片层
            try:
                depth_masks = split_by_depth_histogram(inlier_pts, refined, max_plane_dist, min_points=max(80, min_cluster_points // 4))
                if depth_masks and len(depth_masks) > 1:
                    best_i = 0
                    best_area = -1.0
                    for i, mk in enumerate(depth_masks):
                        sub = inlier_pts[mk]
                        center = np.mean(sub, axis=0)
                        u_vec, v_vec = plane_basis_from_normal(refined[:3])
                        uv = project_to_plane(sub, center, u_vec, v_vec)
                        uv_area = float(estimate_uv_bbox_area(uv))
                        if uv_area > best_area:
                            best_area = uv_area
                            best_i = i
                    mk = depth_masks[best_i]
                    inlier_pts = inlier_pts[mk]
                    inlier_global = inlier_global[mk]
            except Exception:
                pass

            # 在主墙片层上重新精修（最终平面参数）
            final_model = fit_plane_irls(
                inlier_pts,
                init_model=refined,
                max_iters=3,
                huber_delta=max_plane_dist * 0.6
            )

            # ==================== Step 5: UV面积验证 ====================
            center = np.mean(inlier_pts, axis=0)
            u_vec, v_vec = plane_basis_from_normal(final_model[:3])
            uv = project_to_plane(inlier_pts, center, u_vec, v_vec)
            uv_area = estimate_uv_bbox_area(uv)

            # 快速排除：UV面积不足或点云面积不足
            if uv_area < min_facade_area * 0.5:
                continue

            info = build_facade_info(
                facade_id, final_model, inlier_pts, inlier_global,
                support_indices=comp_global,
                core_dist=max_plane_dist * 0.6
            )
            if info['type'] != 'vertical_facade' or info['area'] < max(min_facade_area, float(getattr(Config, 'MIN_FACADE_AREA', 10.0))):
                continue

            facades.append(info)
            facade_id += 1
            consumed_global[inlier_global] = True

    # ==================== Step 6: 智能合并（同一立面被洞口切开）====================
    if enable_merge and len(facades) > 1:
        facades = _merge_vertical_facades(
            facades, points,
            merge_angle_deg=float(getattr(Config, 'FACADE_MERGE_ANGLE_DEG', 5.0)),
            merge_d_thresh=float(getattr(Config, 'FACADE_MERGE_D_THRESH', 0.08)),
            uv_dist_thresh=5.0  # 5m内UV投影邻近则合并（跨越门窗洞口）
        )

    # ==================== Step 7: 二次生长（全局吸收遗漏点）====================
    if enable_grow:
        grown = []
        for facade in facades:
            if facade['type'] != 'vertical_facade':
                continue

            nvec = np.asarray(facade['plane_model'][:3], dtype=float)
            nvec = nvec / (np.linalg.norm(nvec) + 1e-12)
            d = float(facade['plane_model'][3])

            remaining_mask = ~consumed_global.copy()
            
            # 【新增】若存在roi_bounds，限制生长范围
            if roi_bounds is not None:
                try:
                    bmin = np.asarray(roi_bounds[0], dtype=float)
                    bmax = np.asarray(roi_bounds[1], dtype=float)
                    in_bounds = np.all((points >= bmin) & (points <= bmax), axis=1)
                    remaining_mask &= in_bounds
                except Exception:
                    pass
            elif is_roi and roi_indices is not None:
                # 仅有indices时，保守生长：只允许吸收indices内的点
                m = np.zeros(total_points, dtype=bool)
                valid_roi = np.asarray(roi_indices, dtype=int)
                valid_roi = valid_roi[(valid_roi >= 0) & (valid_roi < total_points)]
                m[valid_roi] = True
                remaining_mask &= m

            # 法向点积随当前平面变化；保持原有距离和法向阈值语义。
            all_dists = np.abs(points @ nvec + d)
            normal_agree = np.abs(normals @ nvec) >= np.cos(np.deg2rad(5.0))
            grow_mask = (all_dists <= max_plane_dist * 2.0) & normal_agree & remaining_mask

            original_idx = np.asarray(facade.get('measurement_indices', facade.get('inlier_indices', [])), dtype=int)
            support_idx = np.asarray(facade.get('support_indices', original_idx), dtype=int)
            grow_idx = np.where(grow_mask)[0]
            # 生长点只作为支撑范围；最终核心层重新从深度主峰提取
            support_combined = np.unique(np.concatenate((support_idx, grow_idx)))
            combined = original_idx

            if len(support_combined) < min_cluster_points:
                continue

            combined_pts = points[support_combined]
            # 最终全局精修（一次）
            final_model = fit_plane_irls(
                points[combined],
                init_model=np.asarray(facade['plane_model']),
                max_iters=3,
                huber_delta=max_plane_dist * 0.6
            )

            # 在 UV 平面上进行连通域桥接，确保跨门窗形成“大连通片”
            try:
                center = np.mean(combined_pts, axis=0)
                u_vec, v_vec = plane_basis_from_normal(final_model[:3])
                uv = project_to_plane(combined_pts, center, u_vec, v_vec)
                close_cells = int(getattr(Config, 'UV_CLOSE_RADIUS_CELLS', 1))
                comps = connected_components_2d_grid(uv, grid_size=None, min_cells=3,
                                                     adaptive_ratio=2.5, sample_ratio=0.1,
                                                     connectivity=8, close_radius_cells=close_cells)
                if comps:
                    # 选取最大连通片
                    sizes = [int(np.sum(mk)) for mk in comps]
                    mk = comps[int(np.argmax(sizes))]
                    support_combined = support_combined[mk]
                    combined_pts = points[support_combined]
                    # 精修一次
                    final_model = fit_plane_irls(
                        combined_pts,
                        init_model=final_model,
                        max_iters=3,
                        huber_delta=max_plane_dist * 0.6
                    )
            except Exception:
                pass

            info = build_facade_info(facade['id'], final_model, combined_pts,
                                     support_combined, support_indices=support_combined,
                                     core_dist=max_plane_dist * 0.6)
            if info['type'] == 'vertical_facade' and info['area'] >= max(min_facade_area, float(getattr(Config, 'MIN_FACADE_AREA', 10.0))):
                grown.append(info)
                consumed_global[np.asarray(info['inlier_indices'], dtype=int)] = True

        facades = grown

    # 最终合并（生长后可能产生新的可合并碎片）
    if enable_merge and len(facades) > 1:
        facades = _merge_vertical_facades(
            facades, points,
            merge_angle_deg=float(getattr(Config, 'FACADE_MERGE_ANGLE_DEG', 5.0)),
            merge_d_thresh=float(getattr(Config, 'FACADE_MERGE_D_THRESH', 0.08)),
            uv_dist_thresh=3.0
        )

    # ==================== 构造剩余点云 ====================
    final_consumed = np.zeros(total_points, dtype=bool)
    for f in facades:
        final_consumed[f['inlier_indices']] = True
    # ROI 模式：仅输出 ROI 内的“剩余点”，全局模式保持原逻辑
    if is_roi:
        remaining_mask = (~final_consumed) & active_mask
    else:
        remaining_mask = ~final_consumed

    try:
        remaining_pcd = o3d.geometry.PointCloud()
        remaining_pcd.points = o3d.utility.Vector3dVector(points[remaining_mask])
        if pcd_work.has_colors():
            cols = np.asarray(pcd_work.colors)
            if len(cols) == total_points:
                remaining_pcd.colors = o3d.utility.Vector3dVector(cols[remaining_mask])
        if pcd_work.has_normals():
            nrm = np.asarray(pcd_work.normals)
            if len(nrm) == total_points:
                remaining_pcd.normals = o3d.utility.Vector3dVector(nrm[remaining_mask])
    except Exception:
        remaining_pcd = pcd_work

    facades.sort(key=lambda f: (f['type'] != 'vertical_facade', -f['point_count']))
    for i, facade in enumerate(facades):
        facade['id'] = i

    return {
        'facades': facades,
        'remaining': remaining_pcd,
        'total_points': total_points
    }