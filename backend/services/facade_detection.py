import copy
import numpy as np

from ..core.geometry_utils import (
    fit_plane_svd, classify_plane, project_to_uv, connected_components_2d_grid,
    estimate_plane_area, ensure_normals,
    estimate_point_ranges, range_noise_sigma, adaptive_plane_tolerance,
    fit_plane_weighted, resolve_scan_origin
)
from .facade_postprocess import postprocess_facades
from ..config import Config


def build_facade_info(facade_id, plane_model, points, original_indices, ranges=None,
                      occupancy_cell=None):
    """
    构建立面信息字典

    occupancy_cell: UV占用格边长（米）。给定时 area 采用占用格面积
    （占用格数×格面积），并输出 occupancy_ratio=占用面积/凸包面积。
    凸包面积对"稀疏共面杂点被串成的伪平面"会虚高几个数量级，
    占用面积才反映真实墙面大小；占用率极低即为筛子状伪平面。
    """
    plane_model = np.asarray(plane_model, dtype=float)
    normal = plane_model[:3]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    plane_model[:3] = normal
    plane_model[3] = float(plane_model[3])
    center = np.mean(points, axis=0)
    facade_type, type_label, verticality, horizontality = classify_plane(normal)
    area_hull, bbox_2d = estimate_plane_area(points, normal, center, facade_type)

    area = area_hull
    isolation = None
    if occupancy_cell and len(points) >= 3:
        cell = float(occupancy_cell)
        uv, _, _, _ = project_to_uv(points, plane_model, facade_type)
        cells = set(zip(np.floor(uv[:, 0] / cell).astype(np.int64),
                        np.floor(uv[:, 1] / cell).astype(np.int64)))
        area = float(len(cells)) * cell * cell
        # 孤立度：占用格数/点数。真实墙面连续采样、多点共格(~0.2-0.4)；
        # 散落共面杂点串成的伪平面每点独占一格(~1.0)
        isolation = float(len(cells)) / float(len(points))

    distances = np.abs(points @ normal + plane_model[3])
    flatness_mean = float(np.mean(distances)) if len(distances) else 0.0
    flatness_rmse = float(np.sqrt(np.mean(distances ** 2))) if len(distances) else 0.0
    flatness_max = float(np.max(distances)) if len(distances) else 0.0
    flatness_std = float(np.std(distances)) if len(distances) else 0.0

    info = {
        'id': int(facade_id),
        'type': facade_type,
        'type_label': type_label,
        'plane_model': [float(x) for x in plane_model],
        'center': [float(x) for x in center],
        'normal': [float(x) for x in normal],
        'area': float(area),
        'area_hull': float(area_hull),
        'point_count': int(len(points)),
        'inlier_indices': [int(x) for x in original_indices],
        'verticality': verticality,
        'horizontality': horizontality,
        'flatness': flatness_std,
        'flatness_mean': flatness_mean,
        'flatness_rmse': flatness_rmse,
        'flatness_max': flatness_max,
        'bbox_2d': bbox_2d
    }
    if isolation is not None:
        info['isolation'] = isolation
    # 距离信息：用于评估该立面结果的可信度（远处密度低、噪声大）
    if ranges is not None and len(ranges):
        info['mean_range'] = float(np.mean(ranges))
        info['max_range'] = float(np.max(ranges))
    return info


def detect_facades(pcd, voxel_size=0.05,
                   min_facade_area=5.0,
                   max_plane_dist=None,
                   min_points_ratio=0.003,
                   normal_angle_deg=5.0,
                   dbscan_eps=None,
                   min_cluster_points=200,
                   # 新增：UV-2D连通域参数
                   use_uv_2d_clustering=True,  # 启用UV-2D
                   uv_grid_size=None,  # UV网格大小，默认 voxel_size * 2
                   signed_dist_tolerance=1.0,  # 有符号距离容差倍数（相对于max_plane_dist）
                   min_cells_ratio=0.015,  # 最小连通域占点比例
                   # 新增：距离自适应参数（单站TLS远处密度低、噪声大）
                   range_adaptive=True,  # False 时完全回退旧固定阈值行为
                   scan_origin=None,  # 扫描仪位置；单站数据默认坐标原点
                   range_coeff=0.0012,  # 平面容差随距离增长率（米/米）
                   normal_relax_deg_per_m=0.15,  # 法向门槛随距离放宽（度/米）
                   normal_angle_max_deg=15.0,  # 法向门槛放宽上限
                   irls_iters=2,  # 加权平面拟合IRLS迭代次数
                   bridge_spacing_factor=1.6,  # UV桥接半径 = 局部点间距 × 该系数
                   max_bridge_cells=6,  # UV桥接半径上限（格）
                   max_isolation=0.9,  # 孤立度上限（占用格数/点数），筛除稀疏伪平面
                   max_facade_span_m=80.0,  # 单个立面横向跨度上限，拦截横贯多栋楼的伪平面
                   merge_pass=True,  # 后处理：共面且UV相邻的检出面合并
                   completion_pass=True,  # 后处理：把平面容差带内的未归入邻接点回收进检出面
                   merge_gap_m=3.0,  # 合并邻接可跨越的最大间隙（窗洞尺度），墙垛条带并回整墙
                   min_opening_area=0.5):  # 洞口最小面积（m²），过滤采样空隙
    """
    检测点云中的主要建筑平面

    1. 候选点先用【有符号距离】精筛，排除前后深度不同的平行面
    2. 剩余点投影到UV平面，做2D连通域分析（网格BFS或2D-DBSCAN）
    3. 每个2D连通域独立拟合平面，避免深度不一致平面被合并

    距离自适应（range_adaptive=True）：单站扫描点云的密度与精度随距离衰减，
    固定阈值会系统性丢弃/误判远处点。开启后：
    - 平面距离容差 τ(r) 随扫描距离线性放宽
    - 法向一致性门槛随距离放宽（远处法向估计噪声大）
    - 平面拟合按 1/σ(r)² 加权 + IRLS，近处高精度点主导平面参数
    - UV连通域按局部点间距桥接，远处稀疏区不再碎片化
    """
    pcd_work = ensure_normals(copy.deepcopy(pcd), voxel_size)
    points = np.asarray(pcd_work.points)
    normals = np.asarray(pcd_work.normals)
    total_points = len(points)
    if total_points == 0:
        return {'facades': [], 'remaining': pcd_work, 'total_points': 0}

    max_plane_dist = max_plane_dist or max(float(voxel_size) * 1.5, 0.03)
    normal_cos = float(np.cos(np.deg2rad(normal_angle_deg)))
    dbscan_eps = dbscan_eps or max(float(voxel_size) * 6.0, 0.35)
    min_remaining = max(100, int(total_points * min_points_ratio))
    # 比例阈值封顶600：总点数被近场主导时，远处目标(几百~几千点)会被比例阈值全部挡掉
    min_cluster_points = max(int(min_cluster_points), min(min_remaining, 600))

    # UV网格参数
    uv_grid_size = uv_grid_size or max(float(voxel_size) * 2.0, 0.1)
    min_cells = max(2, int(min_cluster_points * min_cells_ratio))

    # ---- 预计算：局部点间距（孤立度门槛两种模式都要用）----
    if total_points >= 2:
        spacing_all = np.asarray(pcd_work.compute_nearest_neighbor_distance())
    else:
        spacing_all = np.zeros(total_points)

    # ---- 距离自适应预计算（对整个工作点云一次性完成）----
    if range_adaptive:
        ranges_all = estimate_point_ranges(
            points, resolve_scan_origin(points, scan_origin))
        sigma_all = range_noise_sigma(ranges_all)
        fit_weight_all = 1.0 / np.maximum(sigma_all, 1e-4) ** 2
    else:
        ranges_all = fit_weight_all = None

    def _refit_plane(comp_points, original_idx):
        if range_adaptive:
            return fit_plane_weighted(comp_points, fit_weight_all[original_idx],
                                      irls_iters=irls_iters)
        return fit_plane_svd(comp_points)

    def _occupancy_cell(original_idx):
        # 占用格边长随该组件的局部点间距放大，远处稀疏墙面不被低估面积
        if len(original_idx):
            return max(uv_grid_size, 2.0 * float(np.median(spacing_all[original_idx])))
        return uv_grid_size

    def _facade_ok(info):
        if info['area'] < min_facade_area:
            return False
        # 孤立度门槛只作用于竖直立面：跨场景怪物平面只在竖直方向出现
        # （各楼层楼板高度不同不会共面）；地面等水平面斜掠采样各向异性强，
        # 孤立度天然偏高，不适用该判据
        if info['type'] != 'vertical_facade':
            return True
        isolation = info.get('isolation')
        if isolation is not None and isolation > max_isolation:
            return False
        bbox = info.get('bbox_2d') or {}
        if bbox:
            u_span = float(bbox['u_max']) - float(bbox['u_min'])
            if u_span > max_facade_span_m:
                return False
        return True

    def _relax_cluster_floor():
        # 大平面收完后逐级降低点数门槛，继续挖较小的墙
        nonlocal min_cluster_points, min_cells
        if min_cluster_points <= 250:
            return False
        min_cluster_points = max(250, min_cluster_points // 2)
        min_cells = max(2, int(min_cluster_points * min_cells_ratio))
        return True


    def _rebuild(fid, all_idx):
        model = _refit_plane(points[all_idx], all_idx)
        return build_facade_info(
            fid, model, points[all_idx], all_idx,
            ranges=ranges_all[all_idx] if range_adaptive else None,
            occupancy_cell=_occupancy_cell(all_idx))

    remaining_indices = np.arange(total_points, dtype=int)
    remaining = pcd_work
    facades = []
    facade_id = 0

    for _ in range(Config.FACADE_MAX_ITERATIONS):
        if len(remaining.points) < min_remaining:
            break

        # 每点平面距离容差：自适应时 τ(r)，否则固定 max_plane_dist
        if range_adaptive:
            rem_ranges = ranges_all[remaining_indices]
            # 上限4倍基础容差：即使测站估计异常也不会无限放宽
            tol = adaptive_plane_tolerance(rem_ranges, max_plane_dist, range_coeff,
                                           max_tol=max_plane_dist * 4.0)
            # 种子RANSAC只接受标量阈值，用容差中位数，远处平面也能拿到足量种子内点
            seed_thr = float(np.clip(np.median(tol), max_plane_dist, max_plane_dist * 3.0))
        else:
            rem_ranges = None
            tol = np.full(len(remaining.points), max_plane_dist, dtype=float)
            seed_thr = max_plane_dist

        # ---- Step 1: RANSAC种子平面 ----
        plane_model, seed_inliers = remaining.segment_plane(
            distance_threshold=seed_thr,
            ransac_n=3,
            num_iterations=500
        )
        if len(seed_inliers) < max(min_cluster_points, 100):
            if _relax_cluster_floor():
                continue
            break

        plane_model = np.asarray(plane_model, dtype=float)
        normal = plane_model[:3]
        normal = normal / (np.linalg.norm(normal) + 1e-12)

        # ---- Step 2: 候选点筛选（法向一致性 + 绝对距离粗筛）----
        rem_points = np.asarray(remaining.points)
        rem_normals = np.asarray(remaining.normals)
        distances = np.abs(rem_points @ normal + plane_model[3])
        if range_adaptive:
            # 远处法向估计噪声大，门槛随距离线性放宽
            theta = np.clip(normal_angle_deg + normal_relax_deg_per_m * rem_ranges,
                            normal_angle_deg, normal_angle_max_deg)
            normal_agree = np.abs(rem_normals @ normal) >= np.cos(np.deg2rad(theta))
        else:
            normal_agree = np.abs(rem_normals @ normal) >= normal_cos

        # 绝对距离放宽到1.5倍，捕获候选
        candidate_mask = (distances <= tol * 1.5) & normal_agree
        candidate_local = np.where(candidate_mask)[0]

        if len(candidate_local) < min_remaining:
            candidate_local = np.asarray(seed_inliers, dtype=int)

        # ---- Step 3: 有符号距离精筛 ----
        # 只保留紧贴种子平面的点，排除前后深度不同的平行平面
        signed_dists = rem_points[candidate_local] @ normal + float(plane_model[3])
        tight_mask = np.abs(signed_dists) <= tol[candidate_local] * signed_dist_tolerance
        tight_local = candidate_local[tight_mask]

        # 如果精筛后点太少，回退到粗筛结果
        if len(tight_local) < min_cluster_points:
            tight_local = candidate_local

        # ==================== UV-2D连通域 ====================
        if use_uv_2d_clustering and len(tight_local) >= min_cluster_points:
            candidate_points = rem_points[tight_local]

            # 预分类平面类型（用于UV轴选择）
            facade_type, _, _, _ = classify_plane(normal)

            # 投影到UV平面
            uv_coords, u_axis, v_axis, uv_center = project_to_uv(
                candidate_points, plane_model, facade_type
            )

            # 2D连通域分析
            if len(uv_coords) >= min_cluster_points:
                # 桥接半径：远处点间距大于UV网格时按局部点间距成比例桥接
                bridge_m = None
                if range_adaptive:
                    bridge_m = bridge_spacing_factor * spacing_all[
                        remaining_indices[tight_local]]

                # UV网格随候选集局部点间距放大：远处条纹状稀疏采样(0.3~0.7m)
                # 在固定0.1m网格下永远连不通
                grid_used = uv_grid_size
                if range_adaptive and len(tight_local):
                    grid_used = max(uv_grid_size, 1.5 * float(np.median(
                        spacing_all[remaining_indices[tight_local]])))

                # 使用网格BFS
                components = connected_components_2d_grid(
                    uv_coords,
                    grid_size=grid_used,
                    min_cells=min_cells,
                    point_bridge_m=bridge_m,
                    max_bridge_cells=max_bridge_cells
                )

                if len(components) > 0:
                    consumed_local = set()

                    for comp_mask in sorted(components, key=lambda m: np.sum(m), reverse=True):
                        if np.sum(comp_mask) < min_cluster_points:
                            continue

                        # 该连通域的原始3D点索引
                        comp_local = tight_local[comp_mask]
                        comp_points = rem_points[comp_local]

                        # 精修平面（自适应时加权SVD+IRLS，近处高精度点主导）
                        comp_orig = remaining_indices[comp_local]
                        refined_model = _refit_plane(comp_points, comp_orig)
                        comp_normal = refined_model[:3]
                        refined_dist = np.abs(comp_points @ comp_normal + refined_model[3])

                        # 精修后再次用距离容差过滤离群点
                        keep = refined_dist <= tol[comp_local] * 1.8
                        if np.sum(keep) < min_cluster_points:
                            continue

                        comp_local = comp_local[keep]
                        comp_points = rem_points[comp_local]
                        comp_orig = remaining_indices[comp_local]
                        refined_model = _refit_plane(comp_points, comp_orig)

                        # 构建立面信息
                        original_indices = comp_orig
                        info = build_facade_info(
                            facade_id, refined_model, comp_points, original_indices,
                            ranges=ranges_all[comp_orig] if range_adaptive else None,
                            occupancy_cell=_occupancy_cell(comp_orig))

                        if not _facade_ok(info):
                            consumed_local.update(int(i) for i in comp_local)
                            continue

                        facades.append(info)
                        facade_id += 1
                        consumed_local.update(int(i) for i in comp_local)

                    # 移除已消耗点，进入下一轮
                    if consumed_local:
                        keep_mask = np.ones(len(remaining.points), dtype=bool)
                        keep_mask[list(consumed_local)] = False
                        remaining_indices = remaining_indices[keep_mask]
                        remaining = remaining.select_by_index(np.where(keep_mask)[0].tolist())
                        continue
        # =========================================================

        # ---- Fallback: 3D-DBSCAN逻辑----
        # 自适应时 eps 不小于候选点局部点间距的4倍，远处稀疏簇不被拆散；
        # 上限1.2m，防止跨建筑链接出横贯场景的伪簇
        eps_used = dbscan_eps
        if range_adaptive and len(candidate_local):
            cand_spacing = spacing_all[remaining_indices[candidate_local]]
            eps_used = min(max(dbscan_eps, 4.0 * float(np.median(cand_spacing))), 1.2)
        candidate_pcd = remaining.select_by_index(candidate_local.tolist())
        labels = np.asarray(candidate_pcd.cluster_dbscan(
            eps=eps_used,
            min_points=max(12, min_cluster_points // 8),
            print_progress=False
        ))
        # 注意：不再把"全体候选点"当成一个簇兜底——稀疏场景下 DBSCAN 全判噪声时，
        # 那个兜底会把横跨整个场景的共面杂点拟合成一个巨大伪立面
        clusters = []
        if labels.size and labels.max() >= 0:
            for label in range(int(labels.max()) + 1):
                local_cluster = candidate_local[labels == label]
                if len(local_cluster) >= min_cluster_points:
                    clusters.append(local_cluster)

        consumed_local = set(int(i) for i in seed_inliers)
        for local_cluster in sorted(clusters, key=len, reverse=True):
            cluster_points = rem_points[local_cluster]
            cluster_orig = remaining_indices[local_cluster]
            refined_model = _refit_plane(cluster_points, cluster_orig)
            cluster_normal = refined_model[:3]
            refined_dist = np.abs(cluster_points @ cluster_normal + refined_model[3])
            keep = refined_dist <= tol[local_cluster] * 1.8
            if np.sum(keep) < min_cluster_points:
                continue
            local_cluster = local_cluster[keep]
            cluster_points = rem_points[local_cluster]
            cluster_orig = remaining_indices[local_cluster]
            refined_model = _refit_plane(cluster_points, cluster_orig)
            original_indices = cluster_orig
            info = build_facade_info(
                facade_id, refined_model, cluster_points, original_indices,
                ranges=ranges_all[cluster_orig] if range_adaptive else None,
                occupancy_cell=_occupancy_cell(cluster_orig))
            if not _facade_ok(info):
                consumed_local.update(int(i) for i in local_cluster)
                continue
            facades.append(info)
            facade_id += 1
            consumed_local.update(int(i) for i in local_cluster)

        if not consumed_local:
            if _relax_cluster_floor():
                continue
            break
        keep_mask = np.ones(len(remaining.points), dtype=bool)
        keep_mask[list(consumed_local)] = False
        remaining_indices = remaining_indices[keep_mask]
        remaining = remaining.select_by_index(np.where(keep_mask)[0].tolist())

    # ---- 后处理：共面合并 / 完形回收 / 洞口提取 / 墙体分类（facade_postprocess）----
    if facades:
        if range_adaptive:
            tol_full = adaptive_plane_tolerance(ranges_all, max_plane_dist, range_coeff,
                                                max_tol=max_plane_dist * 4.0)
            theta_full = np.clip(normal_angle_deg + normal_relax_deg_per_m * ranges_all,
                                 normal_angle_deg, normal_angle_max_deg)
            cos_full = np.cos(np.deg2rad(theta_full))
        else:
            tol_full = np.full(total_points, max_plane_dist, dtype=float)
            cos_full = np.full(total_points, normal_cos, dtype=float)
        facades = postprocess_facades(
            facades,
            points=points, normals=normals, spacing_all=spacing_all,
            tol_full=tol_full, cos_full=cos_full,
            rebuild=_rebuild, facade_ok=_facade_ok,
            uv_grid_size=uv_grid_size,
            merge_pass=merge_pass, completion_pass=completion_pass,
            merge_gap_m=merge_gap_m, min_opening_area=min_opening_area,
            max_bridge_cells=max_bridge_cells,
            merge_angle_deg=Config.FACADE_MERGE_ANGLE_DEG,
            merge_d_thresh=Config.FACADE_MERGE_D_THRESH)

    # 排序：立面优先，同类型按面积降序
    facades.sort(key=lambda f: (f['type'] != 'vertical_facade', -f['point_count']))
    for i, facade in enumerate(facades):
        facade['id'] = i

    return {
        'facades': facades,
        'remaining': remaining,
        'total_points': total_points
    }