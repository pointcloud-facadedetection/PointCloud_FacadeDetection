import copy
import numpy as np

from ..core.geometry_utils import (
    fit_plane_svd, classify_plane, project_to_uv, connected_components_2d_grid,
    estimate_plane_area, ensure_normals
)
from ..config import Config


def build_facade_info(facade_id, plane_model, points, original_indices):
    """构建立面信息字典"""
    plane_model = np.asarray(plane_model, dtype=float)
    normal = plane_model[:3]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    plane_model[:3] = normal
    plane_model[3] = float(plane_model[3])
    center = np.mean(points, axis=0)
    facade_type, type_label, verticality, horizontality = classify_plane(normal)
    area, bbox_2d = estimate_plane_area(points, normal, center, facade_type)

    distances = np.abs(points @ normal + plane_model[3])
    flatness_mean = float(np.mean(distances)) if len(distances) else 0.0
    flatness_rmse = float(np.sqrt(np.mean(distances ** 2))) if len(distances) else 0.0
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
                   min_cells_ratio=0.015):  # 最小连通域占点比例
    """
    检测点云中的主要建筑平面

    1. 候选点先用【有符号距离】精筛，排除前后深度不同的平行面
    2. 剩余点投影到UV平面，做2D连通域分析（网格BFS或2D-DBSCAN）
    3. 每个2D连通域独立拟合平面，避免深度不一致平面被合并
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
    min_cluster_points = max(int(min_cluster_points), min_remaining)

    # UV网格参数
    uv_grid_size = uv_grid_size or max(float(voxel_size) * 2.0, 0.1)
    min_cells = max(2, int(min_cluster_points * min_cells_ratio))

    remaining_indices = np.arange(total_points, dtype=int)
    remaining = pcd_work
    facades = []
    facade_id = 0

    for _ in range(Config.FACADE_MAX_ITERATIONS):
        if len(remaining.points) < min_remaining:
            break

        # ---- Step 1: RANSAC种子平面 ----
        plane_model, seed_inliers = remaining.segment_plane(
            distance_threshold=max_plane_dist,
            ransac_n=3,
            num_iterations=500
        )
        if len(seed_inliers) < min_remaining:
            break

        plane_model = np.asarray(plane_model, dtype=float)
        normal = plane_model[:3]
        normal = normal / (np.linalg.norm(normal) + 1e-12)

        # ---- Step 2: 候选点筛选（法向一致性 + 绝对距离粗筛）----
        rem_points = np.asarray(remaining.points)
        rem_normals = np.asarray(remaining.normals)
        distances = np.abs(rem_points @ normal + plane_model[3])
        normal_agree = np.abs(rem_normals @ normal) >= normal_cos

        # 绝对距离放宽到1.5倍，捕获候选
        candidate_mask = (distances <= max_plane_dist * 1.5) & normal_agree
        candidate_local = np.where(candidate_mask)[0]

        if len(candidate_local) < min_remaining:
            candidate_local = np.asarray(seed_inliers, dtype=int)

        # ---- Step 3: 有符号距离精筛 ----
        # 只保留紧贴种子平面的点，排除前后深度不同的平行平面
        signed_dists = rem_points[candidate_local] @ normal + float(plane_model[3])
        tight_mask = np.abs(signed_dists) <= max_plane_dist * signed_dist_tolerance
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
                # 使用网格BFS
                components = connected_components_2d_grid(
                    uv_coords,
                    grid_size=uv_grid_size,
                    min_cells=min_cells
                )

                if len(components) > 0:
                    consumed_local = set()

                    for comp_mask in sorted(components, key=lambda m: np.sum(m), reverse=True):
                        if np.sum(comp_mask) < min_cluster_points:
                            continue

                        # 该连通域的原始3D点索引
                        comp_local = tight_local[comp_mask]
                        comp_points = rem_points[comp_local]

                        # SVD精修平面
                        refined_model = fit_plane_svd(comp_points)
                        comp_normal = refined_model[:3]
                        refined_dist = np.abs(comp_points @ comp_normal + refined_model[3])

                        # 精修后再次用有符号距离过滤离群点
                        keep = refined_dist <= max_plane_dist * 1.8
                        if np.sum(keep) < min_cluster_points:
                            continue

                        comp_local = comp_local[keep]
                        comp_points = rem_points[comp_local]
                        refined_model = fit_plane_svd(comp_points)

                        # 构建立面信息
                        original_indices = remaining_indices[comp_local]
                        info = build_facade_info(facade_id, refined_model, comp_points, original_indices)

                        if info['area'] < min_facade_area:
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
        candidate_pcd = remaining.select_by_index(candidate_local.tolist())
        labels = np.asarray(candidate_pcd.cluster_dbscan(
            eps=dbscan_eps,
            min_points=max(12, min_cluster_points // 8),
            print_progress=False
        ))
        clusters = []
        if labels.size and labels.max() >= 0:
            for label in range(int(labels.max()) + 1):
                local_cluster = candidate_local[labels == label]
                if len(local_cluster) >= min_cluster_points:
                    clusters.append(local_cluster)
        elif len(candidate_local) >= min_cluster_points:
            clusters.append(candidate_local)

        consumed_local = set(int(i) for i in seed_inliers)
        for local_cluster in sorted(clusters, key=len, reverse=True):
            cluster_points = rem_points[local_cluster]
            refined_model = fit_plane_svd(cluster_points)
            cluster_normal = refined_model[:3]
            refined_dist = np.abs(cluster_points @ cluster_normal + refined_model[3])
            keep = refined_dist <= max_plane_dist * 1.8
            if np.sum(keep) < min_cluster_points:
                continue
            local_cluster = local_cluster[keep]
            cluster_points = rem_points[local_cluster]
            refined_model = fit_plane_svd(cluster_points)
            original_indices = remaining_indices[local_cluster]
            info = build_facade_info(facade_id, refined_model, cluster_points, original_indices)
            if info['area'] < min_facade_area:
                consumed_local.update(int(i) for i in local_cluster)
                continue
            facades.append(info)
            facade_id += 1
            consumed_local.update(int(i) for i in local_cluster)

        if not consumed_local:
            break
        keep_mask = np.ones(len(remaining.points), dtype=bool)
        keep_mask[list(consumed_local)] = False
        remaining_indices = remaining_indices[keep_mask]
        remaining = remaining.select_by_index(np.where(keep_mask)[0].tolist())

    # 排序：立面优先，同类型按面积降序
    facades.sort(key=lambda f: (f['type'] != 'vertical_facade', -f['point_count']))
    for i, facade in enumerate(facades):
        facade['id'] = i

    return {
        'facades': facades,
        'remaining': remaining,
        'total_points': total_points
    }