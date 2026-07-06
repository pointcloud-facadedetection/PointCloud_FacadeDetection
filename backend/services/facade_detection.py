import copy
import numpy as np

from ..core.geometry_utils import (
    fit_plane_svd, classify_plane, plane_axes,
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
                   normal_angle_deg=18.0,
                   dbscan_eps=None,
                   min_cluster_points=80):
    """检测点云中的主要建筑平面（基于下采样点云）"""
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

    remaining_indices = np.arange(total_points, dtype=int)
    remaining = pcd_work
    facades = []
    facade_id = 0

    for _ in range(Config.FACADE_MAX_ITERATIONS):
        if len(remaining.points) < min_remaining:
            break

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

        rem_points = np.asarray(remaining.points)
        rem_normals = np.asarray(remaining.normals)
        distances = np.abs(rem_points @ normal + plane_model[3])
        normal_agree = np.abs(rem_normals @ normal) >= normal_cos
        candidate_mask = (distances <= max_plane_dist * 1.5) & normal_agree
        candidate_local = np.where(candidate_mask)[0]
        if len(candidate_local) < min_remaining:
            candidate_local = np.asarray(seed_inliers, dtype=int)

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

    facades.sort(key=lambda f: (f['type'] != 'vertical_facade', -f['area']))
    for i, facade in enumerate(facades):
        facade['id'] = i

    return {
        'facades': facades,
        'remaining': remaining,
        'total_points': total_points
    }
