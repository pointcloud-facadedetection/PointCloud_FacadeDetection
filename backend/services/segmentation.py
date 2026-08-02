import numpy as np
import open3d as o3d

from ..core.geometry_utils import (
    fit_plane_svd, classify_plane, plane_axes, ensure_normals
)
from ..config import Config


def compute_segment_dimensions(points, normal):
    """计算局部平面宽度、高度、面积等尺寸"""
    if len(points) < 3:
        return {'width': 0.0, 'height': 0.0, 'area': 0.0, 'bbox_2d': None}

    center = np.mean(points, axis=0)
    facade_type, _, _, _ = classify_plane(normal)
    u_axis, v_axis = plane_axes(normal, facade_type)

    local_u = np.dot(points - center, u_axis)
    local_v = np.dot(points - center, v_axis)

    u_min, u_max = float(np.min(local_u)), float(np.max(local_u))
    v_min, v_max = float(np.min(local_v)), float(np.max(local_v))

    width = float(max(u_max - u_min, 0.0))
    height = float(max(v_max - v_min, 0.0))

    if facade_type == 'vertical_facade':
        height = float(max(np.max(points[:, 2]) - np.min(points[:, 2]), 0.0))

    try:
        lifted = np.column_stack([local_u, local_v, np.zeros(len(points))])
        hull_pcd = o3d.geometry.PointCloud()
        hull_pcd.points = o3d.utility.Vector3dVector(lifted)
        hull, _ = hull_pcd.compute_convex_hull()
        area = float(hull.get_surface_area() * 0.5)
    except Exception:
        area = float(width * height)

    return {
        'width': width,
        'height': height,
        'area': area,
        'bbox_2d': {
            'u_min': u_min, 'u_max': u_max,
            'v_min': v_min, 'v_max': v_max,
            'u_axis': [float(x) for x in u_axis],
            'v_axis': [float(x) for x in v_axis],
            'center': [float(x) for x in center]
        }
    }


def build_segment_info(segment_id, points, original_indices, color):
    """构建框选区域内的局部平面信息"""
    plane_model = fit_plane_svd(points)
    normal = plane_model[:3]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    plane_model[:3] = normal

    distances = np.abs(points @ normal + plane_model[3])
    dims = compute_segment_dimensions(points, normal)
    facade_type, type_label, verticality, horizontality = classify_plane(normal)

    return {
        'id': int(segment_id),
        'type': facade_type,
        'type_label': type_label,
        'plane_model': [float(x) for x in plane_model],
        'normal': [float(x) for x in normal],
        'center': [float(x) for x in np.mean(points, axis=0)],
        'point_count': int(len(points)),
        'indices': [int(x) for x in original_indices],
        'color': [float(x) for x in color],
        'width': dims['width'],
        'height': dims['height'],
        'area': dims['area'],
        'bbox_2d': dims['bbox_2d'],
        'verticality': float(verticality),
        'horizontality': float(horizontality),
        'flatness_mean': float(np.mean(distances)) if len(distances) else 0.0,
        'flatness_rmse': float(np.sqrt(np.mean(distances ** 2))) if len(distances) else 0.0,
        'flatness_max': float(np.max(distances)) if len(distances) else 0.0,
        'flatness_std': float(np.std(distances)) if len(distances) else 0.0,
    }


def segment_selected_region(pcd, selected_indices, voxel_size=0.05,
                            distance_threshold=None, normal_angle_deg=18.0,
                            dbscan_eps=None, min_segment_points=80,
                            max_segments=30):
    """对用户框选点集执行局部多平面细粒度分割"""
    all_count = len(pcd.points)
    point_labels = np.full(all_count, -1, dtype=int)

    if selected_indices is None or len(selected_indices) < min_segment_points:
        return {'segments': [], 'point_labels': point_labels, 'message': '框选点数不足'}

    selected_indices = np.asarray(selected_indices, dtype=int)
    selected_indices = selected_indices[(selected_indices >= 0) & (selected_indices < all_count)]
    selected_indices = np.unique(selected_indices)

    if len(selected_indices) < min_segment_points:
        return {'segments': [], 'point_labels': point_labels, 'message': '有效框选点数不足'}

    pcd_work = ensure_normals(o3d.geometry.PointCloud(pcd), voxel_size)
    all_points = np.asarray(pcd_work.points)

    selected_pcd = pcd_work.select_by_index(selected_indices.tolist())
    selected_points = np.asarray(selected_pcd.points)
    if len(selected_points) < min_segment_points:
        return {'segments': [], 'point_labels': point_labels, 'message': '框选区域为空'}

    distance_threshold = distance_threshold or max(float(voxel_size) * 1.5, 0.03)
    dbscan_eps = dbscan_eps or max(float(voxel_size) * 5.0, 0.25)
    normal_cos = float(np.cos(np.deg2rad(normal_angle_deg)))

    remaining_local = np.arange(len(selected_points), dtype=int)
    segments = []
    segment_id = 0

    for _ in range(int(max_segments)):
        if len(remaining_local) < min_segment_points:
            break

        remaining_pcd = selected_pcd.select_by_index(remaining_local.tolist())

        try:
            plane_model, seed_inliers = remaining_pcd.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=3,
                num_iterations=800
            )
        except Exception:
            break

        if len(seed_inliers) < min_segment_points:
            break

        plane_model = np.asarray(plane_model, dtype=float)
        normal = plane_model[:3]
        normal = normal / (np.linalg.norm(normal) + 1e-12)

        rem_points = np.asarray(remaining_pcd.points)
        rem_normals = np.asarray(remaining_pcd.normals) if remaining_pcd.has_normals() else np.zeros_like(rem_points)

        distances = np.abs(rem_points @ normal + plane_model[3])
        normal_agree = np.abs(rem_normals @ normal) >= normal_cos
        candidate_mask = (distances <= distance_threshold * 1.8) & normal_agree
        candidate_local = np.where(candidate_mask)[0]

        if len(candidate_local) < min_segment_points:
            candidate_local = np.asarray(seed_inliers, dtype=int)

        candidate_pcd = remaining_pcd.select_by_index(candidate_local.tolist())
        labels = np.asarray(candidate_pcd.cluster_dbscan(
            eps=dbscan_eps,
            min_points=max(8, min_segment_points // 10),
            print_progress=False
        ))

        clusters = []
        if labels.size and labels.max() >= 0:
            for label in range(int(labels.max()) + 1):
                cluster = candidate_local[labels == label]
                if len(cluster) >= min_segment_points:
                    clusters.append(cluster)
        elif len(candidate_local) >= min_segment_points:
            clusters.append(candidate_local)

        consumed_local = set(int(i) for i in seed_inliers)

        for cluster_in_remaining in sorted(clusters, key=len, reverse=True):
            if segment_id >= max_segments:
                break

            cluster_selected_local = remaining_local[cluster_in_remaining]
            cluster_original_indices = selected_indices[cluster_selected_local]
            cluster_points = all_points[cluster_original_indices]

            refined_model = fit_plane_svd(cluster_points)
            refined_normal = refined_model[:3]
            refined_normal = refined_normal / (np.linalg.norm(refined_normal) + 1e-12)
            refined_dist = np.abs(cluster_points @ refined_normal + refined_model[3])
            keep = refined_dist <= distance_threshold * 2.0

            if np.sum(keep) < min_segment_points:
                continue

            cluster_original_indices = cluster_original_indices[keep]
            cluster_points = all_points[cluster_original_indices]
            color = Config.SEGMENT_COLORS[segment_id % len(Config.SEGMENT_COLORS)]

            info = build_segment_info(segment_id, cluster_points, cluster_original_indices, color)
            segments.append(info)

            point_labels[cluster_original_indices] = segment_id
            consumed_local.update(int(i) for i in cluster_in_remaining)
            segment_id += 1

        if not consumed_local:
            break

        keep_mask = np.ones(len(remaining_local), dtype=bool)
        keep_mask[list(consumed_local)] = False
        remaining_local = remaining_local[keep_mask]

    return {
        'segments': segments,
        'point_labels': point_labels,
        'message': f'检测到 {len(segments)} 个局部平面'
    }
