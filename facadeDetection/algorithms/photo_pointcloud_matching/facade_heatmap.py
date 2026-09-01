"""按照片标注区域选择立面并生成规则网格凹凸热力图。"""

import cv2
import numpy as np

from algorithms.facade.heatmap_colors import signed_deviation_colors


def _normalize_distortion(distortion):
    if distortion is None:
        return np.zeros((5, 1), dtype=np.float64)
    coeffs = np.asarray(distortion, dtype=np.float64).reshape(-1)
    out = np.zeros((5, 1), dtype=np.float64)
    out[: min(len(coeffs), 5), 0] = coeffs[:5]
    return out


def _plane_axes(normal, facade_type=None):
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal) + 1e-12
    z_axis = np.array([0.0, 0.0, 1.0])
    if facade_type == "vertical_facade" or abs(normal[2]) < 0.45:
        u_axis = np.cross(z_axis, normal)
        if np.linalg.norm(u_axis) < 1e-8:
            u_axis = np.array([1.0, 0.0, 0.0])
        u_axis /= np.linalg.norm(u_axis) + 1e-12
        v_axis = z_axis
    else:
        ref = z_axis if abs(np.dot(normal, z_axis)) < 0.9 else np.array([1, 0, 0])
        u_axis = np.cross(normal, ref)
        u_axis /= np.linalg.norm(u_axis) + 1e-12
        v_axis = np.cross(normal, u_axis)
        v_axis /= np.linalg.norm(v_axis) + 1e-12
    return u_axis, v_axis


def _project_points(points, camera_matrix, rotation, translation, distortion=None):
    points = np.asarray(points, dtype=float)
    camera_matrix = np.asarray(camera_matrix, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    translation = np.asarray(translation, dtype=float).reshape(3)
    distortion = _normalize_distortion(distortion)
    rvec, _ = cv2.Rodrigues(rotation)
    projected, _ = cv2.projectPoints(
        points.reshape(-1, 1, 3),
        rvec,
        translation,
        camera_matrix,
        distortion,
    )
    projected = projected.reshape(-1, 2)
    camera_points = (rotation @ points.T).T + translation.reshape(1, 3)
    depth = camera_points[:, 2]
    return projected, depth


def _heat_color(value, limit):
    """平整灰、凹陷 Blues、凸起 autumn_r。"""
    from algorithms.facade.heatmap_colors import (
        compute_heatmap_scale,
    )

    span = abs(float(limit))
    threshold, vmin, vmax = compute_heatmap_scale([-span, span])
    return signed_deviation_colors(
        [value],
        threshold=threshold,
        vmin=vmin,
        vmax=vmax,
    )[0]


def _points_inside_polygon(points_2d, polygon):
    """使用射线法判断二维点是否在凸/非凸多边形内部。"""
    points_2d = np.asarray(points_2d, dtype=float)
    polygon = np.asarray(polygon, dtype=float)
    if polygon.shape != (4, 2):
        raise ValueError("请在照片中按顺序标注 4 个立面顶点")

    inside = np.zeros(len(points_2d), dtype=bool)
    x, y = points_2d[:, 0], points_2d[:, 1]
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        intersects = ((y1 > y) != (y2 > y)) & (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
        )
        inside ^= intersects
        previous = current
    return inside


def _camera_center(rotation, translation):
    rotation = np.asarray(rotation, dtype=float)
    translation = np.asarray(translation, dtype=float).reshape(3)
    return -rotation.T @ translation


def _pixel_to_world_ray(pixel, camera_matrix, rotation, translation, distortion=None):
    """像素坐标 -> 世界坐标系下的相机中心与射线方向（含畸变校正）。"""
    camera_matrix = np.asarray(camera_matrix, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    distortion = _normalize_distortion(distortion)
    origin = _camera_center(rotation, translation)
    pts = np.asarray([[pixel]], dtype=np.float64)
    normalized = cv2.undistortPoints(pts, camera_matrix, distortion)
    direction_cam = np.array(
        [normalized[0, 0, 0], normalized[0, 0, 1], 1.0],
        dtype=float,
    )
    direction_cam /= np.linalg.norm(direction_cam) + 1e-12
    direction_world = rotation.T @ direction_cam
    direction_world /= np.linalg.norm(direction_world) + 1e-12
    return origin, direction_world


def _ray_plane_intersection(origin, direction, plane_model):
    plane = np.asarray(plane_model, dtype=float)
    normal = plane[:3]
    normal /= np.linalg.norm(normal) + 1e-12
    denom = float(np.dot(normal, direction))
    if abs(denom) < 1e-9:
        return None
    distance = -(float(np.dot(normal, origin)) + plane[3]) / denom
    if distance <= 0:
        return None
    return origin + direction * distance


def _orient_plane_toward_camera(plane_model, reference_point, camera_center):
    plane = np.asarray(plane_model, dtype=float)
    norm = np.linalg.norm(plane[:3])
    if norm < 1e-12:
        raise ValueError("立面平面模型无效")
    plane = plane / norm
    if np.dot(plane[:3], np.asarray(camera_center) - np.asarray(reference_point)) < 0:
        plane *= -1.0
    return plane


def compute_photo_quadrilateral_3d(
    image_quadrilateral,
    camera_matrix,
    rotation,
    translation,
    plane_model,
    distortion=None,
    facade_points=None,
    refine_on_facade=True,
):
    """将 2D 照片四顶点通过相机射线与立面平面求交，得到 3D 角点。"""
    distortion = _normalize_distortion(distortion)
    corners = []
    for index, pixel in enumerate(image_quadrilateral, start=1):
        origin, direction = _pixel_to_world_ray(
            pixel, camera_matrix, rotation, translation, distortion
        )
        point = _ray_plane_intersection(origin, direction, plane_model)
        if point is None:
            raise ValueError(
                f"第 {index} 个照片顶点无法与立面平面求交，请检查相机位姿或顶点位置"
            )
        corners.append(point)
    corners = np.asarray(corners, dtype=float)

    if (
        refine_on_facade
        and facade_points is not None
        and len(facade_points) >= 4
    ):
        corners = refine_quadrilateral_corners_on_facade(
            image_quadrilateral,
            corners,
            facade_points,
            camera_matrix,
            rotation,
            translation,
            distortion=distortion,
        )
    return corners


def refine_quadrilateral_corners_on_facade(
    image_quadrilateral,
    corners_3d_init,
    facade_points,
    camera_matrix,
    rotation,
    translation,
    distortion=None,
    search_radius_m=3.0,
    max_reproj_px=120.0,
):
    """把平面求交得到的 3D 角点吸附到附近立面点云，减小远距离/位姿误差。"""
    facade_points = np.asarray(facade_points, dtype=float)
    corners_3d_init = np.asarray(corners_3d_init, dtype=float)
    pixel = np.asarray(image_quadrilateral, dtype=float)
    origin = _camera_center(rotation, translation)
    refined = []

    for index in range(len(corners_3d_init)):
        init_3d = corners_3d_init[index]
        target_pixel = pixel[index]

        init_projected, init_depth = _project_points(
            init_3d.reshape(1, 3),
            camera_matrix,
            rotation,
            translation,
            distortion,
        )
        init_error = float(np.linalg.norm(init_projected[0] - target_pixel))

        depth = float(init_depth[0]) if init_depth[0] > 0 else float(
            np.linalg.norm(init_3d - origin)
        )
        radius = max(search_radius_m, 1.5 + 0.04 * depth)
        max_perp = max(0.35, 0.15 + 0.008 * depth)

        deltas = facade_points - init_3d
        dists = np.linalg.norm(deltas, axis=1)
        near_mask = dists <= radius

        to_points = facade_points - origin.reshape(1, 3)
        ray_dir = init_3d - origin
        ray_len = np.linalg.norm(ray_dir) + 1e-12
        ray_unit = ray_dir / ray_len
        along = to_points @ ray_unit
        perp = np.linalg.norm(to_points - along.reshape(-1, 1) * ray_unit, axis=1)
        ray_mask = (along > 0.5) & (along < depth + radius * 2.0) & (perp <= max_perp)

        candidate_mask = near_mask | ray_mask
        candidates = facade_points[candidate_mask]
        if len(candidates) == 0:
            refined.append(init_3d)
            continue

        projected, depths = _project_points(
            candidates,
            camera_matrix,
            rotation,
            translation,
            distortion,
        )
        errors = np.linalg.norm(projected - target_pixel.reshape(1, 2), axis=1)
        valid = depths > 0
        if not np.any(valid):
            refined.append(init_3d)
            continue

        errors = np.where(valid, errors, np.inf)
        best_idx = int(np.argmin(errors))
        best_error = float(errors[best_idx])
        if best_error <= max_reproj_px and best_error <= init_error * 1.15 + 2.0:
            refined.append(candidates[best_idx])
        else:
            refined.append(init_3d)

    return np.asarray(refined, dtype=float)


def _project_points_to_plane_uv(points, plane_model, reference_point, facade_type=None):
    plane = np.asarray(plane_model, dtype=float)
    normal = plane[:3]
    u_axis, v_axis = _plane_axes(normal, facade_type)
    plane_center = np.asarray(reference_point, dtype=float) - (
        np.dot(normal, reference_point) + plane[3]
    ) * normal
    points = np.asarray(points, dtype=float)
    local_u = (points - plane_center) @ u_axis
    local_v = (points - plane_center) @ v_axis
    return np.column_stack([local_u, local_v]), plane_center, u_axis, v_axis


def select_points_in_3d_quadrilateral(
    points,
    point_indices,
    plane_model,
    corners_3d,
    facade_type=None,
):
    """在立面平面 UV 坐标系中，选择落在 3D 四边形内的点。"""
    indices = np.asarray(point_indices, dtype=int)
    indices = indices[(indices >= 0) & (indices < len(points))]
    if len(indices) == 0:
        return indices

    reference = np.mean(np.asarray(corners_3d, dtype=float), axis=0)
    poly_uv, _, _, _ = _project_points_to_plane_uv(
        corners_3d, plane_model, reference, facade_type
    )
    pts_uv, _, _, _ = _project_points_to_plane_uv(
        points[indices], plane_model, reference, facade_type
    )
    inside = _points_inside_polygon(pts_uv, poly_uv)
    return indices[inside]


def project_world_points_to_image(
    world_points,
    camera_matrix,
    rotation,
    translation,
    distortion_coefficients=None,
):
    """将世界坐标 3D 点投影到像素坐标（含畸变）。"""
    world_points = np.asarray(world_points, dtype=float)
    if world_points.ndim != 2 or world_points.shape[1] != 3:
        raise ValueError("world_points 必须是 N×3 数组")
    projected, depth = _project_points(
        world_points,
        camera_matrix,
        rotation,
        translation,
        distortion_coefficients,
    )
    return {
        "image_points": projected.tolist(),
        "depths": depth.tolist(),
    }


def _normalize_plane_model(plane_model):
    plane = np.asarray(plane_model, dtype=float)
    norm = np.linalg.norm(plane[:3])
    if norm < 1e-12:
        raise ValueError("立面平面模型无效")
    return plane / norm


def snap_point_to_facade_plane(
    point_zup,
    facades,
    facade_id=None,
    max_plane_dist=4.0,
):
    """将 3D 点正交投影到最近（或指定）立面拟合平面上。"""
    point = np.asarray(point_zup, dtype=float).reshape(3)
    if not facades:
        raise ValueError("尚未检测建筑立面")

    candidates = []
    for facade in facades:
        fid = int(facade["id"])
        if facade_id is not None and fid != int(facade_id):
            continue
        plane = _normalize_plane_model(facade.get("plane_model", []))
        normal = plane[:3]
        signed = float(np.dot(normal, point) + plane[3])
        abs_dist = abs(signed)
        snapped = point - signed * normal
        candidates.append({
            "facade_id": fid,
            "snapped": snapped,
            "distance_m": abs_dist,
            "area": float(facade.get("area", 0.0)),
        })

    if not candidates:
        raise ValueError("未找到可用立面平面")

    if facade_id is None:
        within = [c for c in candidates if c["distance_m"] <= max_plane_dist]
        pool = within if within else candidates
        best = min(pool, key=lambda c: (c["distance_m"], -c["area"]))
    else:
        best = min(candidates, key=lambda c: c["distance_m"])

    return {
        "snapped_point": best["snapped"].tolist(),
        "facade_id": best["facade_id"],
        "plane_distance_m": float(best["distance_m"]),
    }


def resolve_photo_detection_region(
    points,
    facades,
    camera_matrix,
    rotation,
    translation,
    image_quadrilateral,
    distortion_coefficients=None,
    refine_corners=True,
    corners_3d=None,
):
    """按照片四边形求 3D 角点，并返回 3D 框选区域内的立面点。"""
    points = np.asarray(points, dtype=float)
    distortion = _normalize_distortion(distortion_coefficients)
    selected = select_facade_in_photo_quadrilateral(
        points,
        facades,
        camera_matrix,
        rotation,
        translation,
        image_quadrilateral,
        distortion,
    )
    facade = selected["facade"]
    camera_center = _camera_center(rotation, translation)
    facade_indices = np.asarray(facade.get("inlier_indices", []), dtype=int)
    reference = facade.get("center")
    if reference is None and len(facade_indices) > 0:
        reference = np.mean(points[facade_indices], axis=0)
    if reference is None:
        reference = np.mean(points, axis=0)

    plane = _orient_plane_toward_camera(
        facade.get("plane_model", []),
        reference,
        camera_center,
    )
    if corners_3d is not None:
        corners_3d = np.asarray(corners_3d, dtype=float)
        if corners_3d.shape != (4, 3):
            raise ValueError("corners_3d 必须是 4×3 数组")
    else:
        facade_points = points[facade_indices] if len(facade_indices) > 0 else points
        corners_3d = compute_photo_quadrilateral_3d(
            image_quadrilateral,
            camera_matrix,
            rotation,
            translation,
            plane,
            distortion=distortion,
            facade_points=facade_points,
            refine_on_facade=refine_corners,
        )
    region_indices = select_points_in_3d_quadrilateral(
        points,
        facade_indices,
        plane,
        corners_3d,
        facade.get("type"),
    )
    if len(region_indices) < 3:
        raise ValueError("3D 检测框内有效立面点过少，请检查照片顶点或相机位姿")

    return {
        "facade": facade,
        "plane_model": plane,
        "corners_3d": corners_3d,
        "indices": region_indices,
        "area": float(facade.get("area", 0.0)),
        "camera_center": camera_center,
    }


def select_facade_in_photo_quadrilateral(
    points,
    facades,
    camera_matrix,
    rotation,
    translation,
    image_quadrilateral,
    distortion=None,
):
    """选择投影点落入用户照片四边形最多的已检测立面。"""
    points = np.asarray(points, dtype=float)
    candidates = []
    for facade in facades:
        indices = np.asarray(facade.get("inlier_indices", []), dtype=int)
        indices = indices[(indices >= 0) & (indices < len(points))]
        if len(indices) < 3:
            continue

        projected, depth = _project_points(
            points[indices], camera_matrix, rotation, translation, distortion
        )
        inside = (depth > 0) & _points_inside_polygon(
            projected, image_quadrilateral
        )
        selected_indices = indices[inside]
        if len(selected_indices) < 3:
            continue
        candidates.append({
            "facade": facade,
            "indices": selected_indices,
            "area": float(facade.get("area", 0.0)),
            "coverage_ratio": float(len(selected_indices) / len(indices)),
        })

    if not candidates:
        raise ValueError(
            "标注四边形内未找到已检测立面，请检查四个顶点、相机位姿或立面检测结果"
        )
    return max(candidates, key=lambda item: (len(item["indices"]), item["coverage_ratio"]))


def select_visible_largest_facade(
    points,
    facades,
    camera_matrix,
    rotation,
    translation,
    image_width,
    image_height,
    distortion=None,
):
    """选择落入照片视野且可见面积最大的已检测立面。"""
    points = np.asarray(points, dtype=float)
    candidates = []
    for facade in facades:
        indices = np.asarray(facade.get("inlier_indices", []), dtype=int)
        indices = indices[(indices >= 0) & (indices < len(points))]
        if len(indices) < 3:
            continue

        # 限制可见性检查数量，避免大立面投影过慢。
        stride = max(1, len(indices) // 5000)
        sample = points[indices[::stride]]
        projected, depth = _project_points(
            sample, camera_matrix, rotation, translation, distortion
        )
        visible = (
            (depth > 0)
            & (projected[:, 0] >= 0)
            & (projected[:, 0] < image_width)
            & (projected[:, 1] >= 0)
            & (projected[:, 1] < image_height)
        )
        visible_count = int(np.sum(visible))
        visible_ratio = float(visible_count / max(len(sample), 1))
        if visible_count < 3 or visible_ratio < 0.02:
            continue

        candidates.append(
            {
                "facade": facade,
                "indices": indices,
                "visible_ratio": visible_ratio,
                "area": float(facade.get("area", 0.0)),
                "score": float(facade.get("area", 0.0)) * visible_ratio,
            }
        )

    if not candidates:
        raise ValueError("照片视野内未找到已检测立面，请检查相机位姿或先执行立面检测")
    return max(candidates, key=lambda item: (item["score"], item["area"]))


def build_facade_grid_heatmap(
    points,
    facade,
    camera_center,
    grid_size=0.5,
    min_points_per_cell=3,
    percentile=98.0,
):
    """将立面点投影到局部 UV 网格，并生成可供 Three.js 渲染的网格。"""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        raise ValueError("立面点数不足")

    plane = np.asarray(facade.get("plane_model", []), dtype=float)
    if plane.size != 4 or np.linalg.norm(plane[:3]) < 1e-12:
        raise ValueError("立面平面模型无效")
    plane /= np.linalg.norm(plane[:3])
    normal = plane[:3]
    center = np.asarray(facade.get("center", np.mean(points, axis=0)), dtype=float)

    # 法向统一朝向相机：正偏差表示朝相机凸起，负偏差表示背向凹陷。
    if np.dot(normal, np.asarray(camera_center) - center) < 0:
        plane *= -1.0
        normal = plane[:3]

    u_axis, v_axis = _plane_axes(normal, facade.get("type"))
    plane_center = center - (np.dot(normal, center) + plane[3]) * normal
    local_u = (points - plane_center) @ u_axis
    local_v = (points - plane_center) @ v_axis
    signed = points @ normal + plane[3]

    u_min, u_max = float(np.min(local_u)), float(np.max(local_u))
    v_min, v_max = float(np.min(local_v)), float(np.max(local_v))
    # 最多约 200×200 个网格，避免返回过大的 JSON。
    grid_size = max(
        float(grid_size),
        (u_max - u_min) / 200.0,
        (v_max - v_min) / 200.0,
        1e-3,
    )
    u_bins = max(1, int(np.ceil((u_max - u_min) / grid_size)))
    v_bins = max(1, int(np.ceil((v_max - v_min) / grid_size)))
    u_idx = np.clip(((local_u - u_min) / grid_size).astype(int), 0, u_bins - 1)
    v_idx = np.clip(((local_v - v_min) / grid_size).astype(int), 0, v_bins - 1)

    robust_limit = float(np.percentile(np.abs(signed), percentile))
    robust_limit = max(robust_limit, 0.004)
    vertices = []
    colors = []
    triangles = []
    line_positions = []
    values = []
    cells_2d = []
    offset = normal * max(grid_size * 0.002, 0.002)

    cell_ids = u_idx + v_idx * u_bins
    for cell_id in np.unique(cell_ids):
        mask = cell_ids == cell_id
        if int(np.sum(mask)) < int(min_points_per_cell):
            continue
        j, i = divmod(int(cell_id), u_bins)
        u0 = u_min + i * grid_size
        u1 = min(u0 + grid_size, u_max)
        v0 = v_min + j * grid_size
        v1 = min(v0 + grid_size, v_max)
        if u1 <= u0 or v1 <= v0:
            continue

        value = float(np.median(signed[mask]))
        color = _heat_color(value, robust_limit)
        corners = [
            plane_center + u_axis * u0 + v_axis * v0 + offset,
            plane_center + u_axis * u1 + v_axis * v0 + offset,
            plane_center + u_axis * u1 + v_axis * v1 + offset,
            plane_center + u_axis * u0 + v_axis * v1 + offset,
        ]
        base = len(vertices)
        vertices.extend(c.tolist() for c in corners)
        colors.extend(color.tolist() for _ in range(4))
        triangles.extend([base, base + 1, base + 2, base, base + 2, base + 3])
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            line_positions.extend([corners[a].tolist(), corners[b].tolist()])
        values.append(value)
        cells_2d.append({
            "i": int(i),
            "j": int(j),
            "u0": float(u0),
            "u1": float(u1),
            "v0": float(v0),
            "v1": float(v1),
            "value_m": value,
            "color": color.tolist(),
        })

    if not vertices:
        raise ValueError("网格内有效点太少，无法生成热力图")

    return {
        "vertices": vertices,
        "colors": colors,
        "triangles": triangles,
        "line_positions": line_positions,
        "cell_values_m": values,
        "cell_count": len(values),
        "grid_size_m": grid_size,
        "deviation_limit_m": robust_limit,
        "min_deviation_m": float(np.min(values)),
        "max_deviation_m": float(np.max(values)),
        "normal_toward_camera": normal.tolist(),
        "grid_2d": {
            "u_min": u_min,
            "u_max": u_max,
            "v_min": v_min,
            "v_max": v_max,
            "u_bins": u_bins,
            "v_bins": v_bins,
            "grid_size_m": grid_size,
            "cells": cells_2d,
        },
    }


def build_region_highlight_colors(num_points, highlight_indices, base_colors, highlight_color):
    """将照片检测区域内的点云高亮，其余点变暗。"""
    colors = np.asarray(base_colors, dtype=float).reshape(-1, 3).copy()
    if colors.shape[0] != num_points:
        colors = np.ones((num_points, 3), dtype=float) * 0.7
    colors *= 0.22
    indices = np.asarray(highlight_indices, dtype=int)
    indices = indices[(indices >= 0) & (indices < num_points)]
    if len(indices) == 0:
        raise ValueError("检测区域内没有可高亮的点云点")
    colors[indices] = np.asarray(highlight_color, dtype=float)
    return colors


def create_photo_facade_heatmap(
    points,
    facades,
    camera_matrix,
    rotation,
    translation,
    image_width,
    image_height,
    grid_size=0.5,
    image_quadrilateral=None,
    distortion_coefficients=None,
    corners_3d=None,
):
    """按照片四边形区域选择立面并生成网格热力图。"""
    points = np.asarray(points, dtype=float)
    camera_matrix = np.asarray(camera_matrix, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    translation = np.asarray(translation, dtype=float)
    distortion = _normalize_distortion(distortion_coefficients)
    if image_quadrilateral is None:
        selected = select_visible_largest_facade(
            points, facades, camera_matrix, rotation, translation,
            float(image_width), float(image_height), distortion,
        )
        camera_center = -rotation.T @ translation.reshape(3)
        facade_points = points[selected["indices"]]
        corners_3d = None
    else:
        region = resolve_photo_detection_region(
            points,
            facades,
            camera_matrix,
            rotation,
            translation,
            image_quadrilateral,
            distortion_coefficients=distortion,
            corners_3d=corners_3d,
        )
        selected = region
        camera_center = region["camera_center"]
        facade_points = points[region["indices"]]
        corners_3d = region["corners_3d"]
    heatmap = build_facade_grid_heatmap(
        facade_points,
        selected["facade"],
        camera_center,
        grid_size=grid_size,
    )
    return {
        "facade_id": int(selected["facade"]["id"]),
        "facade_area_m2": selected["area"],
        "visible_ratio": selected.get("visible_ratio"),
        "quadrilateral_coverage_ratio": selected.get("coverage_ratio"),
        "corners_3d": None if corners_3d is None else corners_3d.tolist(),
        "region_point_count": int(len(facade_points)),
        "heatmap": heatmap,
    }
