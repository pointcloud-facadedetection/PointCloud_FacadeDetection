"""扫描仪局部坐标系下的针孔投影、自动取景与主体裁剪。"""

from __future__ import annotations

import math

import cv2
import numpy as np

def _rot_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((1, 0, 0), (0, c, -s), (0, s, c)), dtype=np.float64)


def _rot_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)), dtype=np.float64)


def _rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)), dtype=np.float64)


def _focal_from_fov(width, height, fov_deg):
    return (max(width, height) * 0.5) / np.tan(np.radians(fov_deg * 0.5))


def _local_points(points, transform_to_global):
    transform = np.asarray(transform_to_global, dtype=np.float64).reshape(4, 4)
    inverse = np.linalg.inv(transform)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return pts @ inverse[:3, :3].T + inverse[:3, 3]


def _prepare_projection_for_lines(image_bgr):
    """闭运算补点云投影空洞，便于 Canny / LSD 找竖线。"""
    image = np.asarray(image_bgr, dtype=np.uint8)
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if int((gray > 0).sum()) > 100:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _estimate_vertical_vanishing_point(image_bgr):
    """Canny + LSD + RANSAC，与 tiaozheng_roll.estimate_vertical_vp 相同。"""
    image = np.asarray(image_bgr, dtype=np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    detected = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(edges)[0]
    if detected is None:
        raise RuntimeError('Cannot estimate vertical VP')

    height = gray.shape[0]
    lines = []
    segments = np.asarray(detected, dtype=np.float64)
    if segments.size == 0 or segments.size % 4 != 0:
        raise RuntimeError('Cannot estimate vertical VP')
    for segment in segments.reshape(-1, 4):
        x1, y1, x2, y2 = segment
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = math.hypot(dx, dy)
        if length < 0.08 * height:
            continue
        angle = math.degrees(math.atan2(dy, dx)) % 180.0
        if not (55.0 < angle < 125.0):
            continue
        line = np.cross(
            np.array((x1, y1, 1.0), dtype=np.float64),
            np.array((x2, y2, 1.0), dtype=np.float64),
        )
        norm = float(np.linalg.norm(line[:2]))
        if norm < 1e-8:
            continue
        lines.append({'line': line / norm, 'length': length})
    if len(lines) < 2:
        raise RuntimeError('Cannot estimate vertical VP')

    line_matrix = np.array([item['line'] for item in lines], dtype=np.float64)
    lengths = np.array([item['length'] for item in lines], dtype=np.float64)
    rng = np.random.default_rng(0)
    best_score = -1.0
    best_inliers = None
    for _ in range(3000):
        i, j = rng.choice(len(lines), 2, replace=False)
        point = np.cross(line_matrix[i], line_matrix[j])
        if abs(float(point[2])) < 1e-10:
            continue
        point = point / point[2]
        inliers = np.abs(line_matrix @ point) < 4.0
        score = float(lengths[inliers].sum())
        if score > best_score:
            best_score = score
            best_inliers = inliers
    if best_inliers is None:
        raise RuntimeError('Cannot estimate vertical VP')

    weights = np.sqrt(lengths[best_inliers])[:, None]
    _u, _s, vt = np.linalg.svd(line_matrix[best_inliers] * weights)
    vanishing = vt[-1]
    if abs(float(vanishing[2])) < 1e-10:
        raise RuntimeError('Cannot estimate vertical VP')
    vanishing = vanishing / vanishing[2]
    return vanishing[:2]


def _roll_from_vertical_vanishing_point(image_bgr, principal_x=None):
    """由竖直消失点计算 roll，使竖线延长线落在图像中轴上。"""
    image = _prepare_projection_for_lines(image_bgr)
    vx, vy = _estimate_vertical_vanishing_point(image)
    height, width = image.shape[:2]
    cx = float(width * 0.5 if principal_x is None else principal_x)
    cy = height * 0.5
    roll_deg = float(np.degrees(np.arctan2(vx - cx, -(vy - cy))))
    return float((roll_deg + 180.0) % 360.0 - 180.0)


def _refine_roll_from_projection(points, colors, transform_to_global, params, image_size):
    """渲染 roll=0 的点云图，按竖直消失点把竖线会聚到图像中轴。"""
    width, height = map(int, image_size)
    scale = min(1.0, 768.0 / max(width, height))
    detect_size = (
        max(160, int(round(width * scale))),
        max(160, int(round(height * scale))),
    )
    preview = dict(params)
    preview['roll'] = 0.0
    preview['point_size'] = max(5, int(params.get('point_size', 3)))
    rendered = render_projection(
        points,
        colors,
        transform_to_global,
        preview,
        image_size=detect_size,
        crop_subject=False,
    )
    return _roll_from_vertical_vanishing_point(rendered['view_bgr'])


def default_projection_params(
    points,
    transform_to_global,
    image_size=(1024, 576),
    *,
    camera_height=1.75,
    fov_candidates=(100.0, 110.0, 120.0, 130.0),
    back_range=(1.0, 3.0),
    top_margin=0.07,
    near=0.3,
    core_percentile=88.0,
):
    """按二维画幅自动拟合扫描仪局部坐标系中的针孔相机。"""
    width, height = map(int, image_size)
    if width <= 0 or height <= 0:
        raise ValueError('投影图像尺寸必须为正数')
    local = _local_points(points, transform_to_global)
    finite = local[np.isfinite(local).all(axis=1)]
    if len(finite) == 0:
        raise ValueError('点云不包含有效坐标')
    if len(finite) > 120_000:
        rng = np.random.default_rng(0)
        finite = finite[rng.choice(len(finite), 120_000, replace=False)]

    radius = np.linalg.norm(finite[:, :2], axis=1)
    z_low, z_high = np.percentile(finite[:, 2], (0.5, 99.5))
    core = finite[
        (radius <= np.percentile(radius, core_percentile))
        & (finite[:, 2] >= z_low)
        & (finite[:, 2] <= z_high)
    ]
    if len(core) < 100:
        core = finite
    if len(core) < 100:
        raise ValueError('有效点数量不足，无法自动取景')

    xy = core[:, :2]
    z_values = core[:, 2]
    azimuth = np.degrees(np.arctan2(xy[:, 1], xy[:, 0]))
    yaw_grid = np.arange(-180.0, 180.0, 5.0)

    def evaluate(camera_xy, yaw_deg, fov_deg):
        angle = np.radians(yaw_deg)
        direction = np.array((np.cos(angle), np.sin(angle)))
        perpendicular = np.array((-direction[1], direction[0]))
        relative = xy - camera_xy
        forward = relative @ direction
        sideways = relative @ perpendicular
        dz = z_values - camera_height
        focal = _focal_from_fov(width, height, fov_deg)
        horizontal_tangent = (width * 0.5) / focal
        visible_horizontally = (
            (forward > near)
            & (np.abs(sideways) <= forward * horizontal_tangent)
        )
        if int(visible_horizontally.sum()) < 100:
            return None

        top_elevation = np.percentile(
            np.arctan2(dz[visible_horizontally], forward[visible_horizontally]),
            99.5,
        )
        elevation = top_elevation - np.arctan(
            ((0.5 - top_margin) * height) / focal
        )
        elevation = float(np.clip(elevation, np.radians(-85), np.radians(85)))

        rotated_forward = forward * np.cos(elevation) + dz * np.sin(elevation)
        rotated_up = -forward * np.sin(elevation) + dz * np.cos(elevation)
        in_front = rotated_forward > near
        if int(in_front.sum()) < 100:
            return None
        u = width * 0.5 - focal * sideways[in_front] / rotated_forward[in_front]
        v = height * 0.5 - focal * rotated_up[in_front] / rotated_forward[in_front]
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if int(inside.sum()) < 100:
            return None
        top_gap = float(np.clip(v[inside].min(), 0, height)) / height
        return (
            float(inside.sum()) / len(xy),
            float(np.degrees(elevation)),
            top_gap,
        )

    def densest_yaw(fov_deg):
        half_fov = np.degrees(
            np.arctan((width * 0.5) / _focal_from_fov(width, height, fov_deg))
        )
        counts = [
            np.sum(
                np.abs((azimuth - yaw + 180.0) % 360.0 - 180.0)
                <= half_fov
            )
            for yaw in yaw_grid
        ]
        return float(yaw_grid[int(np.argmax(counts))])

    best = None
    best_score = -np.inf
    for fov in fov_candidates:
        initial_yaw = densest_yaw(fov)
        for yaw in initial_yaw + np.arange(-10.0, 11.0, 5.0):
            angle = np.radians(yaw)
            direction = np.array((np.cos(angle), np.sin(angle)))
            for distance_back in np.linspace(back_range[0], back_range[1], 9):
                camera_xy = -direction * distance_back
                result = evaluate(camera_xy, yaw, fov)
                if result is None:
                    continue
                fraction, elevation_deg, gap = result
                score = fraction + 0.35 * min(gap / top_margin, 1.0)
                candidate = {
                    'fov': float(fov),
                    'yaw': float(yaw),
                    'pitch': float(-elevation_deg),
                    'roll': 0.0,
                    'tx': float(camera_xy[0]),
                    'ty': float(camera_xy[1]),
                    'tz': float(camera_height),
                    'near': float(near),
                    'point_size': 3,
                    'center_h': True,
                    'center_v': False,
                    '_score': score,
                }
                if score > best_score:
                    best = candidate
                    best_score = score

    if best is None:
        raise ValueError('点云主体未落入候选视场，无法自动取景')
    distance = np.linalg.norm(
        core - np.array((best['tx'], best['ty'], best['tz'])), axis=1
    )
    best['far'] = float(
        np.clip(np.percentile(distance, 99.0) * 1.15, 10.0, 300.0)
    )
    best.pop('_score', None)
    try:
        best['roll'] = _refine_roll_from_projection(
            points,
            None,
            transform_to_global,
            best,
            image_size,
        )
    except (RuntimeError, ValueError, AttributeError, cv2.error):
        best['roll'] = 0.0
    return best


def projection_viewport_camera(params, transform_to_global):
    """把针孔投影姿态转换为 Open3D 视口的 eye/lookat/up。"""
    yaw, pitch, roll = (
        np.radians(float(params[key])) for key in ('yaw', 'pitch', 'roll')
    )
    rotation_local = _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)
    transform = np.asarray(transform_to_global, dtype=np.float64).reshape(4, 4)
    local_eye = np.array(
        (params['tx'], params['ty'], params['tz']), dtype=np.float64
    )
    eye = transform[:3, :3] @ local_eye + transform[:3, 3]
    forward = transform[:3, :3] @ rotation_local[:, 0]
    up = transform[:3, :3] @ rotation_local[:, 2]
    forward /= np.linalg.norm(forward) + 1e-12
    up -= forward * float(np.dot(forward, up))
    up /= np.linalg.norm(up) + 1e-12
    focus_distance = float(
        np.clip(float(params.get('far', 20.0)) * 0.5, 1.0, 20.0)
    )
    return {
        'eye': eye.tolist(),
        'lookat': (eye + forward * focus_distance).tolist(),
        'up': up.tolist(),
    }


def _intensity_rgb(colors, point_count):
    """将 PLY colors 统一转换为测试查看器使用的归一化灰度强度。"""
    if colors is None or len(colors) != point_count:
        return np.full((point_count, 3), 220, dtype=np.uint8)
    values = np.asarray(colors, dtype=np.float32)
    if values.ndim == 1:
        intensity = values.reshape(-1)
    elif values.ndim == 2 and values.shape[1] >= 3:
        intensity = (
            0.299 * values[:, 0]
            + 0.587 * values[:, 1]
            + 0.114 * values[:, 2]
        )
    else:
        raise ValueError('点云颜色格式无效，无法转换为 intensity')
    finite = np.isfinite(intensity)
    normalized = np.zeros(point_count, dtype=np.float32)
    if finite.any():
        low, high = np.percentile(intensity[finite], (1.0, 99.0))
        normalized[finite] = np.clip(
            (intensity[finite] - low) / (high - low + 1e-9), 0.0, 1.0
        )
    gray = np.rint(normalized * 255.0).astype(np.uint8)
    return np.repeat(gray[:, None], 3, axis=1)


def projection_camera(params, transform_to_global, width, height):
    """构造与软件投影一致的 OpenCV 针孔内外参。"""
    yaw, pitch, roll = (
        np.radians(float(params[key])) for key in ('yaw', 'pitch', 'roll')
    )
    rotation_local = _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)
    global_to_local = np.linalg.inv(
        np.asarray(transform_to_global, dtype=np.float64).reshape(4, 4)
    )
    # 软件相机为 x 前、y 左、z 上；转换为 OpenCV 的 x 右、y 下、z 前。
    axis = np.array(((0, -1, 0), (0, 0, -1), (1, 0, 0)), dtype=np.float64)
    camera_position = np.array(
        (params['tx'], params['ty'], params['tz']), dtype=np.float64
    )
    local_rotation = rotation_local.T
    rotation = axis @ local_rotation @ global_to_local[:3, :3]
    translation = axis @ local_rotation @ (
        global_to_local[:3, 3] - camera_position
    )
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = translation

    focal = _focal_from_fov(width, height, float(params['fov']))
    intrinsic = np.array(
        ((focal, 0, width * 0.5), (0, focal, height * 0.5), (0, 0, 1)),
        dtype=np.float64,
    )
    return intrinsic, extrinsic


def render_projection(
    points,
    colors,
    transform_to_global,
    params,
    image_size=(1024, 576),
    *,
    crop_subject=False,
):
    """将全局点云渲染为彩色针孔投影，并返回深度、索引图及相机参数。"""
    width, height = map(int, image_size)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    intrinsic, extrinsic = projection_camera(
        params, transform_to_global, width, height
    )
    camera = pts @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    z = camera[:, 2]
    finite = np.isfinite(camera).all(axis=1)
    distance = np.linalg.norm(camera, axis=1)
    valid = (
        finite
        & (z > float(params.get('near', 0.3)))
        & (distance < float(params.get('far', 300.0)))
    )
    indices = np.flatnonzero(valid)
    camera, z = camera[valid], z[valid]

    offset_u = intrinsic[0, 0] * camera[:, 0] / z
    offset_v = intrinsic[1, 1] * camera[:, 1] / z
    if len(offset_u):
        if bool(params.get('center_h', True)):
            clipped = np.clip(offset_u, -2 * width, 2 * width)
            intrinsic[0, 2] = width * 0.5 - 0.5 * (
                np.percentile(clipped, 5.0) + np.percentile(clipped, 95.0)
            )
        if bool(params.get('center_v', False)):
            clipped = np.clip(offset_v, -2 * height, 2 * height)
            intrinsic[1, 2] = height * 0.5 - 0.5 * (
                np.percentile(clipped, 5.0) + np.percentile(clipped, 95.0)
            )
    u = np.rint(offset_u + intrinsic[0, 2]).astype(np.int32)
    v = np.rint(offset_v + intrinsic[1, 2]).astype(np.int32)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z, indices = u[inside], v[inside], z[inside], indices[inside]

    rgb = _intensity_rgb(colors, len(pts))[indices]

    image = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.zeros((height, width), dtype=np.float64)
    point_index = np.full((height, width), -1, dtype=np.int32)
    order = np.argsort(z)[::-1]
    radius = max(0, int(params.get('point_size', 3)) // 2)
    for dv in range(-radius, radius + 1):
        for du in range(-radius, radius + 1):
            px = np.clip(u[order] + du, 0, width - 1)
            py = np.clip(v[order] + dv, 0, height - 1)
            image[py, px] = rgb[order, ::-1]
            depth[py, px] = z[order]
            point_index[py, px] = indices[order]

    if crop_subject:
        image, depth, point_index, intrinsic = _crop_subject(
            image, depth, point_index, intrinsic
        )
    return {
        'view_bgr': np.ascontiguousarray(image),
        'depth_image': np.ascontiguousarray(depth),
        'pixel_point_index': np.ascontiguousarray(point_index),
        'camera_matrix': intrinsic,
        'extrinsic': extrinsic,
        'cloud_points': np.asarray(points),
    }


def _crop_subject(image, depth, point_index, intrinsic):
    mask = (depth > 0).astype(np.uint8)
    if int(mask.sum()) < 100:
        return image, depth, point_index, intrinsic
    joined = cv2.dilate(mask, np.ones((11, 11), np.uint8), iterations=2)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    if count <= 1:
        return image, depth, point_index, intrinsic
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero((labels == component) & (mask > 0))
    if len(xs) < 100:
        return image, depth, point_index, intrinsic
    margin_x = max(8, int((xs.max() - xs.min() + 1) * 0.04))
    margin_y = max(8, int((ys.max() - ys.min() + 1) * 0.04))
    x0 = max(0, int(xs.min()) - margin_x)
    x1 = min(image.shape[1], int(xs.max()) + margin_x + 1)
    y0 = max(0, int(ys.min()) - margin_y)
    y1 = min(image.shape[0], int(ys.max()) + margin_y + 1)
    adjusted = intrinsic.copy()
    adjusted[0, 2] -= x0
    adjusted[1, 2] -= y0
    return (
        image[y0:y1, x0:x1],
        depth[y0:y1, x0:x1],
        point_index[y0:y1, x0:x1],
        adjusted,
    )


__all__ = [
    'default_projection_params',
    'projection_camera',
    'projection_viewport_camera',
    'render_projection',
]
