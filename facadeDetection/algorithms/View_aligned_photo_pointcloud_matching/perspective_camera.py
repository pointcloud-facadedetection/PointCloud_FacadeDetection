"""扫描仪局部坐标系下的针孔投影、自动取景与主体裁剪。"""

from __future__ import annotations

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


def default_projection_params(points, transform_to_global):
    """按 `perspective_view_gui.py` 的主体密度思想生成初始投影参数。"""
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
        (radius <= np.percentile(radius, 88.0))
        & (finite[:, 2] >= z_low)
        & (finite[:, 2] <= z_high)
    ]
    if len(core) < 100:
        core = finite

    azimuth = np.degrees(np.arctan2(core[:, 1], core[:, 0]))
    yaw_grid = np.arange(-180.0, 180.0, 3.0)
    half_fov = 50.0
    counts = [
        np.sum(np.abs((azimuth - yaw + 180.0) % 360.0 - 180.0) <= half_fov)
        for yaw in yaw_grid
    ]
    yaw = float(yaw_grid[int(np.argmax(counts))])
    direction = np.array(
        (np.cos(np.radians(yaw)), np.sin(np.radians(yaw))),
        dtype=np.float64,
    )
    camera_xy = -direction * 2.0
    distance = np.linalg.norm(
        core - np.array((camera_xy[0], camera_xy[1], 1.75)), axis=1
    )
    return {
        'fov': 100.0,
        'yaw': yaw,
        'pitch': 0.0,
        'roll': 0.0,
        'tx': float(camera_xy[0]),
        'ty': float(camera_xy[1]),
        'tz': 1.75,
        'near': 0.3,
        'far': float(np.clip(np.percentile(distance, 99.0) * 1.15, 10.0, 300.0)),
        'point_size': 3,
    }


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
    valid = (
        finite
        & (z > float(params.get('near', 0.3)))
        & (z < float(params.get('far', 300.0)))
    )
    indices = np.flatnonzero(valid)
    camera, z = camera[valid], z[valid]

    u = np.rint(
        intrinsic[0, 0] * camera[:, 0] / z + intrinsic[0, 2]
    ).astype(np.int32)
    v = np.rint(
        intrinsic[1, 1] * camera[:, 1] / z + intrinsic[1, 2]
    ).astype(np.int32)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z, indices = u[inside], v[inside], z[inside], indices[inside]

    if colors is None or len(colors) != len(pts):
        rgb = np.full((len(pts), 3), 220, dtype=np.uint8)
    else:
        rgb = np.asarray(colors)
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = np.clip(rgb * 255.0, 0, 255)
        rgb = rgb.astype(np.uint8, copy=False)
    rgb = rgb[indices]

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
    'render_projection',
]
