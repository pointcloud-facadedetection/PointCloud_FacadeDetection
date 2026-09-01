"""将照片和点云投影校正到选中立面的正视视角。"""

from __future__ import annotations

# OpenCV 的 Python API 由二进制扩展动态导出。
# pylint: disable=no-member

import cv2
import numpy as np

from algorithms.geometry import plane_axes


def _as_bgr(image, name):
    value = np.asarray(image, dtype=np.uint8)
    if value.ndim == 2:
        value = cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f'{name}格式无效')
    return np.ascontiguousarray(value)


def _project(points, rotation, translation, camera_matrix):
    xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    camera = xyz @ rotation.T + translation
    if np.any(~np.isfinite(camera)) or np.any(camera[:, 2] <= 1e-6):
        raise ValueError('选中立面不在相机前方，无法调整照片角度')
    intrinsic = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    homogeneous = camera @ intrinsic.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def _quadrilateral_area(points):
    xy = np.asarray(points, dtype=np.float64).reshape(4, 2)
    return abs(
        0.5
        * float(
            np.dot(xy[:, 0], np.roll(xy[:, 1], -1))
            - np.dot(xy[:, 1], np.roll(xy[:, 0], -1))
        )
    )


def _homography(source, target, label):
    source = np.asarray(source, dtype=np.float64).reshape(4, 2)
    target = np.asarray(target, dtype=np.float64).reshape(4, 2)
    if not np.isfinite(source).all() or _quadrilateral_area(source) < 4.0:
        raise ValueError(f'{label}中的立面投影范围过小或已退化')
    matrix = cv2.getPerspectiveTransform(
        source.astype(np.float32),
        target.astype(np.float32),
    )
    if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-12:
        raise ValueError(f'{label}的正视变换矩阵无效')
    return matrix


def _facade_geometry(facade_points, plane_model, camera_center):
    points = np.asarray(facade_points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        raise ValueError('所选立面没有足够的有效点')

    plane = np.asarray(plane_model, dtype=np.float64).reshape(4).copy()
    norm = float(np.linalg.norm(plane[:3]))
    if not np.isfinite(plane).all() or norm < 1e-12:
        raise ValueError('所选立面的法向量无效')
    plane /= norm

    center = np.mean(points, axis=0)
    toward_camera = np.asarray(camera_center, dtype=np.float64).reshape(3) - center
    if float(np.dot(plane[:3], toward_camera)) < 0.0:
        plane *= -1.0

    facade_type = (
        'horizontal' if abs(float(plane[2])) > 0.85 else 'vertical_facade'
    )
    u_axis, v_axis = plane_axes(plane[:3], facade_type)
    uv = np.column_stack(((points - center) @ u_axis, (points - center) @ v_axis))
    finite = np.isfinite(uv).all(axis=1)
    if int(finite.sum()) < 3:
        raise ValueError('所选立面的平面坐标无效')
    lower = np.min(uv[finite], axis=0)
    upper = np.max(uv[finite], axis=0)
    spans = upper - lower
    if np.any(spans < 1e-4):
        raise ValueError('所选立面的宽度或高度过小，无法生成正视图')

    boundary_uv = np.asarray(
        (
            (lower[0], lower[1]),
            (upper[0], lower[1]),
            (upper[0], upper[1]),
            (lower[0], upper[1]),
        ),
        dtype=np.float64,
    )
    boundary = (
        center
        + boundary_uv[:, 0, None] * u_axis
        + boundary_uv[:, 1, None] * v_axis
    )
    view_center = (
        center
        + 0.5 * (lower[0] + upper[0]) * u_axis
        + 0.5 * (lower[1] + upper[1]) * v_axis
    )
    return plane, u_axis, v_axis, boundary, view_center, spans


def _target_camera(normal, u_axis, v_axis, center, spans, output_size, margin):
    width, height = output_size
    content_width = width - 2 * margin
    content_height = height - 2 * margin
    pixels_per_metre = min(
        content_width / float(spans[0]),
        content_height / float(spans[1]),
    )
    distance = max(2.0, 2.0 * float(np.max(spans)))
    eye = center + normal * distance
    rotation = np.vstack((u_axis, -v_axis, -normal))
    if float(np.linalg.det(rotation)) < 0.0:
        rotation[0] *= -1.0
    translation = -rotation @ eye
    focal = pixels_per_metre * distance
    intrinsic = np.asarray(
        (
            (focal, 0.0, width * 0.5),
            (0.0, focal, height * 0.5),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    return rotation, translation, intrinsic


def rectify_facade_views(
    photo_bgr,
    cloud_view_bgr,
    facade_points,
    plane_model,
    *,
    photo_rotation,
    photo_translation,
    photo_camera_matrix,
    photo_distortion=None,
    cloud_rotation,
    cloud_translation,
    cloud_camera_matrix,
    target_max_dim=1600,
    margin_ratio=0.06,
):
    """把照片与点云映射图变换到同一立面正视相机。

    返回的是无热力图底图以及目标相机参数；调用方可用目标相机重新投影
    热力图，避免把色标随单应变换拉伸。
    """
    photo = _as_bgr(photo_bgr, '照片')
    cloud_view = _as_bgr(cloud_view_bgr, '点云映射图')
    photo_rotation = np.asarray(photo_rotation, dtype=np.float64).reshape(3, 3)
    photo_translation = np.asarray(photo_translation, dtype=np.float64).reshape(3)
    photo_camera = np.asarray(photo_camera_matrix, dtype=np.float64).reshape(3, 3)
    camera_center = -photo_rotation.T @ photo_translation

    plane, u_axis, v_axis, boundary, view_center, spans = _facade_geometry(
        facade_points,
        plane_model,
        camera_center,
    )

    maximum = int(np.clip(target_max_dim, 320, 4096))
    margin_ratio = float(np.clip(margin_ratio, 0.0, 0.2))
    content_limit = max(64, int(round(maximum * (1.0 - 2.0 * margin_ratio))))
    if spans[0] >= spans[1]:
        content_width = content_limit
        content_height = max(64, int(round(content_limit * spans[1] / spans[0])))
    else:
        content_height = content_limit
        content_width = max(64, int(round(content_limit * spans[0] / spans[1])))
    margin = max(8, int(round(maximum * margin_ratio)))
    output_size = (content_width + 2 * margin, content_height + 2 * margin)

    target_rotation, target_translation, target_camera = _target_camera(
        plane[:3],
        u_axis,
        v_axis,
        view_center,
        spans,
        output_size,
        margin,
    )
    destination = _project(
        boundary,
        target_rotation,
        target_translation,
        target_camera,
    )

    distortion = np.asarray(
        np.zeros(5) if photo_distortion is None else photo_distortion,
        dtype=np.float64,
    ).reshape(-1)
    if np.any(np.abs(distortion) > 1e-12):
        photo = cv2.undistort(photo, photo_camera, distortion, None, photo_camera)
    photo_boundary = _project(
        boundary,
        photo_rotation,
        photo_translation,
        photo_camera,
    )
    cloud_boundary = _project(
        boundary,
        cloud_rotation,
        cloud_translation,
        cloud_camera_matrix,
    )
    photo_h = _homography(photo_boundary, destination, '照片')
    cloud_h = _homography(cloud_boundary, destination, '点云映射图')

    flags = cv2.INTER_CUBIC
    border = cv2.BORDER_CONSTANT
    return {
        'photo_bgr': cv2.warpPerspective(
            photo, photo_h, output_size, flags=flags, borderMode=border
        ),
        'cloud_bgr': cv2.warpPerspective(
            cloud_view, cloud_h, output_size, flags=flags, borderMode=border
        ),
        'camera_matrix': target_camera,
        'rotation_matrix': target_rotation,
        'translation_vector': target_translation,
        'photo_homography': photo_h,
        'cloud_homography': cloud_h,
        'boundary_points_3d': boundary,
        'output_size': output_size,
        'plane_model': plane,
    }


__all__ = ['rectify_facade_views']
