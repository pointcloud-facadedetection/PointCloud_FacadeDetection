"""手动 2D-3D 标注点对：估算匹配矩阵，并将 3D 点重映射回照片。"""

from __future__ import annotations

import numpy as np

from ..photo_pointcloud_matching import solve_camera_pose

MIN_MATCH_PAIRS = 6


def _as_object_points(object_points) -> np.ndarray:
    points = np.asarray(object_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('3D 点必须是 N×3 数组')
    return points


def _as_image_points(image_points) -> np.ndarray:
    points = np.asarray(image_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError('2D 点必须是 N×2 数组')
    return points


def estimate_match_matrix(
    object_points,
    image_points,
    image_width: int,
    image_height: int,
) -> dict:
    """
    用不少于 6 对 2D-3D 点估计匹配矩阵。

    匹配矩阵为 3×4 投影矩阵 P = K [R | t]，可将齐次三维点映射到像素。
    """
    object_xyz = _as_object_points(object_points)
    image_xy = _as_image_points(image_points)
    if len(object_xyz) != len(image_xy):
        raise ValueError('2D 点与 3D 点数量不一致')
    if len(object_xyz) < MIN_MATCH_PAIRS:
        raise ValueError(f'估算匹配矩阵至少需要 {MIN_MATCH_PAIRS} 对完整匹配点')
    if int(image_width) <= 0 or int(image_height) <= 0:
        raise ValueError('缺少照片原始宽高')

    result = solve_camera_pose(
        object_points=object_xyz,
        image_points=image_xy,
        image_width=int(image_width),
        image_height=int(image_height),
    )
    camera_matrix = np.asarray(result['camera_matrix'], dtype=np.float64).reshape(3, 3)
    extrinsic = np.asarray(result['extrinsic_matrix'], dtype=np.float64).reshape(3, 4)
    match_matrix = camera_matrix @ extrinsic
    result['match_matrix'] = match_matrix.tolist()
    result['projection_matrix'] = match_matrix.tolist()
    return result


def remap_cloud_points_to_photo(
    object_points,
    match_matrix=None,
    *,
    camera_matrix=None,
    rotation_matrix=None,
    translation_vector=None,
    image_width: int | None = None,
    image_height: int | None = None,
) -> dict:
    """
    用已估计的匹配矩阵，将 3D 标注点投影回照片像素坐标。
    """
    object_xyz = _as_object_points(object_points)
    if len(object_xyz) == 0:
        raise ValueError('没有可重映射的 3D 标注点')

    if match_matrix is not None:
        matrix = np.asarray(match_matrix, dtype=np.float64).reshape(3, 4)
    else:
        if camera_matrix is None or rotation_matrix is None or translation_vector is None:
            raise ValueError('请先估算匹配矩阵')
        camera = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        rotation = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
        translation = np.asarray(translation_vector, dtype=np.float64).reshape(3, 1)
        matrix = camera @ np.hstack([rotation, translation])

    homogeneous = np.column_stack(
        [object_xyz, np.ones(len(object_xyz), dtype=np.float64)]
    )
    projected = (matrix @ homogeneous.T).T
    depth = projected[:, 2]
    valid_depth = depth > 1e-9
    pixels = np.full((len(object_xyz), 2), np.nan, dtype=np.float64)
    pixels[valid_depth] = projected[valid_depth, :2] / depth[valid_depth, None]

    in_image = valid_depth.copy()
    if image_width is not None and image_height is not None:
        in_image &= (
            np.isfinite(pixels[:, 0])
            & np.isfinite(pixels[:, 1])
            & (pixels[:, 0] >= 0)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 0] < float(image_width))
            & (pixels[:, 1] < float(image_height))
        )

    return {
        'match_matrix': matrix.tolist(),
        'image_points': pixels.tolist(),
        'depths': depth.tolist(),
        'valid_mask': valid_depth.tolist(),
        'in_image_mask': in_image.tolist(),
        'valid_count': int(np.sum(valid_depth)),
        'in_image_count': int(np.sum(in_image)),
        'point_count': int(len(object_xyz)),
    }


__all__ = [
    'MIN_MATCH_PAIRS',
    'estimate_match_matrix',
    'remap_cloud_points_to_photo',
]
