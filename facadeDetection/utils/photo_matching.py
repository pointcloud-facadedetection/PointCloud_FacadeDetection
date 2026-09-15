"""2D-3D 对齐：点云映射、照片匹配、立面热力图正视贴图；主入口为 run_photo_facade_heatmap。

本文件只承担算法实现，不依赖 View_aligned_photo_pointcloud_matching。
调用方负责读盘、缓存和 UI。
"""

from __future__ import annotations

import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

_POSE_TRANSFORM_KEYS = (
    'transformToGlobal',
    'transform_to_global',
)


@dataclass(frozen=True)
class CloudMappingInput:
    """算法输入：原始三维点云 + 扫描仪位姿。"""

    points: np.ndarray
    scan_pose: Any
    colors: np.ndarray | None = None
    image_size: tuple[int, int] = (1024, 576)
    crop_subject: bool = True


@dataclass(frozen=True)
class CloudMappingResult:
    """算法输出：自动取景后的点云映射图，以及后续匹配所需的相机信息。"""

    mapping_image: np.ndarray
    depth_image: np.ndarray
    pixel_point_index: np.ndarray
    camera_matrix: np.ndarray
    extrinsic: np.ndarray


@dataclass(frozen=True)
class PhotoMatchResult:
    """算法输出：照片与点云的 3×4 匹配矩阵 P = K[R|t]。"""

    match_matrix: np.ndarray
    camera_matrix: np.ndarray
    extrinsic: np.ndarray
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    correspondences: list
    inlier_count: int
    point_count: int
    reprojection_rmse_px: float


@dataclass(frozen=True)
class FacadeHeatmapOverlayResult:
    """算法输出：立面正视矩形图上贴好热力图后的图片。"""

    image_bgr: np.ndarray
    rectified_photo_bgr: np.ndarray
    heatmap_warped_bgr: np.ndarray
    camera_matrix: np.ndarray
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    photo_homography: np.ndarray
    boundary_points_3d: np.ndarray
    output_size: tuple[int, int]


def generate_cloud_mapping_image(
    point_cloud,
    scan_pose,
    *,
    colors=None,
    image_size=(1024, 576),
    crop_subject=True,
) -> CloudMappingResult:
    """根据扫描仪位姿自动调整观察角度，生成可与照片自动匹配的映射图。

    Parameters
    ----------
    point_cloud:
        原始三维点云，形状 ``(N, 3)`` 的世界坐标。
    scan_pose:
        扫描仪位姿。可为 ``4x4`` ``transformToGlobal``，或含该字段的字典。
    colors:
        可选强度或 RGB。``None`` 时按相机深度着色。
    image_size:
        映射图画幅 ``(width, height)``。
    crop_subject:
        是否裁到点云主体，便于与照片自动匹配。

    Returns
    -------
    CloudMappingResult
        ``mapping_image`` 为 BGR 映射图；同时给出深度、像素到点索引
        以及针孔内外参，供后续 2D-3D 自动匹配使用。
    """
    points = _as_points(point_cloud)
    transform = _as_transform_to_global(scan_pose)
    values = _as_optional_colors(colors, len(points))
    width, height = _as_image_size(image_size)

    params = default_projection_params(
        points,
        transform,
        image_size=(width, height),
    )
    rendered = render_projection(
        points,
        values,
        transform,
        params,
        image_size=(width, height),
        crop_subject=crop_subject,
    )
    return CloudMappingResult(
        mapping_image=np.ascontiguousarray(rendered['view_bgr']),
        depth_image=np.ascontiguousarray(rendered['depth_image']),
        pixel_point_index=np.ascontiguousarray(rendered['pixel_point_index']),
        camera_matrix=np.asarray(rendered['camera_matrix'], dtype=np.float64),
        extrinsic=np.asarray(rendered['extrinsic'], dtype=np.float64),
    )


def generate_cloud_mapping_image_from_input(
    payload: CloudMappingInput,
) -> CloudMappingResult:
    """结构化输入版本，语义与 ``generate_cloud_mapping_image`` 相同。"""
    return generate_cloud_mapping_image(
        payload.points,
        payload.scan_pose,
        colors=payload.colors,
        image_size=payload.image_size,
        crop_subject=payload.crop_subject,
    )


def estimate_photo_cloud_match(
    point_cloud,
    photo,
    mapping_image,
    camera_matrix,
    extrinsic,
    *,
    depth_image=None,
    pixel_point_index=None,
) -> PhotoMatchResult:
    """用映射图与照片做 SuperPoint+LightGlue 自动匹配，输出匹配矩阵。

    Parameters
    ----------
    point_cloud:
        原始三维点云，形状 ``(N, 3)``。
    photo:
        照片 BGR 图像。
    mapping_image:
        ``generate_cloud_mapping_image`` 得到的映射图；也可直接传入
        ``CloudMappingResult``，此时内外参等字段可省略。
    camera_matrix:
        映射图针孔内参 ``3x3``。
    extrinsic:
        映射图针孔外参 ``4x4``。
    depth_image:
        映射图对应的相机深度。
    pixel_point_index:
        映射图像素到点云行号。

    Returns
    -------
    PhotoMatchResult
        ``match_matrix`` 为 ``3x4`` 投影矩阵 ``P = K[R|t]``。
    """
    if isinstance(mapping_image, CloudMappingResult):
        mapping = mapping_image
        mapping_image = mapping.mapping_image
        camera_matrix = (
            mapping.camera_matrix if camera_matrix is None else camera_matrix
        )
        extrinsic = mapping.extrinsic if extrinsic is None else extrinsic
        depth_image = (
            mapping.depth_image if depth_image is None else depth_image
        )
        pixel_point_index = (
            mapping.pixel_point_index
            if pixel_point_index is None
            else pixel_point_index
        )

    points = _as_points(point_cloud)
    photo_bgr = _as_bgr_image(photo, '照片')
    view_bgr = _as_bgr_image(mapping_image, '点云映射图')
    if camera_matrix is None or extrinsic is None:
        raise ValueError('自动匹配需要映射图的相机内参与外参')
    if depth_image is None:
        raise ValueError('自动匹配需要映射图对应的深度图')

    matched = _match_photo_to_cloud_view(
        photo_bgr,
        view_bgr,
        depth_image,
        camera_matrix,
        extrinsic,
        cloud_points=points,
        pixel_point_index=pixel_point_index,
    )
    if not matched.get('pose_estimated'):
        raise ValueError(
            f'有效 2D-3D 点对不足 {MIN_MATCH_PAIRS} 对，无法估计匹配矩阵；'
            f'当前 {matched.get("point_count", 0)} 对'
        )
    matrix = np.asarray(
        matched.get('match_matrix') or matched.get('projection_matrix'),
        dtype=np.float64,
    ).reshape(3, 4)
    return PhotoMatchResult(
        match_matrix=matrix,
        camera_matrix=np.asarray(
            matched['camera_matrix'], dtype=np.float64
        ).reshape(3, 3),
        extrinsic=np.asarray(
            matched['extrinsic_matrix'], dtype=np.float64
        ).reshape(3, 4),
        rotation_matrix=np.asarray(
            matched['rotation_matrix'], dtype=np.float64
        ).reshape(3, 3),
        translation_vector=np.asarray(
            matched['translation_vector'], dtype=np.float64
        ).reshape(3),
        correspondences=list(matched.get('correspondences') or []),
        inlier_count=int(matched.get('inlier_count', 0)),
        point_count=int(matched.get('point_count', 0)),
        reprojection_rmse_px=float(matched.get('reprojection_rmse_px', 0.0)),
    )


def overlay_facade_heatmap_on_photo(
    facade,
    heatmap,
    photo,
    match_matrix,
    *,
    point_cloud=None,
    alpha=0.72,
    target_max_dim=1600,
    margin_ratio=0.06,
    crop_padding=16,
    distortion=None,
) -> FacadeHeatmapOverlayResult:
    """从照片截取选中立面并调正为矩形，再把热力图贴到对应位置。

    Parameters
    ----------
    facade:
        选中立面。需含 ``plane_model``；三维范围可来自立面点、热力图采样，
        或 ``bbox_2d``。
    heatmap:
        该立面热力图。优先使用 ``grid_layout`` 中的 ``patch_bgr`` /
        ``patch_mask`` / ``corners_3d``；也可直接传入网格图。
    photo:
        原始照片，BGR。
    match_matrix:
        点云-照片匹配结果。可为 ``3x4`` 投影矩阵 ``P = K[R|t]``，
        或 ``PhotoMatchResult`` / 含内外参的字典。
    point_cloud:
        可选原始点云。仅当立面只有内点索引、热力图也没有三维采样时需要。
    crop_padding:
        立面投影外接框四周各扩展的像素数，默认 16（约 10–20）。
    """
    photo_bgr = _as_bgr_image(photo, '照片')
    camera_matrix, rotation, translation, pose_distortion = (
        _photo_pose_from_match(match_matrix)
    )
    if distortion is None:
        distortion = pose_distortion

    facade_points, plane_model = _resolve_facade_points_and_plane(
        facade, heatmap, point_cloud
    )
    grid_bgr, grid_mask, grid_corners = _resolve_heatmap_grid(heatmap)

    aligned = _rectify_photo_to_facade(
        photo_bgr,
        facade_points,
        plane_model,
        rotation,
        translation,
        camera_matrix,
        distortion=distortion,
        target_max_dim=target_max_dim,
        margin_ratio=margin_ratio,
        crop_padding=crop_padding,
    )
    blended, warped = _overlay_heatmap_on_rectified(
        aligned['photo_bgr'],
        grid_bgr,
        grid_mask,
        grid_corners,
        aligned['rotation_matrix'],
        aligned['translation_vector'],
        aligned['camera_matrix'],
        aligned['boundary_points_3d'],
        alpha=alpha,
    )
    return FacadeHeatmapOverlayResult(
        image_bgr=blended,
        rectified_photo_bgr=aligned['photo_bgr'],
        heatmap_warped_bgr=warped,
        camera_matrix=aligned['camera_matrix'],
        rotation_matrix=aligned['rotation_matrix'],
        translation_vector=aligned['translation_vector'],
        photo_homography=aligned['photo_homography'],
        boundary_points_3d=aligned['boundary_points_3d'],
        output_size=aligned['output_size'],
    )


@dataclass(frozen=True)
class PhotoFacadeHeatmapArgs:
    """主函数可选参数。未给出的项使用默认值。"""

    colors: Any = None
    mapping_image_size: tuple[int, int] = (1024, 576)
    crop_subject: bool = True
    alpha: float = 0.72
    target_max_dim: int = 1600
    margin_ratio: float = 0.06
    crop_padding: int = 16
    distortion: Any = None


def run_photo_facade_heatmap(
    point_cloud,
    photo,
    scan_pose,
    facade,
    args=None,
) -> np.ndarray:
    """由原始点云、照片、扫描仪位姿和选中立面，输出贴好热力图的正视图片。

    Parameters
    ----------
    point_cloud:
        原始三维点云，形状 ``(N, 3)``。
    photo:
        原始照片，BGR。
    scan_pose:
        扫描仪位姿。可为 ``4x4`` ``transformToGlobal``，或含该字段的字典。
    facade:
        选中立面信息。需含平面位置（如 ``plane_model``、立面点或
        ``bbox_2d``）以及该立面热力图（``grid_layout`` / ``heatmap`` /
        ``heatmap_data``）。也可写成
        ``{'facade': 立面, 'heatmap': 热力图}``。
    args:
        其余参数。可为 ``PhotoFacadeHeatmapArgs``、字典或带同名属性的对象。

    Returns
    -------
    numpy.ndarray
        贴了热力图的正视矩形图，BGR。
    """
    options = _as_photo_facade_args(args)
    photo_bgr = _as_bgr_image(photo, '照片')
    mapping = generate_cloud_mapping_image(
        point_cloud,
        scan_pose,
        colors=options.colors,
        image_size=options.mapping_image_size,
        crop_subject=options.crop_subject,
    )
    match = estimate_photo_cloud_match(
        point_cloud,
        photo_bgr,
        mapping,
        None,
        None,
    )
    facade_info, heatmap = _split_facade_and_heatmap(facade)
    overlay = overlay_facade_heatmap_on_photo(
        facade_info,
        heatmap,
        photo_bgr,
        match,
        point_cloud=point_cloud,
        alpha=options.alpha,
        target_max_dim=options.target_max_dim,
        margin_ratio=options.margin_ratio,
        crop_padding=options.crop_padding,
        distortion=options.distortion,
    )
    return overlay.image_bgr


def _as_photo_facade_args(args) -> PhotoFacadeHeatmapArgs:
    if args is None:
        return PhotoFacadeHeatmapArgs()
    if isinstance(args, PhotoFacadeHeatmapArgs):
        return args

    def _get(key, default):
        if isinstance(args, Mapping):
            if key in args:
                return args[key]
            if key == 'mapping_image_size' and 'image_size' in args:
                return args['image_size']
            return default
        if hasattr(args, key):
            return getattr(args, key)
        if key == 'mapping_image_size' and hasattr(args, 'image_size'):
            return getattr(args, 'image_size')
        return default

    defaults = PhotoFacadeHeatmapArgs()
    return PhotoFacadeHeatmapArgs(
        colors=_get('colors', defaults.colors),
        mapping_image_size=_get(
            'mapping_image_size', defaults.mapping_image_size
        ),
        crop_subject=bool(_get('crop_subject', defaults.crop_subject)),
        alpha=float(_get('alpha', defaults.alpha)),
        target_max_dim=int(_get('target_max_dim', defaults.target_max_dim)),
        margin_ratio=float(_get('margin_ratio', defaults.margin_ratio)),
        crop_padding=int(_get('crop_padding', defaults.crop_padding)),
        distortion=_get('distortion', defaults.distortion),
    )


def _split_facade_and_heatmap(facade):
    if facade is None:
        raise ValueError('缺少选中立面信息')
    if not isinstance(facade, Mapping):
        raise ValueError('立面信息应为字典，需包含位置与热力图')
    nested = facade.get('facade')
    heatmap = facade.get('heatmap')
    if heatmap is None:
        heatmap = facade.get('heatmap_data')
    if isinstance(nested, Mapping):
        return nested, facade if heatmap is None else heatmap
    return facade, facade if heatmap is None else heatmap


def _as_points(point_cloud) -> np.ndarray:
    if point_cloud is None:
        raise ValueError('原始点云不能为空')
    points = np.asarray(point_cloud, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f'原始点云应为 (N, 3)，实际形状为 {tuple(points.shape)}')
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite) == 0:
        raise ValueError('原始点云不包含有效坐标')
    return np.ascontiguousarray(points)


def _as_optional_colors(colors, point_count: int) -> np.ndarray | None:
    if colors is None:
        return None
    values = np.asarray(colors)
    if len(values) != point_count:
        raise ValueError(
            f'颜色/强度数量 {len(values)} 与点数 {point_count} 不一致'
        )
    return np.ascontiguousarray(values)


def _as_image_size(image_size: Sequence[int]) -> tuple[int, int]:
    if image_size is None or len(image_size) != 2:
        raise ValueError('image_size 应为 (width, height)')
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError('投影图像尺寸必须为正数')
    return width, height


def _as_transform_to_global(scan_pose) -> np.ndarray:
    """把扫描仪位姿规范为世界系 ``4x4 transformToGlobal``。"""
    if scan_pose is None:
        raise ValueError('扫描仪位姿不能为空')

    if isinstance(scan_pose, Mapping):
        transform = _first_present(scan_pose, _POSE_TRANSFORM_KEYS)
        if transform is None:
            nested = scan_pose.get('scan_pose_meta') or scan_pose.get('meta')
            if isinstance(nested, Mapping):
                transform = _first_present(nested, _POSE_TRANSFORM_KEYS)
        if transform is None:
            raise ValueError(
                '扫描仪位姿缺少 transformToGlobal，无法自动调整点云角度'
            )
        return _as_matrix4(transform)

    return _as_matrix4(scan_pose)


def _as_matrix4(value) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.size != 16:
        raise ValueError(f'transformToGlobal 应为 4x4，实际元素数 {matrix.size}')
    matrix = matrix.reshape(4, 4)
    if not np.isfinite(matrix).all():
        raise ValueError('transformToGlobal 包含无效数值')
    return matrix


def _first_present(data: Mapping, keys: tuple[str, ...]):
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


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


def _projection_scalars(colors, points, params, camera_xyz):
    """与测试查看器一致：有 intensity/RGB 则 1–99 分位归一化，否则用 depth。"""
    count = len(points)
    far = max(float(params.get('far', 300.0)), 1e-6)
    intensity = None
    if colors is not None and len(colors) == count:
        values = np.asarray(colors, dtype=np.float32)
        if values.ndim == 1 or (values.ndim == 2 and values.shape[1] == 1):
            intensity = values.reshape(-1)
        elif values.ndim == 2 and values.shape[1] >= 3:
            intensity = (
                0.299 * values[:, 0]
                + 0.587 * values[:, 1]
                + 0.114 * values[:, 2]
            )
        else:
            raise ValueError('点云颜色格式无效，无法转换为 intensity')
    if intensity is not None:
        finite = np.isfinite(intensity)
        normalized = np.zeros(count, dtype=np.float32)
        if finite.any():
            low, high = np.percentile(intensity[finite], (1.0, 99.0))
            if high - low > 1e-6:
                normalized[finite] = np.clip(
                    (intensity[finite] - low) / (high - low), 0.0, 1.0
                )
            else:
                scale = 1.0 if high <= 1.5 else 255.0
                normalized[finite] = np.clip(
                    intensity[finite] / scale, 0.0, 1.0
                )
        return normalized
    distance = np.linalg.norm(
        np.asarray(camera_xyz, dtype=np.float64).reshape(-1, 3), axis=1
    )
    return np.clip(distance / far, 0.0, 1.0).astype(np.float32)


def _intensity_image_to_bgr(image, gamma=1.0):
    """把带 NaN 空洞的 float intensity 图转成测试查看器同款灰阶 BGR。"""
    vis = np.asarray(image, dtype=np.float32)
    gamma = float(gamma)
    if np.isfinite(gamma) and abs(gamma - 1.0) > 1e-6:
        vis = np.power(vis, gamma)
    vis = np.nan_to_num(vis, nan=0.0)
    gray = np.rint(np.clip(vis, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


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
    """将全局点云渲染为与测试查看器一致的灰阶 intensity 针孔投影。"""
    width, height = map(int, image_size)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    intrinsic, extrinsic = projection_camera(
        params, transform_to_global, width, height
    )
    camera = pts @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    scalars_all = _projection_scalars(colors, pts, params, camera)
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
    scalars = scalars_all[indices]

    intensity = np.full((height, width), np.nan, dtype=np.float32)
    depth = np.zeros((height, width), dtype=np.float64)
    point_index = np.full((height, width), -1, dtype=np.int32)
    order = np.argsort(z)[::-1]
    radius = max(0, int(params.get('point_size', 3)) // 2)
    for dv in range(-radius, radius + 1):
        for du in range(-radius, radius + 1):
            px = np.clip(u[order] + du, 0, width - 1)
            py = np.clip(v[order] + dv, 0, height - 1)
            intensity[py, px] = scalars[order]
            depth[py, px] = z[order]
            point_index[py, px] = indices[order]
    image = _intensity_image_to_bgr(intensity, params.get('gamma', 1.0))

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


def _as_bgr_image(image, name: str) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f'{name}必须是灰度图或 BGR 彩色图')
    return np.ascontiguousarray(array[:, :, :3], dtype=np.uint8)


MIN_MATCH_PAIRS = 6
_MAX_PROCESSING_SIDE = 1600
_MAX_KEYPOINTS = 4096
_MATCH_CONFIDENCE = 0.10
_LIGHTGLUE_ENGINE = None
_CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / 'checkpoints'
_SUPERPOINT_CHECKPOINT = _CHECKPOINT_DIR / 'superpoint_v1.pth'
_LIGHTGLUE_CHECKPOINT = _CHECKPOINT_DIR / 'superpoint_lightglue_v0-1_arxiv.pth'
_CHECKPOINT_URLS = {
    _SUPERPOINT_CHECKPOINT: (
        'https://github.com/cvg/LightGlue/releases/download/'
        'v0.1_arxiv/superpoint_v1.pth'
    ),
    _LIGHTGLUE_CHECKPOINT: (
        'https://github.com/cvg/LightGlue/releases/download/'
        'v0.1_arxiv/superpoint_lightglue.pth'
    ),
}


def _ensure_checkpoints():
    """本地缺失时从 LightGlue 官方 Release 下载权重，并写入 checkpoints 目录。"""
    missing = [
        (path, url)
        for path, url in _CHECKPOINT_URLS.items()
        if not path.is_file()
    ]
    if not missing:
        return
    try:
        _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f'无法创建 SuperPoint + LightGlue 权重目录：{_CHECKPOINT_DIR}\n'
            f'原始错误：{exc}'
        ) from exc
    for path, url in missing:
        temporary = path.with_suffix(path.suffix + '.tmp')
        print(f'[PCFD] checkpoint.download url={url} dest={path}', flush=True)
        try:
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(path)
        except (OSError, urllib.error.URLError, ValueError) as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise RuntimeError(
                'SuperPoint + LightGlue 权重下载失败。\n'
                '首次自动匹配需要连接网络；下载成功后会保存到本地并离线复用。\n'
                f'目标目录：{_CHECKPOINT_DIR}\n'
                f'失败文件：{path.name}\n'
                f'下载地址：{url}\n'
                f'原始错误：{exc}'
            ) from exc
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(
                'SuperPoint + LightGlue 权重下载后文件无效。\n'
                f'目标文件：{path}\n'
                f'下载地址：{url}'
            )


def _resize_for_matching(image):
    array = np.asarray(image, dtype=np.uint8)
    height, width = array.shape[:2]
    scale = min(1.0, float(_MAX_PROCESSING_SIDE) / max(height, width))
    if scale < 1.0:
        array = cv2.resize(
            array,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return array, scale


def _get_lightglue_engine():
    """延迟加载本地 SuperPoint + LightGlue，并复用模型实例。"""
    global _LIGHTGLUE_ENGINE
    if _LIGHTGLUE_ENGINE is not None:
        return _LIGHTGLUE_ENGINE
    try:
        import torch
        from lightglue import LightGlue, SuperPoint
    except ImportError as exc:
        raise RuntimeError(
            '缺少 SuperPoint + LightGlue 依赖，请在项目根目录执行：\n'
            'pip install -r facadeDetection/requirements.txt'
        ) from exc

    _ensure_checkpoints()

    def load_local_checkpoint(url, *args, **kwargs):
        """将 LightGlue 构造函数的默认在线加载重定向到本地文件。"""
        checkpoint = (
            _SUPERPOINT_CHECKPOINT
            if 'superpoint_v1' in str(url)
            else _LIGHTGLUE_CHECKPOINT
        )
        return torch.load(checkpoint, map_location='cpu', weights_only=False)

    original_loader = torch.hub.load_state_dict_from_url
    torch.hub.load_state_dict_from_url = load_local_checkpoint
    try:
        extractor = SuperPoint(max_num_keypoints=_MAX_KEYPOINTS).eval()
        matcher = LightGlue(features='superpoint', filter_threshold=0.0).eval()
    except Exception as exc:
        raise RuntimeError(
            'SuperPoint + LightGlue 权重加载失败。\n'
            f'本地目录：{_CHECKPOINT_DIR}\n'
            f'原始错误：{exc}'
        ) from exc
    finally:
        torch.hub.load_state_dict_from_url = original_loader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = extractor.to(device)
    matcher = matcher.to(device)
    _LIGHTGLUE_ENGINE = (torch, extractor, matcher, device)
    return _LIGHTGLUE_ENGINE


def _image_tensor(image, torch, device):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return (
        torch.from_numpy(np.ascontiguousarray(rgb))
        .permute(2, 0, 1)
        .float()
        .div_(255.0)
        .to(device)
    )


def _feature_matches(photo_bgr, view_bgr):
    photo_image, photo_scale = _resize_for_matching(photo_bgr)
    view_image, view_scale = _resize_for_matching(view_bgr)
    torch, extractor, matcher, device = _get_lightglue_engine()
    view_tensor = _image_tensor(view_image, torch, device)
    photo_tensor = _image_tensor(photo_image, torch, device)
    with torch.inference_mode():
        view_features = extractor.extract(view_tensor)
        photo_features = extractor.extract(photo_tensor)
        matched = matcher({
            'image0': view_features,
            'image1': photo_features,
        })

    view_keys = view_features['keypoints'][0].detach().cpu().numpy()
    photo_keys = photo_features['keypoints'][0].detach().cpu().numpy()
    pairs = matched['matches'][0].detach().cpu().numpy()
    scores = matched['scores'][0].detach().cpu().numpy()
    keep_confident = scores >= _MATCH_CONFIDENCE
    pairs = pairs[keep_confident]
    view_xy = np.asarray(
        view_keys[pairs[:, 0]] if len(pairs) else [],
        dtype=np.float64,
    ).reshape(-1, 2) / view_scale
    photo_xy = np.asarray(
        photo_keys[pairs[:, 1]] if len(pairs) else [],
        dtype=np.float64,
    ).reshape(-1, 2) / photo_scale
    confident_count = len(pairs)

    if confident_count >= 4:
        homography, mask = cv2.findHomography(
            view_xy,
            photo_xy,
            cv2.RANSAC,
            5.0,
            maxIters=5000,
            confidence=0.995,
        )
        if homography is not None and mask is not None:
            keep = mask.reshape(-1).astype(bool)
            return (
                view_xy[keep],
                photo_xy[keep],
                confident_count,
                str(device),
            )
    return view_xy, photo_xy, confident_count, str(device)


def _depth_at(depth: np.ndarray, x: float, y: float, radius: int = 10):
    height, width = depth.shape
    cx, cy = int(round(x)), int(round(y))
    x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    patch = depth[y0:y1, x0:x1]
    yy, xx = np.nonzero(np.isfinite(patch) & (patch > 1e-8))
    if len(xx) == 0:
        return None
    distances = np.square(xx + x0 - x) + np.square(yy + y0 - y)
    index = int(np.argmin(distances))
    return float(patch[yy[index], xx[index]]), float(xx[index] + x0), float(yy[index] + y0)


def _build_pixel_point_index(points, intrinsic, extrinsic, image_shape):
    """建立当前视图每个像素对应的最近点云行号。"""
    cloud = np.asarray(points)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) == 0:
        return None
    height, width = image_shape
    rotation = extrinsic[:3, :3]
    translation = extrinsic[:3, 3]
    camera = cloud @ rotation.T + translation
    z = camera[:, 2]
    finite = np.isfinite(camera).all(axis=1) & (z > 1e-8)
    source_indices = np.flatnonzero(finite)
    camera = camera[finite]
    z = z[finite]
    x = np.rint(
        intrinsic[0, 0] * camera[:, 0] / z + intrinsic[0, 2]
    ).astype(np.int64)
    y = np.rint(
        intrinsic[1, 1] * camera[:, 1] / z + intrinsic[1, 2]
    ).astype(np.int64)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not np.any(inside):
        return None
    x, y, z = x[inside], y[inside], z[inside]
    source_indices = source_indices[inside]
    flat = y * width + x

    order = np.lexsort((z, flat))
    sorted_flat = flat[order]
    first = np.empty(len(order), dtype=bool)
    first[0] = True
    first[1:] = sorted_flat[1:] != sorted_flat[:-1]
    selected = order[first]
    index_map = np.full(height * width, -1, dtype=np.int32)
    index_map[flat[selected]] = source_indices[selected].astype(np.int32)
    return index_map.reshape(height, width)


def _point_index_at(index_map, x: float, y: float, radius: int = 10):
    height, width = index_map.shape
    cx, cy = int(round(x)), int(round(y))
    x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    patch = index_map[y0:y1, x0:x1]
    yy, xx = np.nonzero(patch >= 0)
    if len(xx) == 0:
        return None
    distances = np.square(xx + x0 - x) + np.square(yy + y0 - y)
    nearest = int(np.argmin(distances))
    return int(patch[yy[nearest], xx[nearest]])


def _world_point(pixel_x, pixel_y, depth, intrinsic, inverse_extrinsic):
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    camera_xyz = np.array(
        [
            (pixel_x - cx) * depth / fx,
            (pixel_y - cy) * depth / fy,
            depth,
            1.0,
        ],
        dtype=np.float64,
    )
    world = inverse_extrinsic @ camera_xyz
    return world[:3] / world[3]


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


def _estimate_camera_matrix(image_width, image_height, horizontal_fov_deg=60.0):
    width = float(image_width)
    height = float(image_height)
    if width <= 0 or height <= 0:
        raise ValueError('图像宽高必须大于 0')
    if not 10.0 <= float(horizontal_fov_deg) <= 150.0:
        raise ValueError('水平视场角应在 10° 到 150° 之间')
    focal = width / (2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0))
    return np.array(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _validate_pnp_points(object_points, image_points, minimum=4):
    object_points = np.asarray(object_points, dtype=np.float64)
    image_points = np.asarray(image_points, dtype=np.float64)
    if object_points.ndim != 2 or object_points.shape[1] != 3:
        raise ValueError('3D 点坐标必须是 N×3 数组')
    if image_points.ndim != 2 or image_points.shape[1] != 2:
        raise ValueError('2D 像素坐标必须是 N×2 数组')
    if len(object_points) != len(image_points):
        raise ValueError('2D 点与 3D 点数量不一致')
    if len(object_points) < minimum:
        raise ValueError(f'至少需要 {minimum} 对 2D-3D 匹配点')
    if not np.all(np.isfinite(object_points)) or not np.all(np.isfinite(image_points)):
        raise ValueError('匹配点中包含无效数值')
    return object_points, image_points


def _clamp_focal(focal, width, height, fallback):
    min_focal = max(width, height) * 0.15
    max_focal = max(width, height) * 10.0
    if not np.isfinite(focal) or not min_focal <= focal <= max_focal:
        return float(fallback)
    return float(focal)


def _estimate_camera_matrix_from_correspondences(
    object_points,
    image_points,
    image_width,
    image_height,
    horizontal_fov_deg=60.0,
):
    object_points, image_points = _validate_pnp_points(
        object_points, image_points, minimum=6
    )
    width = int(round(float(image_width)))
    height = int(round(float(image_height)))
    initial_matrix = _estimate_camera_matrix(width, height, horizontal_fov_deg)
    distortion = np.zeros((5, 1), dtype=np.float64)
    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_FIX_ASPECT_RATIO
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K1
        | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3
    )
    rms, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
        [object_points.astype(np.float32)],
        [image_points.astype(np.float32)],
        (width, height),
        initial_matrix.copy(),
        distortion,
        flags=flags,
    )
    focal = _clamp_focal(
        float(camera_matrix[0, 0]),
        width,
        height,
        initial_matrix[0, 0],
    )
    camera_matrix[0, 0] = focal
    camera_matrix[1, 1] = focal
    centered = object_points - np.mean(object_points, axis=0)
    point_rank = int(np.linalg.matrix_rank(centered))
    warning = None
    if point_rank < 3:
        warning = (
            '3D 匹配点近似共面；内参仅作粗略估计，'
            '后续将尝试多种 PnP 策略求解外参。'
        )
    return {
        'camera_matrix': camera_matrix,
        'distortion_coefficients': distortion,
        'calibration_rms_px': float(rms),
        'method': 'single_view_calibrateCamera_fixed_principal_aspect',
        'point_rank': point_rank,
        'warning': warning,
    }


def _estimate_camera_matrix_with_radial_distortion(
    object_points,
    image_points,
    image_width,
    image_height,
    horizontal_fov_deg=60.0,
):
    object_points, image_points = _validate_pnp_points(
        object_points, image_points, minimum=8
    )
    width = int(round(float(image_width)))
    height = int(round(float(image_height)))
    initial_matrix = _estimate_camera_matrix(width, height, horizontal_fov_deg)
    distortion = np.zeros((5, 1), dtype=np.float64)
    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_FIX_ASPECT_RATIO
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3
    )
    rms, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
        [object_points.astype(np.float32)],
        [image_points.astype(np.float32)],
        (width, height),
        initial_matrix.copy(),
        distortion,
        flags=flags,
    )
    focal = _clamp_focal(
        float(camera_matrix[0, 0]),
        width,
        height,
        initial_matrix[0, 0],
    )
    camera_matrix[0, 0] = focal
    camera_matrix[1, 1] = focal
    k1 = float(distortion[0, 0])
    if not np.isfinite(k1) or abs(k1) > 1.5:
        k1 = 0.0
        distortion[0, 0] = 0.0
    return {
        'camera_matrix': camera_matrix,
        'distortion_coefficients': distortion,
        'calibration_rms_px': float(rms),
        'method': 'single_view_calibrateCamera_k1',
        'point_rank': int(
            np.linalg.matrix_rank(object_points - np.mean(object_points, axis=0))
        ),
        'warning': None,
        'k1': k1,
    }


def _pnp_method_candidates():
    methods = []
    if hasattr(cv2, 'SOLVEPNP_IPPE'):
        methods.append(('IPPE', cv2.SOLVEPNP_IPPE))
    if hasattr(cv2, 'SOLVEPNP_SQPNP'):
        methods.append(('SQPNP', cv2.SOLVEPNP_SQPNP))
    methods.extend([
        ('EPNP', cv2.SOLVEPNP_EPNP),
        ('ITERATIVE', cv2.SOLVEPNP_ITERATIVE),
    ])
    return methods


def _collect_intrinsic_candidates(
    object_points,
    image_points,
    image_width,
    image_height,
    horizontal_fov_deg,
    camera_matrix=None,
):
    width = int(round(float(image_width)))
    height = int(round(float(image_height)))
    candidates = []
    zero_dist = np.zeros((5, 1), dtype=np.float64)
    if camera_matrix is not None:
        candidates.append((
            np.asarray(camera_matrix, dtype=np.float64),
            'provided',
            None,
            None,
            zero_dist,
        ))
        return candidates
    if len(object_points) >= 6:
        try:
            calibration = _estimate_camera_matrix_from_correspondences(
                object_points,
                image_points,
                image_width,
                image_height,
                horizontal_fov_deg,
            )
            candidates.append((
                calibration['camera_matrix'],
                calibration['method'],
                calibration.get('warning'),
                calibration.get('calibration_rms_px'),
                calibration['distortion_coefficients'],
            ))
        except (ValueError, cv2.error):
            pass
    if len(object_points) >= 8:
        try:
            calibration = _estimate_camera_matrix_with_radial_distortion(
                object_points,
                image_points,
                image_width,
                image_height,
                horizontal_fov_deg,
            )
            k1 = calibration.get('k1', 0.0)
            warning = None
            if abs(k1) > 0.05:
                warning = (
                    f'检测到径向畸变 k1={k1:.4f}；'
                    '已用于改善照片边缘/远距离区域对齐。'
                )
            candidates.append((
                calibration['camera_matrix'],
                calibration['method'],
                warning,
                calibration.get('calibration_rms_px'),
                calibration['distortion_coefficients'],
            ))
        except (ValueError, cv2.error):
            pass
    fov_guesses = sorted(set([
        float(horizontal_fov_deg),
        40.0, 50.0, 55.0, 60.0, 65.0, 70.0, 80.0, 90.0,
    ]))
    for fov in fov_guesses:
        candidates.append((
            _estimate_camera_matrix(width, height, fov),
            f'fov_guess_{fov:.0f}deg',
            '内参来自假定视场角，适用于手动标注粗略匹配。',
            None,
            zero_dist.copy(),
        ))
    return candidates


def _try_pnp_ransac(
    object_points,
    image_points,
    camera_matrix,
    distortion,
    reproj_err,
    flag,
    iterations,
    confidence,
):
    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points.astype(np.float64),
            image_points.astype(np.float64),
            camera_matrix,
            distortion,
            iterationsCount=int(iterations),
            reprojectionError=float(reproj_err),
            confidence=float(confidence),
            flags=int(flag),
        )
        if success and rvec is not None and tvec is not None:
            return rvec, tvec, inliers
    except cv2.error:
        pass
    return None, None, None


def _try_pnp_direct(
    object_points,
    image_points,
    camera_matrix,
    distortion,
    flag,
):
    try:
        success, rvec, tvec = cv2.solvePnP(
            object_points.astype(np.float64),
            image_points.astype(np.float64),
            camera_matrix,
            distortion,
            flags=int(flag),
        )
        if success and rvec is not None and tvec is not None:
            return rvec, tvec, np.arange(len(object_points), dtype=int).reshape(-1, 1)
    except cv2.error:
        pass
    return None, None, None


def _evaluate_pose(
    object_points,
    image_points,
    rvec,
    tvec,
    camera_matrix,
    distortion,
    inliers,
):
    inlier_indices = (
        inliers.reshape(-1).astype(int)
        if inliers is not None
        else np.arange(len(object_points), dtype=int)
    )
    if len(inlier_indices) < 3:
        return None
    if hasattr(cv2, 'solvePnPRefineLM'):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points[inlier_indices],
                image_points[inlier_indices],
                camera_matrix,
                distortion,
                rvec,
                tvec,
            )
        except cv2.error:
            pass
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, distortion
    )
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - image_points, axis=1)
    inlier_errors = errors[inlier_indices]
    inlier_rmse = float(np.sqrt(np.mean(np.square(inlier_errors))))
    overall_rmse = float(np.sqrt(np.mean(np.square(errors))))
    return {
        'rvec': rvec,
        'tvec': tvec,
        'inlier_indices': inlier_indices,
        'inlier_count': int(len(inlier_indices)),
        'inlier_rmse': inlier_rmse,
        'overall_rmse': overall_rmse,
        'errors': errors,
        'projected': projected,
    }


def _search_best_pose(
    object_points,
    image_points,
    intrinsic_candidates,
    distortion,
    reproj_thresholds,
    iterations,
    confidence,
    use_ransac=True,
):
    best = None
    best_meta = None
    best_score = None
    attempt = _try_pnp_ransac if use_ransac else _try_pnp_direct
    for (
        camera_matrix,
        intrinsic_method,
        intrinsic_warning,
        calibration_rms,
        candidate_distortion,
    ) in intrinsic_candidates:
        trial_distortion = np.asarray(candidate_distortion, dtype=np.float64).reshape(-1, 1)
        for method_name, flag in _pnp_method_candidates():
            thresholds = reproj_thresholds if use_ransac else [None]
            for reproj in thresholds:
                if use_ransac:
                    rvec, tvec, inliers = attempt(
                        object_points,
                        image_points,
                        camera_matrix,
                        trial_distortion,
                        reproj,
                        flag,
                        iterations,
                        confidence,
                    )
                else:
                    rvec, tvec, inliers = attempt(
                        object_points,
                        image_points,
                        camera_matrix,
                        trial_distortion,
                        flag,
                    )
                if rvec is None:
                    continue
                evaluated = _evaluate_pose(
                    object_points,
                    image_points,
                    rvec,
                    tvec,
                    camera_matrix,
                    trial_distortion,
                    inliers,
                )
                if evaluated is None:
                    continue
                score = (evaluated['inlier_count'], -evaluated['inlier_rmse'])
                if best_score is None or score > best_score:
                    best = evaluated
                    best_score = score
                    best_meta = {
                        'camera_matrix': camera_matrix,
                        'distortion_coefficients': trial_distortion,
                        'intrinsic_method': intrinsic_method,
                        'intrinsic_warning': intrinsic_warning,
                        'calibration_rms_px': calibration_rms,
                        'pnp_method': method_name,
                        'pnp_mode': 'ransac' if use_ransac else 'direct',
                        'reprojection_threshold_px': reproj,
                    }
    return best, best_meta


def _solve_camera_pose(
    object_points,
    image_points,
    image_width,
    image_height,
    camera_matrix=None,
    distortion_coefficients=None,
    horizontal_fov_deg=60.0,
    reprojection_error_px=20.0,
    confidence=0.95,
    iterations=500,
):
    object_points, image_points = _validate_pnp_points(object_points, image_points)
    distortion = (
        np.zeros((5, 1), dtype=np.float64)
        if distortion_coefficients is None
        else np.asarray(distortion_coefficients, dtype=np.float64).reshape(-1, 1)
    )
    intrinsic_candidates = _collect_intrinsic_candidates(
        object_points,
        image_points,
        image_width,
        image_height,
        horizontal_fov_deg,
        camera_matrix=camera_matrix,
    )
    reproj_thresholds = sorted(set([
        float(reprojection_error_px),
        12.0, 20.0, 30.0, 45.0, 60.0, 80.0,
    ]))
    best, best_meta = _search_best_pose(
        object_points,
        image_points,
        intrinsic_candidates,
        distortion,
        reproj_thresholds,
        iterations,
        confidence,
        use_ransac=True,
    )
    if best is None:
        best, best_meta = _search_best_pose(
            object_points,
            image_points,
            intrinsic_candidates,
            distortion,
            reproj_thresholds,
            iterations,
            confidence,
            use_ransac=False,
        )
    if best is None or best_meta is None:
        raise ValueError(
            'PnP 求解失败，请检查 2D/3D 对应关系；'
            '可尝试增加分布更均匀的匹配点'
        )
    rotation, _ = cv2.Rodrigues(best['rvec'])
    extrinsic = np.hstack([rotation, best['tvec'].reshape(3, 1)])
    inlier_indices = best['inlier_indices']
    inlier_errors = best['errors'][inlier_indices]
    best_distortion = best_meta['distortion_coefficients']
    return {
        'camera_matrix': best_meta['camera_matrix'].tolist(),
        'distortion_coefficients': best_distortion.reshape(-1).tolist(),
        'intrinsic_method': best_meta['intrinsic_method'],
        'intrinsic_warning': best_meta['intrinsic_warning'],
        'calibration_rms_px': best_meta['calibration_rms_px'],
        'pnp_method': best_meta['pnp_method'],
        'pnp_mode': best_meta['pnp_mode'],
        'reprojection_threshold_px': best_meta['reprojection_threshold_px'],
        'rotation_matrix': rotation.tolist(),
        'rotation_vector': best['rvec'].reshape(-1).tolist(),
        'translation_vector': best['tvec'].reshape(-1).tolist(),
        'extrinsic_matrix': extrinsic.tolist(),
        'projection_matrix': (best_meta['camera_matrix'] @ extrinsic).tolist(),
        'camera_center_world': (-rotation.T @ best['tvec'].reshape(3)).tolist(),
        'inlier_indices': inlier_indices.tolist(),
        'inlier_count': best['inlier_count'],
        'point_count': int(len(object_points)),
        'reprojection_rmse_px': float(np.sqrt(np.mean(np.square(inlier_errors)))),
        'reprojection_mean_px': float(np.mean(inlier_errors)),
        'reprojection_max_px': float(np.max(inlier_errors)),
        'projected_points': best['projected'].tolist(),
        'coordinate_convention': 'X_camera = R * X_world + T',
    }


def _estimate_match_matrix(
    object_points,
    image_points,
    image_width: int,
    image_height: int,
) -> dict:
    object_xyz = _as_object_points(object_points)
    image_xy = _as_image_points(image_points)
    if len(object_xyz) != len(image_xy):
        raise ValueError('2D 点与 3D 点数量不一致')
    if len(object_xyz) < MIN_MATCH_PAIRS:
        raise ValueError(f'估算匹配矩阵至少需要 {MIN_MATCH_PAIRS} 对完整匹配点')
    if int(image_width) <= 0 or int(image_height) <= 0:
        raise ValueError('缺少照片原始宽高')
    result = _solve_camera_pose(
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


def _match_photo_to_cloud_view(
    photo_bgr,
    view_bgr,
    depth_image,
    view_camera_matrix,
    view_extrinsic,
    cloud_points=None,
    pixel_point_index=None,
) -> dict:
    """通过点云映射图生成照片像素与世界三维点对应关系，并估计匹配矩阵。"""
    photo = np.asarray(photo_bgr, dtype=np.uint8)
    view = np.asarray(view_bgr, dtype=np.uint8)
    depth = np.asarray(depth_image, dtype=np.float64)
    intrinsic = np.asarray(view_camera_matrix, dtype=np.float64).reshape(3, 3)
    extrinsic = np.asarray(view_extrinsic, dtype=np.float64).reshape(4, 4)
    if photo.ndim != 3 or view.ndim != 3 or depth.ndim != 2:
        raise ValueError('自动匹配需要照片、点云映射图和深度图')
    if view.shape[:2] != depth.shape:
        raise ValueError('点云映射图与深度图尺寸不一致')

    (
        view_points,
        photo_points,
        ratio_match_count,
        inference_device,
    ) = _feature_matches(photo, view)
    lines = [
        f'[AutoMatch][SuperPoint+LightGlue][{inference_device}] '
        f'置信度匹配 {ratio_match_count} 对，'
        f'几何一致匹配 {len(view_points)} 对：'
    ]
    for sequence, (view_xy, photo_xy) in enumerate(
        zip(view_points, photo_points),
        start=1,
    ):
        lines.append(
            f'  #{sequence}: 点云图({view_xy[0]:.3f}, {view_xy[1]:.3f})'
            f' <-> 照片({photo_xy[0]:.3f}, {photo_xy[1]:.3f})'
        )
    print('\n'.join(lines), flush=True)

    inverse_extrinsic = np.linalg.inv(extrinsic)
    cloud = None
    point_index_map = (
        np.asarray(pixel_point_index, dtype=np.int32)
        if pixel_point_index is not None
        else None
    )
    if point_index_map is not None and point_index_map.shape != depth.shape:
        raise ValueError('像素点索引图与点云映射图尺寸不一致')
    if cloud_points is not None:
        cloud = np.asarray(cloud_points)
        if point_index_map is None:
            point_index_map = _build_pixel_point_index(
                cloud,
                intrinsic,
                extrinsic,
                depth.shape,
            )
    correspondences = []
    object_points = []
    image_points = []
    used_point_indices = set()
    for view_xy, photo_xy in zip(view_points, photo_points):
        point_index = (
            _point_index_at(point_index_map, view_xy[0], view_xy[1])
            if point_index_map is not None
            else None
        )
        if point_index is not None:
            if point_index in used_point_indices:
                continue
            used_point_indices.add(point_index)
            world = np.asarray(cloud[point_index], dtype=np.float64)
            depth_x, depth_y = float(view_xy[0]), float(view_xy[1])
            camera_xyz = extrinsic[:3, :3] @ world + extrinsic[:3, 3]
            z = float(camera_xyz[2])
            mapping_method = 'pixel_point_index'
        else:
            sample = _depth_at(depth, view_xy[0], view_xy[1])
            if sample is None:
                continue
            z, depth_x, depth_y = sample
            world = _world_point(
                depth_x,
                depth_y,
                z,
                intrinsic,
                inverse_extrinsic,
            )
            mapping_method = 'depth_buffer'
        if not np.isfinite(world).all():
            continue
        object_points.append(world)
        image_points.append(photo_xy)
        correspondences.append({
            'view_point': [float(view_xy[0]), float(view_xy[1])],
            'depth_point': [depth_x, depth_y],
            'image_point': photo_xy.tolist(),
            'object_point': world.tolist(),
            'view_depth': z,
            'cloud_index': point_index,
            'mapping_method': mapping_method,
        })

    result = {
        'correspondences': correspondences,
        'feature_match_count': int(ratio_match_count),
        'feature_algorithm': 'SuperPoint + LightGlue',
        'inference_device': inference_device,
        'match_confidence': float(_MATCH_CONFIDENCE),
        'geometric_match_count': int(len(view_points)),
        'depth_match_count': int(len(correspondences)),
        'point_count': int(len(correspondences)),
        'inlier_count': 0,
        'inlier_indices': [],
        'pose_estimated': False,
        'view_camera_matrix': intrinsic.tolist(),
        'view_extrinsic': extrinsic.tolist(),
        'view_image_size': [int(view.shape[1]), int(view.shape[0])],
        'photo_image_size': [int(photo.shape[1]), int(photo.shape[0])],
        'point_mapping_method': (
            'pixel_point_index'
            if point_index_map is not None
            else 'depth_buffer'
        ),
    }
    if len(correspondences) >= MIN_MATCH_PAIRS:
        pose = _estimate_match_matrix(
            object_points=object_points,
            image_points=image_points,
            image_width=photo.shape[1],
            image_height=photo.shape[0],
        )
        result.update(pose)
        result['pose_estimated'] = True
    return result


def _photo_pose_from_match(match):
    """把匹配结果规范为照片针孔相机的 K、R、t。"""
    if isinstance(match, PhotoMatchResult):
        return (
            np.asarray(match.camera_matrix, dtype=np.float64).reshape(3, 3),
            np.asarray(match.rotation_matrix, dtype=np.float64).reshape(3, 3),
            np.asarray(match.translation_vector, dtype=np.float64).reshape(3),
            None,
        )

    distortion = None
    matrix = None
    if isinstance(match, Mapping):
        distortion = match.get('distortion_coefficients')
        has_pose = all(
            match.get(key) is not None
            for key in (
                'camera_matrix',
                'rotation_matrix',
                'translation_vector',
            )
        )
        if has_pose:
            return (
                np.asarray(match['camera_matrix'], dtype=np.float64).reshape(3, 3),
                np.asarray(match['rotation_matrix'], dtype=np.float64).reshape(3, 3),
                np.asarray(
                    match['translation_vector'], dtype=np.float64
                ).reshape(3),
                distortion,
            )
        matrix = match.get('match_matrix')
        if matrix is None:
            matrix = match.get('projection_matrix')
    else:
        matrix = match

    if matrix is None:
        raise ValueError(
            '匹配矩阵应为 3x4 投影矩阵 P=K[R|t]，或含内外参的匹配结果'
        )
    camera_matrix, rotation, translation = _decompose_match_matrix(matrix)
    return camera_matrix, rotation, translation, distortion


def _decompose_match_matrix(match_matrix):
    projection = np.asarray(match_matrix, dtype=np.float64).reshape(3, 4)
    if not np.isfinite(projection).all():
        raise ValueError('匹配矩阵包含无效数值')

    camera_matrix, rotation, camera_center_h, *_ = cv2.decomposeProjectionMatrix(
        projection
    )
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    camera_matrix = camera_matrix / camera_matrix[2, 2]
    camera_center = (
        camera_center_h[:3, 0] / camera_center_h[3, 0]
    ).reshape(3)
    for axis in (0, 1):
        if camera_matrix[axis, axis] < 0.0:
            camera_matrix[:, axis] *= -1.0
            rotation[axis, :] *= -1.0
    if float(np.linalg.det(rotation)) < 0.0:
        rotation *= -1.0
    translation = -rotation @ camera_center
    return camera_matrix, rotation, translation


def _resolve_facade_points_and_plane(facade, heatmap, point_cloud=None):
    facade = facade or {}
    heatmap_data = heatmap if isinstance(heatmap, Mapping) else {}
    facade_id = facade.get('id')
    heatmap_id = heatmap_data.get('facade_id')
    if (
        facade_id is not None
        and heatmap_id is not None
        and int(facade_id) != int(heatmap_id)
    ):
        raise ValueError('当前热力图不属于所选立面，请重新生成热力图')

    plane = heatmap_data.get('plane_model')
    if plane is None:
        plane = facade.get('plane_model')
    if plane is None:
        raise ValueError('所选立面缺少拟合平面参数')
    plane = np.asarray(plane, dtype=np.float64).reshape(4)

    points = None
    for source in (heatmap_data, facade):
        if points is not None:
            break
        for key in ('points_3d', 'points'):
            points = _optional_points3(source.get(key))
            if points is not None:
                break
    if points is None and point_cloud is not None:
        cloud = _as_points(point_cloud)
        indices = np.asarray(
            facade.get('proxy_indices')
            or facade.get('inlier_indices')
            or [],
            dtype=np.int64,
        )
        indices = indices[(indices >= 0) & (indices < len(cloud))]
        if len(indices) >= 3:
            points = cloud[indices]
    if points is None:
        points = _facade_points_from_bbox(facade, plane, heatmap_data)
    if points is None or len(points) < 3:
        raise ValueError('所选立面没有足够的有效三维点，无法截取照片区域')
    return np.ascontiguousarray(points, dtype=np.float64), plane


def _optional_points3(value):
    if value is None:
        return None
    points = np.asarray(value, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        return None
    return points


def _facade_points_from_bbox(facade, plane, heatmap_data):
    layout = heatmap_data.get('grid_layout') if isinstance(heatmap_data, Mapping) else None
    if not isinstance(layout, Mapping):
        layout = heatmap_data if isinstance(heatmap_data, Mapping) else {}
    corners = layout.get('corners_3d')
    if corners is None and isinstance(heatmap_data, Mapping):
        corners = heatmap_data.get('corners_3d')
    corner_points = _optional_points3(corners)
    if corner_points is not None:
        return corner_points

    bbox = facade.get('bbox_2d') or {}
    u_axis = np.asarray(bbox.get('u_axis', []), dtype=np.float64).reshape(-1)
    v_axis = np.asarray(bbox.get('v_axis', []), dtype=np.float64).reshape(-1)
    if u_axis.size != 3 or v_axis.size != 3:
        facade_type = (
            'horizontal'
            if abs(float(plane[2])) > 0.85
            else 'vertical_facade'
        )
        u_axis, v_axis = _plane_axes(plane[:3], facade_type)
    try:
        u_min, u_max = float(bbox['u_min']), float(bbox['u_max'])
        v_min, v_max = float(bbox['v_min']), float(bbox['v_max'])
    except (KeyError, TypeError, ValueError):
        return None
    center = np.asarray(
        facade.get('center', (0.0, 0.0, 0.0)), dtype=np.float64
    ).reshape(3)
    return np.vstack(
        (
            center + u_min * u_axis + v_min * v_axis,
            center + u_max * u_axis + v_min * v_axis,
            center + u_max * u_axis + v_max * v_axis,
            center + u_min * u_axis + v_max * v_axis,
        )
    )


def _resolve_heatmap_grid(heatmap):
    if heatmap is None:
        raise ValueError('缺少立面热力图')
    if not isinstance(heatmap, Mapping):
        patch = _as_bgr_image(heatmap, '立面热力图')
        mask = np.full(patch.shape[:2], 255, dtype=np.uint8)
        return patch, mask, None

    layout = heatmap.get('grid_layout')
    if not isinstance(layout, Mapping):
        layout = {}
    patch = None
    for source in (layout, heatmap):
        for key in ('patch_bgr', 'grid_bgr'):
            patch = source.get(key)
            if patch is not None:
                break
        if patch is not None:
            break
    if patch is None:
        raise ValueError('立面热力图缺少可贴图的网格像素')
    patch = _as_bgr_image(patch, '立面热力图')
    mask = layout.get('patch_mask')
    if mask is None:
        mask = heatmap.get('patch_mask')
    if mask is None:
        mask = np.full(patch.shape[:2], 255, dtype=np.uint8)
    else:
        mask = np.asarray(mask, dtype=np.uint8)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.shape[:2] != patch.shape[:2]:
            raise ValueError('热力图与有效像素掩膜尺寸不一致')
    corners = layout.get('corners_3d')
    if corners is None:
        corners = heatmap.get('corners_3d')
    if corners is not None:
        corners = np.asarray(corners, dtype=np.float64).reshape(4, 3)
        if not np.isfinite(corners).all():
            raise ValueError('热力图三维四角坐标无效')
    return patch, np.ascontiguousarray(mask), corners


def _plane_axes(normal, facade_type=None):
    normal = np.asarray(normal, dtype=np.float64).reshape(3)
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    z_axis = np.array([0.0, 0.0, 1.0])
    if facade_type == 'vertical_facade' or abs(normal[2]) < 0.45:
        u_axis = np.cross(z_axis, normal)
        if np.linalg.norm(u_axis) < 1e-8:
            u_axis = np.array([1.0, 0.0, 0.0])
        u_axis = u_axis / (np.linalg.norm(u_axis) + 1e-12)
        v_axis = z_axis
    else:
        reference = (
            z_axis
            if abs(float(np.dot(normal, z_axis))) < 0.9
            else np.array([1.0, 0.0, 0.0])
        )
        u_axis = np.cross(normal, reference)
        u_axis = u_axis / (np.linalg.norm(u_axis) + 1e-12)
        v_axis = np.cross(normal, u_axis)
        v_axis = v_axis / (np.linalg.norm(v_axis) + 1e-12)
    return u_axis, v_axis


def _project_points(points, rotation, translation, camera_matrix):
    xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    camera = xyz @ rotation.T + translation
    if np.any(~np.isfinite(camera)) or np.any(camera[:, 2] <= 1e-6):
        raise ValueError('选中立面不在相机前方，无法从照片截取该区域')
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


def _facade_front_geometry(facade_points, plane_model, camera_center):
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
    u_axis, v_axis = _plane_axes(plane[:3], facade_type)
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


def _front_view_camera(normal, u_axis, v_axis, center, spans, output_size, margin):
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


def _crop_photo_around_facade(photo, photo_boundary, padding=16):
    """按立面投影外接框裁照片，四周各扩 ``padding`` 像素后丢掉其余区域。"""
    quad = np.asarray(photo_boundary, dtype=np.float64).reshape(4, 2)
    if not np.isfinite(quad).all():
        raise ValueError('立面在照片上的投影坐标无效')
    height, width = photo.shape[:2]
    pad = int(np.clip(padding, 10, 20))
    x0 = int(np.floor(quad[:, 0].min())) - pad
    y0 = int(np.floor(quad[:, 1].min())) - pad
    x1 = int(np.ceil(quad[:, 0].max())) + pad
    y1 = int(np.ceil(quad[:, 1].max())) + pad
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(width, x1)
    y1 = min(height, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        raise ValueError('立面在照片中的投影区域过小，无法截取')
    cropped = np.ascontiguousarray(photo[y0:y1, x0:x1])
    crop_boundary = quad - np.asarray((x0, y0), dtype=np.float64)
    return cropped, crop_boundary


def _apply_homography(points, matrix):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((len(pts), 1), dtype=np.float64)
    projected = np.hstack((pts, ones)) @ np.asarray(matrix, dtype=np.float64).T
    denom = projected[:, 2:3]
    if np.any(np.abs(denom) < 1e-12):
        raise ValueError('正视变换后的图像坐标无效')
    return projected[:, :2] / denom


def _homography_including_border(crop_boundary, destination, crop_size):
    """立面四角映射到正视矩形；裁切图四角随同一单应变换，输出画布包住边框。"""
    matrix = _homography(crop_boundary, destination, '照片')
    width, height = int(crop_size[0]), int(crop_size[1])
    crop_corners = np.asarray(
        (
            (0.0, 0.0),
            (width - 1.0, 0.0),
            (width - 1.0, height - 1.0),
            (0.0, height - 1.0),
        ),
        dtype=np.float64,
    )
    warped = np.vstack(
        (
            _apply_homography(crop_corners, matrix),
            np.asarray(destination, dtype=np.float64).reshape(4, 2),
        )
    )
    if not np.isfinite(warped).all():
        raise ValueError('正视变换后的边框坐标无效')
    x0 = float(np.floor(warped[:, 0].min()))
    y0 = float(np.floor(warped[:, 1].min()))
    x1 = float(np.ceil(warped[:, 0].max()))
    y1 = float(np.ceil(warped[:, 1].max()))
    shift = np.asarray(
        ((1.0, 0.0, -x0), (0.0, 1.0, -y0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    output_size = (
        max(2, int(x1 - x0) + 1),
        max(2, int(y1 - y0) + 1),
    )
    return shift @ matrix, output_size, (-x0, -y0)


def _rectify_photo_to_facade(
    photo_bgr,
    facade_points,
    plane_model,
    photo_rotation,
    photo_translation,
    photo_camera_matrix,
    *,
    distortion=None,
    target_max_dim=1600,
    margin_ratio=0.06,
    crop_padding=16,
):
    """按立面法向把裁出的立面拉成正视矩形，边框像素随同一单应变换。"""
    photo = _as_bgr_image(photo_bgr, '照片')
    photo_rotation = np.asarray(photo_rotation, dtype=np.float64).reshape(3, 3)
    photo_translation = np.asarray(photo_translation, dtype=np.float64).reshape(3)
    photo_camera = np.asarray(photo_camera_matrix, dtype=np.float64).reshape(3, 3)
    camera_center = -photo_rotation.T @ photo_translation

    plane, u_axis, v_axis, boundary, view_center, spans = _facade_front_geometry(
        facade_points,
        plane_model,
        camera_center,
    )

    maximum = int(np.clip(target_max_dim, 320, 4096))
    if spans[0] >= spans[1]:
        content_width = maximum
        content_height = max(64, int(round(maximum * spans[1] / spans[0])))
    else:
        content_height = maximum
        content_width = max(64, int(round(maximum * spans[0] / spans[1])))
    facade_size = (content_width, content_height)

    target_rotation, target_translation, target_camera = _front_view_camera(
        plane[:3],
        u_axis,
        v_axis,
        view_center,
        spans,
        facade_size,
        margin=0,
    )
    destination = _project_points(
        boundary,
        target_rotation,
        target_translation,
        target_camera,
    )

    coeffs = np.asarray(
        np.zeros(5) if distortion is None else distortion,
        dtype=np.float64,
    ).reshape(-1)
    if np.any(np.abs(coeffs) > 1e-12):
        photo = cv2.undistort(photo, photo_camera, coeffs, None, photo_camera)
    photo_boundary = _project_points(
        boundary,
        photo_rotation,
        photo_translation,
        photo_camera,
    )
    cropped, crop_boundary = _crop_photo_around_facade(
        photo,
        photo_boundary,
        padding=crop_padding,
    )
    photo_h, output_size, shift_xy = _homography_including_border(
        crop_boundary,
        destination,
        (cropped.shape[1], cropped.shape[0]),
    )
    target_camera = target_camera.copy()
    target_camera[0, 2] += float(shift_xy[0])
    target_camera[1, 2] += float(shift_xy[1])
    rectified = cv2.warpPerspective(
        cropped,
        photo_h,
        output_size,
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return {
        'photo_bgr': rectified,
        'camera_matrix': target_camera,
        'rotation_matrix': target_rotation,
        'translation_vector': target_translation,
        'photo_homography': photo_h,
        'boundary_points_3d': boundary,
        'output_size': output_size,
        'plane_model': plane,
    }


def _overlay_heatmap_on_rectified(
    rectified_bgr,
    grid_bgr,
    grid_mask,
    corners_3d,
    rotation,
    translation,
    camera_matrix,
    facade_boundary,
    *,
    alpha=0.72,
):
    """把热力网格透视贴到正视矩形图的对应位置。"""
    photo = _as_bgr_image(rectified_bgr, '正视照片')
    grid = _as_bgr_image(grid_bgr, '立面热力图')
    mask = np.asarray(grid_mask, dtype=np.uint8)
    if mask.shape[:2] != grid.shape[:2]:
        raise ValueError('热力图与有效像素掩膜尺寸不一致')

    height, width = grid.shape[:2]
    source = np.asarray(
        (
            (0.0, height - 1.0),
            (width - 1.0, height - 1.0),
            (width - 1.0, 0.0),
            (0.0, 0.0),
        ),
        dtype=np.float32,
    )
    if corners_3d is None:
        destination = _project_points(
            facade_boundary,
            rotation,
            translation,
            camera_matrix,
        ).astype(np.float32)
    else:
        corners = np.asarray(corners_3d, dtype=np.float64).reshape(4, 3)
        depths = (rotation @ corners.T + translation.reshape(3, 1)).T[:, 2]
        if np.any(~np.isfinite(depths)) or np.any(depths <= 1e-6):
            raise ValueError('热力图不在正视相机前方，无法贴到立面矩形图上')
        destination = _project_points(
            corners,
            rotation,
            translation,
            camera_matrix,
        ).astype(np.float32)
    if not np.isfinite(destination).all():
        raise ValueError('热力图投影坐标无效')

    matrix = cv2.getPerspectiveTransform(source, destination)
    photo_h, photo_w = photo.shape[:2]
    warped = cv2.warpPerspective(
        grid,
        matrix,
        (photo_w, photo_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    warped_mask = cv2.warpPerspective(
        mask,
        matrix,
        (photo_w, photo_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    weight = (
        warped_mask.astype(np.float32)[:, :, np.newaxis]
        / 255.0
        * float(np.clip(alpha, 0.0, 1.0))
    )
    blended = np.clip(
        photo.astype(np.float32) * (1.0 - weight)
        + warped.astype(np.float32) * weight,
        0,
        255,
    ).astype(np.uint8)
    return blended, warped


__all__ = [
    'CloudMappingInput',
    'CloudMappingResult',
    'PhotoMatchResult',
    'FacadeHeatmapOverlayResult',
    'PhotoFacadeHeatmapArgs',
    'generate_cloud_mapping_image',
    'generate_cloud_mapping_image_from_input',
    'estimate_photo_cloud_match',
    'overlay_facade_heatmap_on_photo',
    'run_photo_facade_heatmap',
]
