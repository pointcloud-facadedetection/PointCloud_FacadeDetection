"""当前点云视图与照片的自动 2D-3D 匹配。"""

from __future__ import annotations

import cv2
import numpy as np

from .manual_match import MIN_MATCH_PAIRS, estimate_match_matrix

_MAX_PROCESSING_SIDE = 1600
_MAX_KEYPOINTS = 4096
_MATCH_CONFIDENCE = 0.10
_LIGHTGLUE_ENGINE = None


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
    """延迟加载 SuperPoint + LightGlue，并复用模型实例。"""
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = SuperPoint(max_num_keypoints=_MAX_KEYPOINTS).eval().to(device)
    matcher = (
        LightGlue(features='superpoint', filter_threshold=0.0)
        .eval()
        .to(device)
    )
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

    # 按像素、深度排序，每个像素保留离相机最近的点。
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


def match_photo_to_cloud_view(
    photo_bgr,
    view_bgr,
    depth_image,
    view_camera_matrix,
    view_extrinsic,
    cloud_points=None,
    pixel_point_index=None,
) -> dict:
    """通过当前点云渲染视图生成照片像素与世界三维点对应关系。"""
    photo = np.asarray(photo_bgr, dtype=np.uint8)
    view = np.asarray(view_bgr, dtype=np.uint8)
    depth = np.asarray(depth_image, dtype=np.float64)
    intrinsic = np.asarray(view_camera_matrix, dtype=np.float64).reshape(3, 3)
    extrinsic = np.asarray(view_extrinsic, dtype=np.float64).reshape(4, 4)
    if photo.ndim != 3 or view.ndim != 3 or depth.ndim != 2:
        raise ValueError('自动匹配需要照片、点云彩色截图和深度图')
    if view.shape[:2] != depth.shape:
        raise ValueError('点云截图与深度图尺寸不一致')

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
        raise ValueError('像素点索引图与点云截图尺寸不一致')
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
        pose = estimate_match_matrix(
            object_points=object_points,
            image_points=image_points,
            image_width=photo.shape[1],
            image_height=photo.shape[0],
        )
        result.update(pose)
        result['pose_estimated'] = True
    return result


__all__ = ['match_photo_to_cloud_view']
