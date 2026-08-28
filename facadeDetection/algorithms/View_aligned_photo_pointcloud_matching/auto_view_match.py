"""当前点云视图与照片的自动 2D-3D 匹配。"""

from __future__ import annotations

import cv2
import numpy as np

from .manual_match import MIN_MATCH_PAIRS, estimate_match_matrix


def _prepare_gray(image, max_side: int = 2200):
    array = np.asarray(image)
    if array.ndim == 3:
        gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    else:
        gray = array.astype(np.uint8, copy=False)
    height, width = gray.shape[:2]
    scale = min(1.0, float(max_side) / max(height, width))
    if scale < 1.0:
        gray = cv2.resize(
            gray,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(enhanced, 45, 140)
    edges = cv2.dilate(edges, np.ones((2, 2), dtype=np.uint8), iterations=1)
    matched_image = cv2.addWeighted(enhanced, 0.65, edges, 0.35, 0.0)
    return matched_image, scale


def _feature_matches(photo_bgr, view_bgr):
    photo_gray, photo_scale = _prepare_gray(photo_bgr)
    view_gray, view_scale = _prepare_gray(view_bgr)
    if not hasattr(cv2, 'SIFT_create'):
        raise ValueError('当前 OpenCV 不支持 SIFT，无法执行自动匹配')

    detector = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.02)
    photo_keys, photo_desc = detector.detectAndCompute(photo_gray, None)
    view_keys, view_desc = detector.detectAndCompute(view_gray, None)
    if photo_desc is None or view_desc is None:
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, empty.copy(), 0

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(view_desc, photo_desc, k=2)
    reverse = matcher.knnMatch(photo_desc, view_desc, k=2)
    reverse_best = {
        pair[0].queryIdx: pair[0].trainIdx
        for pair in reverse
        if len(pair) == 2 and pair[0].distance < 0.78 * pair[1].distance
    }
    good = [
        pair[0]
        for pair in forward
        if (
            len(pair) == 2
            and pair[0].distance < 0.78 * pair[1].distance
            and reverse_best.get(pair[0].trainIdx) == pair[0].queryIdx
        )
    ]
    view_xy = np.asarray(
        [view_keys[item.queryIdx].pt for item in good], dtype=np.float64
    ).reshape(-1, 2) / view_scale
    photo_xy = np.asarray(
        [photo_keys[item.trainIdx].pt for item in good], dtype=np.float64
    ).reshape(-1, 2) / photo_scale

    if len(good) >= 4:
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
            return view_xy[keep], photo_xy[keep], len(good)
    return view_xy, photo_xy, len(good)


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

    view_points, photo_points, ratio_match_count = _feature_matches(photo, view)
    lines = [
        f'[AutoMatch][2D-2D] 几何一致匹配共 {len(view_points)} 对：'
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
    point_index_map = None
    if cloud_points is not None:
        cloud = np.asarray(cloud_points)
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
