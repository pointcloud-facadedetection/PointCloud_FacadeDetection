"""扫描位姿渲染 + 照片摆正 + 按 2D-3D 匹配矩阵贴到点云视图。"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .photo_rectifier import PhotoPerspectiveRectifier


def find_scan_pose_json(ply_path=None, search_dir=None) -> Path:
    """在点云同目录查找记录扫描位姿的 JSON。"""
    candidates = []
    if ply_path:
        ply = Path(ply_path)
        candidates.append(ply.with_suffix('.json'))
        if ply.parent.is_dir():
            search_dir = search_dir or ply.parent
    if search_dir:
        folder = Path(search_dir)
        if folder.is_dir():
            candidates.extend(sorted(folder.glob('*.json')))
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        if _has_scan_pose_fields(data):
            return path
    raise FileNotFoundError(
        '未找到扫描位姿 JSON。请将含 transformToGlobal / scan_origin 的文件'
        '放在点云上传目录（建议与 .ply 同名）。'
    )


def _has_scan_pose_fields(data) -> bool:
    if not isinstance(data, dict):
        return False
    keys = (
        'transformToGlobal', 'transform_to_global',
        'scan_origin', 'scanOrigin', 'scannerPosition',
        'camera_matrix', 'rotation_matrix',
    )
    if any(key in data for key in keys):
        return True
    scans = data.get('scans')
    return isinstance(scans, list) and any(
        isinstance(item, dict) and _has_scan_pose_fields(item) for item in scans
    )


def _first_pose_dict(data: dict) -> dict:
    if _has_scan_pose_fields(data) and 'scans' not in data:
        return data
    for item in data.get('scans') or []:
        if isinstance(item, dict) and _has_scan_pose_fields(item):
            return item
    return data


def load_scan_camera(json_path, lookat, image_size=(960, 720), horizontal_fov_deg=70.0):
    """从 JSON 解析测站相机，使光轴指向 lookat（通常是选中立面中心）。"""
    data = json.loads(Path(json_path).read_text(encoding='utf-8'))
    pose_src = _first_pose_dict(data)
    width, height = int(image_size[0]), int(image_size[1])
    lookat = np.asarray(lookat, dtype=np.float64).reshape(3)

    if pose_src.get('camera_matrix') is not None and pose_src.get('rotation_matrix') is not None:
        K = np.asarray(pose_src['camera_matrix'], dtype=np.float64).reshape(3, 3)
        R = np.asarray(pose_src['rotation_matrix'], dtype=np.float64).reshape(3, 3)
        t = np.asarray(
            pose_src.get('translation_vector') or pose_src.get('tvec') or [0, 0, 0],
            dtype=np.float64,
        ).reshape(3)
        return {
            'K': K, 'R': R, 't': t,
            'width': width, 'height': height,
            'origin': (-R.T @ t).tolist(),
            'json_path': str(json_path),
        }

    origin = pose_src.get('scan_origin') or pose_src.get('scanOrigin') or pose_src.get('scannerPosition')
    transform = pose_src.get('transformToGlobal') or pose_src.get('transform_to_global')
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if transform is not None:
        matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
        origin = matrix[:3, 3]
        axis_up = matrix[:3, 2]
        if np.linalg.norm(axis_up) > 1e-6:
            world_up = axis_up / (np.linalg.norm(axis_up) + 1e-12)
    if origin is None:
        raise ValueError('JSON 中缺少 transformToGlobal 或 scan_origin')
    origin = np.asarray(origin, dtype=np.float64).reshape(3)

    forward = lookat - origin
    if np.linalg.norm(forward) < 1e-3:
        forward = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    if abs(float(np.dot(forward, world_up))) > 0.97:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis = np.cross(forward, world_up)
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-12)
    y_down = np.cross(forward, x_axis)
    y_down = y_down / (np.linalg.norm(y_down) + 1e-12)
    R = np.vstack([x_axis, y_down, forward])
    t = -R @ origin

    fov = np.deg2rad(float(horizontal_fov_deg))
    fx = width / (2.0 * np.tan(fov / 2.0))
    K = np.array(
        [[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return {
        'K': K, 'R': R, 't': t,
        'width': width, 'height': height,
        'origin': origin.tolist(),
        'json_path': str(json_path),
    }


def render_points_to_image(points, colors, camera) -> np.ndarray:
    """按测站相机把点云光栅化为 BGR 图。"""
    width, height = int(camera['width']), int(camera['height'])
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if colors is None:
        cols = np.full((len(pts), 3), 200, dtype=np.uint8)
    else:
        cols = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
        if cols.max() <= 1.0 + 1e-6:
            cols = np.clip(cols * 255.0, 0, 255)
        cols = cols.astype(np.uint8)
        if len(cols) != len(pts):
            cols = np.full((len(pts), 3), 200, dtype=np.uint8)

    cam = (camera['R'] @ pts.T).T + camera['t'].reshape(1, 3)
    z = cam[:, 2]
    uvw = (camera['K'] @ cam.T).T
    valid = z > 1e-4
    u = np.full(len(pts), -1.0)
    v = np.full(len(pts), -1.0)
    u[valid] = uvw[valid, 0] / z[valid]
    v[valid] = uvw[valid, 1] / z[valid]
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    inside = valid & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if not np.any(inside):
        return image
    order = np.argsort(-z[inside])
    ui_i = ui[inside][order]
    vi_i = vi[inside][order]
    rgb = cols[inside][order]
    bgr = rgb[:, ::-1]
    image[vi_i, ui_i] = bgr
    return image


def project_points(points, camera):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cam = (camera['R'] @ pts.T).T + camera['t'].reshape(1, 3)
    z = cam[:, 2]
    uvw = (camera['K'] @ cam.T).T
    uv = np.column_stack([uvw[:, 0] / np.maximum(z, 1e-9), uvw[:, 1] / np.maximum(z, 1e-9)])
    return uv, z


def _apply_homography(points_xy, matrix):
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((len(pts), 1), dtype=np.float64)
    mapped = (np.asarray(matrix, dtype=np.float64).reshape(3, 3) @ np.hstack([pts, ones]).T).T
    mapped = mapped[:, :2] / np.maximum(mapped[:, 2:3], 1e-9)
    return mapped


def map_photo_onto_scan_view(
    photo_path,
    points,
    colors,
    correspondences,
    facade,
    ply_path=None,
    search_dir=None,
    overlay_alpha=0.55,
    photo_points_are_rectified=False,
):
    """
    1. 用上传目录 JSON 中的扫描位姿渲染点云 2D 图
    2. 摆正现场照片（消除近大远小）
    3. 用 2D-3D 匹配点估计单应，把选中立面像素贴到点云图上
    """
    if not correspondences:
        raise ValueError('请先完成 2D-3D 匹配')
    json_path = find_scan_pose_json(ply_path=ply_path, search_dir=search_dir)
    inliers = np.asarray(facade.get('inlier_indices') or [], dtype=int)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if inliers.size:
        inliers = inliers[(inliers >= 0) & (inliers < len(pts))]
        lookat = np.mean(pts[inliers], axis=0) if inliers.size else np.mean(pts, axis=0)
    else:
        lookat = np.asarray(facade.get('center') or np.mean(pts, axis=0), dtype=np.float64)

    camera = load_scan_camera(json_path, lookat)
    scan_bgr = render_points_to_image(pts, colors, camera)

    photo = cv2.imread(str(photo_path), cv2.IMREAD_COLOR)
    if photo is None:
        raise FileNotFoundError(f'无法读取照片: {photo_path}')
    rectifier = PhotoPerspectiveRectifier(target_max_dim=1280)
    rect_bgr, h_rect, _h_inv = rectifier.rectify(photo)

    photo_xy = []
    cloud_xyz = []
    for pair in correspondences:
        photo = getattr(pair, 'photo', None)
        cloud = getattr(pair, 'cloud', None)
        if photo is None and isinstance(pair, dict):
            photo = pair.get('photo') or pair.get('image_point')
            cloud = pair.get('cloud') or pair.get('object_point')
        if photo is None or cloud is None:
            continue
        photo_xy.append(photo)
        cloud_xyz.append(cloud)
    photo_xy = np.asarray(photo_xy, dtype=np.float64).reshape(-1, 2)
    cloud_xyz = np.asarray(cloud_xyz, dtype=np.float64).reshape(-1, 3)
    if len(photo_xy) < 4:
        raise ValueError('匹配点不足 4 对，无法估计映射单应矩阵')

    if photo_points_are_rectified:
        rect_xy = photo_xy
    else:
        rect_xy = _apply_homography(photo_xy, h_rect)
    scan_xy, depth = project_points(cloud_xyz, camera)
    keep = (depth > 1e-4) & np.isfinite(scan_xy).all(axis=1) & np.isfinite(rect_xy).all(axis=1)
    if int(np.sum(keep)) < 4:
        raise ValueError('有效匹配点不足以把照片映射到扫描视图')
    homography, inlier_mask = cv2.findHomography(
        rect_xy[keep].astype(np.float32),
        scan_xy[keep].astype(np.float32),
        cv2.RANSAC,
        8.0,
    )
    if homography is None:
        raise ValueError('无法从 2D-3D 匹配点估计映射矩阵')

    warped = cv2.warpPerspective(
        rect_bgr, homography, (camera['width'], camera['height']),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
    )
    overlay = scan_bgr.copy()
    photo_mask = np.any(warped > 12, axis=2)
    if inliers.size:
        facade_uv, facade_z = project_points(pts[inliers], camera)
        fx = np.round(facade_uv[:, 0]).astype(np.int32)
        fy = np.round(facade_uv[:, 1]).astype(np.int32)
        facade_mask = np.zeros((camera['height'], camera['width']), dtype=bool)
        inside = (
            (facade_z > 1e-4)
            & (fx >= 0) & (fx < camera['width'])
            & (fy >= 0) & (fy < camera['height'])
        )
        facade_mask[fy[inside], fx[inside]] = True
        kernel = np.ones((7, 7), np.uint8)
        facade_mask = cv2.dilate(facade_mask.astype(np.uint8), kernel, iterations=2).astype(bool)
        photo_mask &= facade_mask
    alpha = float(overlay_alpha)
    overlay[photo_mask] = (
        alpha * warped[photo_mask] + (1.0 - alpha) * scan_bgr[photo_mask]
    ).astype(np.uint8)

    return {
        'rectified_bgr': rect_bgr,
        'scan_bgr': scan_bgr,
        'overlay_bgr': overlay,
        'homography': homography.tolist(),
        'rectify_method': rectifier.last_method,
        'json_path': str(json_path),
        'inlier_count': int(np.sum(inlier_mask)) if inlier_mask is not None else 0,
        'camera_origin': camera['origin'],
    }


__all__ = [
    'find_scan_pose_json',
    'load_scan_camera',
    'map_photo_onto_scan_view',
]
