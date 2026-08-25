"""将 2D 照片（含热力图叠加）重采样到立面正射视角。"""

from __future__ import annotations

import base64

import cv2
import numpy as np

from .facade_render import _frame_resolution
from .heatmap_overplay import _encode_png_b64, create_facade_heatmap_overlay


def _default_distortion(dist_coeffs=None):
    if dist_coeffs is None:
        return np.zeros((5, 1), dtype=np.float64)
    dist = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
    if dist.shape[0] < 5:
        padded = np.zeros((5, 1), dtype=np.float64)
        padded[: dist.shape[0]] = dist
        return padded
    return dist[:5]


def _world_grid_from_frame(frame):
    """正射图像素网格 → 3D 世界坐标 (H,W,3)。"""
    h = int(frame['height_px'])
    w = int(frame['width_px'])
    px = np.arange(w, dtype=np.float64)
    py = np.arange(h, dtype=np.float64)
    px_grid, py_grid = np.meshgrid(px, py)

    res = _frame_resolution(frame)
    origin = np.asarray(frame['origin'], dtype=np.float64)
    u_axis = np.asarray(frame['u_axis'], dtype=np.float64)
    v_axis = np.asarray(frame['v_axis'], dtype=np.float64)

    u = float(frame['u_min']) + px_grid * res
    v = float(frame['v_max']) - py_grid * res
    xyz = (
        origin.reshape(1, 1, 3)
        + u[:, :, np.newaxis] * u_axis.reshape(1, 1, 3)
        + v[:, :, np.newaxis] * v_axis.reshape(1, 1, 3)
    )
    return xyz.astype(np.float32)


def rectify_image_to_ortho_view(source_bgr, mapping_frame, pose, interpolation=cv2.INTER_LINEAR):
    """以正射图网格为输出，将 source 重采样到立面正射视角。"""
    source = np.asarray(source_bgr)
    if source.ndim == 2:
        source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    src_h, src_w = source.shape[:2]

    frame = mapping_frame
    out_h = int(frame['height_px'])
    out_w = int(frame['width_px'])
    if out_h <= 0 or out_w <= 0:
        raise ValueError('mapping.frame 宽高无效')

    R = np.asarray(pose['rotation_matrix'], dtype=np.float64).reshape(3, 3)
    T = np.asarray(pose['translation_vector'], dtype=np.float64).reshape(3, 1)
    K = np.asarray(pose['camera_matrix'], dtype=np.float64).reshape(3, 3)
    dist = _default_distortion(pose.get('distortion_coefficients'))
    rvec, _ = cv2.Rodrigues(R)

    world_grid = _world_grid_from_frame(frame)
    pts = world_grid.reshape(-1, 1, 3)
    projected, _ = cv2.projectPoints(pts, rvec, T, K, dist)
    projected = projected.reshape(out_h, out_w, 2)

    map_x = projected[:, :, 0].astype(np.float32)
    map_y = projected[:, :, 1].astype(np.float32)

    cam_pts = (R @ world_grid.reshape(-1, 3).T + T).T
    depth = cam_pts[:, 2].reshape(out_h, out_w)
    valid = (
        (depth > 1e-6)
        & (map_x >= 0)
        & (map_x < src_w - 1)
        & (map_y >= 0)
        & (map_y < src_h - 1)
    )

    rectified = cv2.remap(
        source,
        map_x,
        map_y,
        interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    rectified[~valid] = 0

    meta = {
        'width_px': out_w,
        'height_px': out_h,
        'valid_pixel_count': int(np.sum(valid)),
        'total_pixel_count': int(out_h * out_w),
        'valid_pixel_ratio': float(np.mean(valid)),
        'resolution_m': float(_frame_resolution(frame)),
    }
    return rectified, valid.astype(np.uint8), meta


def _compose_comparison(ortho_bgr, rectified_bgr, gap=8):
    h = max(ortho_bgr.shape[0], rectified_bgr.shape[0])

    def _pad(img):
        if img.shape[0] == h:
            return img
        scale = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), h), interpolation=cv2.INTER_AREA)

    left = _pad(ortho_bgr)
    right = _pad(rectified_bgr)
    sep = np.full((h, gap, 3), 32, dtype=np.uint8)
    combo = np.hstack([left, sep, right])
    cv2.putText(combo, 'Orthographic ref', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(
        combo,
        'Rectified photo view',
        (left.shape[1] + gap + 10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return combo


def create_ortho_rectified_view(
    photo_path,
    facade,
    points,
    pose,
    ortho_result,
    use_heatmap_overlay=True,
    overlay_alpha=0.5,
):
    """将照片（可选含热力图）调正到立面正射视角。"""
    if ortho_result is None or not ortho_result.get('mapping'):
        raise ValueError('缺少立面正射图 mapping，请先生成正射图')

    frame = ortho_result['mapping']['frame']

    if use_heatmap_overlay:
        overlay = create_facade_heatmap_overlay(
            photo_path, facade, points, pose, alpha=overlay_alpha,
        )
        src_bytes = base64.b64decode(overlay['image_base64'])
        source = cv2.imdecode(np.frombuffer(src_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if source is None:
            raise ValueError('热力图叠加图解码失败')
    else:
        source = cv2.imread(str(photo_path))
        if source is None:
            raise FileNotFoundError(f'无法读取照片: {photo_path}')

    rectified, valid_mask, meta = rectify_image_to_ortho_view(source, frame, pose)

    ortho_bytes = base64.b64decode(ortho_result['image_base64'])
    ortho_bgr = cv2.imdecode(np.frombuffer(ortho_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if ortho_bgr is None:
        raise ValueError('正射参考图解码失败')
    comparison = _compose_comparison(ortho_bgr, rectified)

    return {
        'facade_id': int(facade.get('id', ortho_result.get('facade_id', -1))),
        'rectified_image_base64': _encode_png_b64(rectified),
        'comparison_image_base64': _encode_png_b64(comparison),
        'valid_mask_base64': _encode_png_b64(valid_mask),
        'image_mime': 'image/png',
        'width_px': meta['width_px'],
        'height_px': meta['height_px'],
        'resolution_m': meta['resolution_m'],
        'rectify_meta': meta,
        'used_heatmap_overlay': bool(use_heatmap_overlay),
    }


__all__ = ['rectify_image_to_ortho_view', 'create_ortho_rectified_view']
