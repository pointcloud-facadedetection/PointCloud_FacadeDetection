"""从原始点云按当前相机导出多模态视图。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def render_original_view(
    points,
    colors,
    camera_matrix,
    extrinsic,
    image_size,
    *,
    chunk_size=1_000_000,
):
    """分块投影原始点云，返回 BGR 彩色图和相机 Z 深度图。"""
    width, height = map(int, image_size)
    intrinsic = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    pose = np.asarray(extrinsic, dtype=np.float64).reshape(4, 4)
    rotation, translation = pose[:3, :3], pose[:3, 3]
    pts = np.asarray(points)
    cols = None if colors is None else np.asarray(colors)
    if cols is not None and len(cols) != len(pts):
        cols = None

    image = np.zeros((height, width, 3), dtype=np.uint8)
    z_buffer = np.full(height * width, np.inf, dtype=np.float64)
    image_flat = image.reshape(-1, 3)

    for start in range(0, len(pts), int(chunk_size)):
        stop = min(len(pts), start + int(chunk_size))
        world = np.asarray(pts[start:stop], dtype=np.float64)
        camera = world @ rotation.T + translation
        z = camera[:, 2]
        valid = np.isfinite(camera).all(axis=1) & (z > 1e-6)
        if not np.any(valid):
            continue
        camera, z = camera[valid], z[valid]
        source_ids = np.flatnonzero(valid)
        x = np.rint(
            intrinsic[0, 0] * camera[:, 0] / z + intrinsic[0, 2]
        ).astype(np.int32)
        y = np.rint(
            intrinsic[1, 1] * camera[:, 1] / z + intrinsic[1, 2]
        ).astype(np.int32)
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if not np.any(inside):
            continue
        x, y, z = x[inside], y[inside], z[inside]
        source_ids = source_ids[inside]
        flat = y.astype(np.int64) * width + x

        # 每块内先保留每个像素最近点，再与全局 Z-buffer 比较。
        order = np.lexsort((z, flat))
        sorted_flat = flat[order]
        first = np.empty(len(order), dtype=bool)
        first[0] = True
        first[1:] = sorted_flat[1:] != sorted_flat[:-1]
        selected = order[first]
        pixels = flat[selected]
        depths = z[selected]
        nearer = depths < z_buffer[pixels]
        if not np.any(nearer):
            continue
        pixels = pixels[nearer]
        z_buffer[pixels] = depths[nearer]
        chosen = source_ids[selected[nearer]]
        if cols is None:
            image_flat[pixels] = 220
        else:
            rgb = cols[start:stop][chosen]
            if np.issubdtype(rgb.dtype, np.floating):
                rgb = np.clip(rgb * 255.0, 0, 255)
            image_flat[pixels] = np.asarray(rgb, dtype=np.uint8)[:, ::-1]

    depth = z_buffer.reshape(height, width)
    depth[~np.isfinite(depth)] = 0.0
    return image, depth


def build_export_images(raw_bgr, depth, pinhole_bgr):
    """由原始彩色和深度图生成平滑、深度及深度边缘图。"""
    raw = np.asarray(raw_bgr, dtype=np.uint8)
    z = np.asarray(depth, dtype=np.float64)
    valid = np.isfinite(z) & (z > 0)

    mask = valid.astype(np.uint8) * 255
    closed = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2
    )
    holes = ((closed > 0) & ~valid).astype(np.uint8) * 255
    filled = cv2.inpaint(raw, holes, 3.0, cv2.INPAINT_NS)
    smooth = cv2.bilateralFilter(filled, 9, 55, 55)
    smooth[closed == 0] = 0

    depth_gray = np.zeros(z.shape, dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(z[valid], (1.0, 99.0))
        if high <= low:
            high = low + 1.0
        normalized = 1.0 - np.clip((z - low) / (high - low), 0.0, 1.0)
        depth_gray[valid] = np.rint(normalized[valid] * 255.0).astype(np.uint8)
    depth_edge = cv2.Canny(depth_gray, 35, 110)
    depth_edge = cv2.dilate(
        depth_edge, np.ones((2, 2), np.uint8), iterations=1
    )
    return {
        'raw': raw,
        'smooth': smooth,
        'depth': depth_gray,
        'depth-edge': depth_edge,
        'pinhole': np.asarray(pinhole_bgr, dtype=np.uint8),
    }


def save_export_images(images, output_dir, stem='pointcloud_view'):
    """以支持 Windows Unicode 路径的方式保存 PNG 图像。"""
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, image in images.items():
        path = folder / f'{stem}_{name}.png'
        ok, encoded = cv2.imencode('.png', np.asarray(image))
        if not ok:
            raise OSError(f'无法编码图片：{path.name}')
        encoded.tofile(str(path))
        paths[name] = str(path)
    return paths


__all__ = [
    'render_original_view',
    'build_export_images',
    'save_export_images',
]
