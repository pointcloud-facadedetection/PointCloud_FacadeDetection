"""Small, format-neutral PLY writer used by import adapters."""
from __future__ import annotations

from pathlib import Path
import os
import uuid
import numpy as np


def write_point_cloud_ply_atomic(path, points, colors=None):
    """写入统一 Runtime PLY：总是包含 xyz + uchar rgb。

    参数
    ----
    points : (N, 3) float32
        全局坐标；必须已过滤 NaN/Inf。
    colors : (N, 3) float32 或 None
        值域 [0,1] 或 [0,255]；None 时填充中性灰 (128,128,128)。

    返回
    ----
    dict
        包含 path、point_count、has_colors、bounds_min、bounds_max。
    """
    path = Path(path)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("PLY 点云坐标为空或形状不是 Nx3")
    if not np.isfinite(points).all():
        raise ValueError("PLY 点云坐标包含 NaN/Inf")

    n = len(points)

    # 颜色：无颜色时补齐中性灰；兼容 [0,1] float 与 [0,255]
    colors_arr = np.asarray(colors, dtype=np.float32) if colors is not None else None
    if colors_arr is not None and colors_arr.shape != (n, 3):
        colors_arr = None

    if colors_arr is None:
        color_data = np.full((n, 3), 128, dtype=np.uint8)
        has_colors = False
    else:
        if colors_arr.max() > 1.0 + 1e-6:
            colors_arr = colors_arr / 255.0
        color_data = np.clip(colors_arr * 255.0, 0, 255).astype(np.uint8)
        has_colors = True

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment PCFD runtime_ply_version 1",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    header = "\n".join(header_lines).encode("ascii") + b"\n"

    # 构建二进制记录：确保与 read_ply_fast 的解析规则对齐
    records = np.empty(n, dtype=[
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ])
    records['x'] = points[:, 0]
    records['y'] = points[:, 1]
    records['z'] = points[:, 2]
    records['red'] = color_data[:, 0]
    records['green'] = color_data[:, 1]
    records['blue'] = color_data[:, 2]

    binary_data = records.tobytes()

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.ply")
    try:
        with open(tmp, 'wb') as f:
            f.write(header)
            f.write(binary_data)
        if not tmp.exists() or tmp.stat().st_size == 0:
            raise IOError(f"PLY 写入结果为空: {path}")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)

    return {
        'path': str(path),
        'point_count': int(n),
        'has_colors': bool(has_colors),
        'bounds_min': points.min(axis=0).tolist(),
        'bounds_max': points.max(axis=0).tolist(),
    }
