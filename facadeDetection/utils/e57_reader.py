"""E57 point-cloud reader with a stable ``(points, colors)`` contract.

E57 is not consistently supported by the Open3D wheels used on Windows, so
the optional pye57 reader is preferred.  Keeping this adapter at the file
boundary lets the existing proxy, denoise, detection and report pipelines
remain format agnostic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def _field(raw, name: str):
    """Read a structured-array field while tolerating pye57 field aliases."""
    if isinstance(raw, dict):
        for candidate in (name, name.lower(), name.upper()):
            if candidate in raw:
                return np.asarray(raw[candidate])
        return None
    names = getattr(getattr(raw, "dtype", None), "names", None) or ()
    for candidate in (name, name.lower(), name.upper()):
        if candidate in names:
            return np.asarray(raw[candidate])
    return None


def _normalise_colour(values, count: int) -> Optional[np.ndarray]:
    if values is None:
        return None
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] != count:
        return None
    arr = arr.astype(np.float32, copy=False)
    if np.nanmax(arr) > 1.0:
        arr /= 255.0
    return np.ascontiguousarray(np.clip(arr, 0.0, 1.0), dtype=np.float32)


def read_e57(path: str):
    """Return finite E57 points and optional RGB colors as float32 arrays."""
    source = Path(path).expanduser().resolve()
    try:
        import pye57
    except ImportError as exc:
        raise RuntimeError(
            '读取 E57 需要安装 pye57；请执行 pip install pye57') from exc

    try:
        reader = pye57.E57(str(source))
        chunks = []
        color_chunks = []
        have_color = True
        scan_count = int(getattr(reader, 'scan_count', 0))
        if not scan_count and hasattr(reader, 'root'):
            scan_count = int(getattr(reader.root, 'header_count', 0))
        for scan_index in range(scan_count):
            # read_scan applies the scan pose when transform=True and exposes
            # RGB fields through the same dictionary contract.  This keeps
            # E57 coordinates in the global frame used by existing operators.
            raw = reader.read_scan(
                scan_index, colors=True, transform=True,
                ignore_missing_fields=True)
            x = _field(raw, 'cartesianX')
            y = _field(raw, 'cartesianY')
            z = _field(raw, 'cartesianZ')
            if x is None or y is None or z is None:
                continue
            points = np.column_stack((x, y, z)).astype(np.float32, copy=False)
            finite = np.isfinite(points).all(axis=1)
            chunks.append(points[finite])
            red = _field(raw, 'colorRed')
            green = _field(raw, 'colorGreen')
            blue = _field(raw, 'colorBlue')
            if red is None or green is None or blue is None:
                have_color = False
            elif have_color:
                colors = np.column_stack((red, green, blue))[finite]
                color_chunks.append(colors)
        if not chunks:
            raise ValueError('E57 中没有可读取的笛卡尔坐标点')
        points = np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)
        colors = (_normalise_colour(np.concatenate(color_chunks), len(points))
                  if have_color and color_chunks and
                  sum(len(chunk) for chunk in color_chunks) == len(points)
                  else None)
        return points, colors
    except Exception:
        # Do not silently reinterpret malformed E57 data as an empty cloud.
        raise