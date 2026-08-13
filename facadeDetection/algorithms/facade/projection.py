"""Facade orthographic projection and sparse defect heatmap rasterization."""
from __future__ import annotations
import numpy as np
from algorithms.geometry import plane_axes


def rasterize_facade(points, colors, plane_model, signed_gap, flatness_limit,
                     pixel_size=0.01, max_size=6000, defect_values=None,
                     defect_colors=None, vmin=None, vmax=None):
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    gaps = np.asarray(signed_gap, dtype=float).reshape(-1)
    rgb = np.asarray(colors if colors is not None else np.full((len(pts), 3), .75), dtype=float)
    if len(pts) == 0 or len(gaps) != len(pts):
        raise ValueError('points and signed_gap must have equal non-zero length')
    n = np.asarray(plane_model[:3], dtype=float); n /= np.linalg.norm(n) + 1e-12
    u, v = plane_axes(n, 'vertical_facade')
    origin = np.mean(pts, axis=0)
    uv = np.column_stack(((pts-origin) @ u, (pts-origin) @ v))
    lo = uv.min(axis=0); hi = uv.max(axis=0)
    size = max(float(pixel_size), 1e-4)
    shape = np.maximum(np.ceil((hi-lo)/size).astype(int)+1, 1)
    if max(shape) > max_size:
        size *= max(shape) / float(max_size)
        shape = np.maximum(np.ceil((hi-lo)/size).astype(int)+1, 1)
    h, w = int(shape[1]), int(shape[0])
    x = np.clip(((uv[:, 0]-lo[0])/size).astype(int), 0, w-1)
    y = np.clip((h-1-(uv[:, 1]-lo[1])/size).astype(int), 0, h-1)
    flat = y*w+x
    count = np.bincount(flat, minlength=h*w).reshape(h,w).astype(np.int32)
    base = np.zeros((h*w,3), dtype=np.float32); np.add.at(base, flat, rgb.astype(np.float32)); base /= np.maximum(count.reshape(-1,1),1); base=base.reshape(h,w,3)
    # Prefer the values/colors produced by quality.py.  The old signed-gap
    # calculation remains a compatibility fallback for old callers.
    if defect_values is None:
        abs_gap = np.abs(gaps)
        defect = np.isfinite(abs_gap) & (abs_gap > float(flatness_limit))
        values = abs_gap[defect]
        vmax = max(float(np.percentile(values, 98)) if values.size else float(flatness_limit) * 1.05,
                   float(flatness_limit) * 1.05)
        t = np.clip((values - float(flatness_limit)) / (vmax - float(flatness_limit) + 1e-12), 0, 1)
        heat = np.column_stack((np.clip(1.5*t, 0, 1),
                                np.clip(1.5-np.abs(2*t-1)*1.5, 0, 1),
                                np.clip(1.2*(1-t), 0, 1))).astype(np.float32)
        defect_indices = np.flatnonzero(defect)
    else:
        values = np.asarray(defect_values, dtype=float).reshape(-1)
        defect_indices = np.flatnonzero(np.isfinite(values))
        values = values[defect_indices]
        heat = np.asarray(defect_colors, dtype=float).reshape(-1, 3)[defect_indices] if defect_colors is not None else np.zeros((len(values), 3))
        vmax = float(vmax if vmax is not None else (np.max(values) if len(values) else flatness_limit))
    vmin = float(flatness_limit if vmin is None else vmin)
    # Keep the largest defect in each pixel; last-write-wins made dense clouds
    # visually and numerically dependent on point ordering.
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    # Compare magnitudes, while retaining the signed value for diagnostics.
    # This is important for recessed (negative) points: comparing raw signed
    # values would make the least-negative point win inside a pixel.
    pixel_values = np.full(h*w, -np.inf, dtype=float)
    pixel_ids = flat[defect_indices]
    for p, value, colour in zip(pixel_ids, values, heat):
        if abs(value) > pixel_values[p]:
            pixel_values[p] = abs(value)
            overlay.reshape(-1, 4)[p, :3] = np.clip(colour, 0, 1) * 255
            overlay.reshape(-1, 4)[p, 3] = 255
    return {'base_rgb':(np.clip(base,0,1)*255).astype(np.uint8), 'overlay_rgba':overlay,
            'u_axis':u, 'v_axis':v, 'origin':origin, 'bounds':np.r_[lo,hi],
            'pixel_size':size, 'vmin':vmin, 'vmax':vmax, 'count':count,
            'defect_mask':np.isfinite(gaps) & (np.abs(gaps) > float(flatness_limit)), 'uv':uv,
            'pixel_defect_values':pixel_values.reshape(h, w),
            'polarity':'signed: negative=recessed, positive=protruding'}