from __future__ import annotations
import numpy as np
from algorithms.geometry import plane_axes


def rasterize_facade(points, colors, plane_model, defect_values, defect_limit,
                     pixel_size=0.01, max_size=6000, defect_colors=None,
                     vmin=None, vmax=None):
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    gaps = np.asarray(defect_values, dtype=float).reshape(-1)
    rgb = np.asarray(colors if colors is not None else np.full((len(pts), 3), .75), dtype=float)
    if len(pts) == 0 or len(gaps) != len(pts):
        raise ValueError('points and defect_values must have equal non-zero length')
    
    n = np.asarray(plane_model[:3], dtype=float)
    n /= np.linalg.norm(n) + 1e-12
    u, v = plane_axes(n, 'vertical_facade')
    origin = np.mean(pts, axis=0)
    uv = np.column_stack(((pts-origin) @ u, (pts-origin) @ v))
    lo = uv.min(axis=0)
    hi = uv.max(axis=0)
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
    base = np.zeros((h*w,3), dtype=np.float32)
    np.add.at(base, flat, rgb.astype(np.float32))
    base /= np.maximum(count.reshape(-1,1), 1)
    base = base.reshape(h,w,3)
    
    # ==================== Unified defect heatmap ====================
    abs_gap = np.abs(gaps)
    defect = np.isfinite(abs_gap) & (abs_gap > float(defect_limit))
    values = abs_gap[defect]  # Always use absolute values
    
    # Auto-scale vmax
    vmax = max(float(np.percentile(values, 98)) if values.size else float(defect_limit) * 1.05,
               float(defect_limit) * 1.05)
    vmin = float(defect_limit if vmin is None else vmin)
    
    # Conventional heatmap: gray(qualified) -> yellow -> orange -> red
    # t: 0 = at limit (just qualified), 1 = max defect
    t = np.clip((values - vmin) / (vmax - vmin + 1e-12), 0, 1)
    
    heat = np.zeros((len(values), 3), dtype=np.float32)
    
    # Gray (0.75, 0.75, 0.75) at t=0 -> Yellow (1, 1, 0) at t=0.33
    # -> Orange (1, 0.5, 0) at t=0.66 -> Red (1, 0, 0) at t=1.0
    
    # Segment 1: Gray -> Yellow (t: 0 -> 0.33)
    mask1 = t <= 0.33
    tt1 = t[mask1] / 0.33
    heat[mask1, 0] = 0.75 + 0.25 * tt1  # R: 0.75 -> 1.0
    heat[mask1, 1] = 0.75 + 0.25 * tt1  # G: 0.75 -> 1.0
    heat[mask1, 2] = 0.75 - 0.75 * tt1  # B: 0.75 -> 0.0
    
    # Segment 2: Yellow -> Orange (t: 0.33 -> 0.66)
    mask2 = (t > 0.33) & (t <= 0.66)
    tt2 = (t[mask2] - 0.33) / 0.33
    heat[mask2, 0] = 1.0  # R: 1.0
    heat[mask2, 1] = 1.0 - 0.5 * tt2  # G: 1.0 -> 0.5
    heat[mask2, 2] = 0.0  # B: 0.0
    
    # Segment 3: Orange -> Red (t: 0.66 -> 1.0)
    mask3 = t > 0.66
    tt3 = (t[mask3] - 0.66) / 0.34
    heat[mask3, 0] = 1.0  # R: 1.0
    heat[mask3, 1] = 0.5 - 0.5 * tt3  # G: 0.5 -> 0.0
    heat[mask3, 2] = 0.0  # B: 0.0
    
    defect_indices = np.flatnonzero(defect)
    
    # Pixel-level aggregation: keep max defect value per pixel
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    pixel_values = np.full(h*w, -np.inf, dtype=float)
    pixel_ids = flat[defect_indices]
    
    for p, value, colour in zip(pixel_ids, values, heat):
        if value > pixel_values[p]:
            pixel_values[p] = value
            overlay.reshape(-1, 4)[p, :3] = np.clip(colour, 0, 1) * 255
            overlay.reshape(-1, 4)[p, 3] = 255
    
    return {
        'base_rgb': (np.clip(base, 0, 1) * 255).astype(np.uint8),
        'overlay_rgba': overlay,
        'u_axis': u, 'v_axis': v, 'origin': origin, 'bounds': np.r_[lo, hi],
        'pixel_size': size, 'vmin': vmin, 'vmax': vmax, 'count': count,
        'defect_mask': defect,
        'uv': uv,
        'pixel_defect_values': pixel_values.reshape(h, w),
        'pixel_defect_abs': pixel_values.reshape(h, w),
    }