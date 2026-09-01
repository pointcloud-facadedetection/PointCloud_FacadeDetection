from __future__ import annotations
import numpy as np
from algorithms.geometry import plane_axes


def rasterize_facade(points, colors, plane_model, defect_values, defect_limit,
                     pixel_size=0.01, max_size=6000, defect_colors=None,
                     vmin=None, vmax=None, base_points=None, base_colors=None,
                     projection_origin=None, projection_u_axis=None,
                     projection_v_axis=None):
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    frame_pts = pts if base_points is None else np.asarray(base_points, dtype=float).reshape(-1, 3)
    gaps = np.asarray(defect_values, dtype=float).reshape(-1)
    rgb = np.asarray(colors if colors is not None else np.full((len(pts), 3), .75), dtype=float)
    frame_rgb = np.asarray(base_colors if base_colors is not None else
                           np.full((len(frame_pts), 3), .75), dtype=float).reshape(-1, 3)
    if len(frame_rgb) != len(frame_pts):
        frame_rgb = np.full((len(frame_pts), 3), .75, dtype=float)
    if len(pts) == 0 or len(gaps) != len(pts):
        raise ValueError('points and defect_values must have equal non-zero length')
    
    n = np.asarray(plane_model[:3], dtype=float)
    n /= np.linalg.norm(n) + 1e-12
    if projection_u_axis is None or projection_v_axis is None:
        u, v = plane_axes(n, 'vertical_facade')
    else:
        u = np.asarray(projection_u_axis, dtype=float).reshape(3)
        v = np.asarray(projection_v_axis, dtype=float).reshape(3)
        u /= np.linalg.norm(u) + 1e-12
        v /= np.linalg.norm(v) + 1e-12
    origin = (np.asarray(projection_origin, dtype=float).reshape(3)
              if projection_origin is not None else np.mean(frame_pts, axis=0))
    uv = np.column_stack(((pts-origin) @ u, (pts-origin) @ v))
    frame_uv = np.column_stack(((frame_pts-origin) @ u, (frame_pts-origin) @ v))
    lo = frame_uv.min(axis=0)
    hi = frame_uv.max(axis=0)
    size = max(float(pixel_size), 1e-4)
    shape = np.maximum(np.ceil((hi-lo)/size).astype(int)+1, 1)
    if max(shape) > max_size:
        size = max(size, float(np.max(hi - lo)) / max(max_size - 1, 1))
        shape = np.maximum(np.ceil((hi-lo)/size).astype(int)+1, 1)
    h, w = int(shape[1]), int(shape[0])
    x = np.clip(((uv[:, 0]-lo[0])/size).astype(int), 0, w-1)
    y = np.clip((h-1-(uv[:, 1]-lo[1])/size).astype(int), 0, h-1)
    flat = y*w+x
    count = np.bincount(flat, minlength=h*w).reshape(h,w).astype(np.int32)
    base_flat = np.zeros((h*w, 3), dtype=np.float32)
    frame_x = np.clip(((frame_uv[:, 0]-lo[0])/size).astype(int), 0, w-1)
    frame_y = np.clip((h-1-(frame_uv[:, 1]-lo[1])/size).astype(int), 0, h-1)
    frame_flat = frame_y*w+frame_x
    np.add.at(base_flat, frame_flat, frame_rgb.astype(np.float32))
    base_flat /= np.maximum(np.bincount(frame_flat, minlength=h*w).reshape(-1, 1), 1)
    base = base_flat.reshape(h,w,3)
    
    # ==================== 统一缺陷热力图 ====================
    abs_gap = np.abs(gaps)
    if defect_colors is not None:
        defect = np.isfinite(abs_gap)
    else:
        defect = np.isfinite(abs_gap) & (abs_gap > float(defect_limit))
    values = abs_gap[defect]  # Always use absolute values

    if vmax is not None:
        vmax = float(vmax)
    else:
        vmax = max(float(np.percentile(values, 98)) if values.size else float(defect_limit) * 1.05,
                   float(defect_limit) * 1.05)
    
    vmin = float(defect_limit if vmin is None else vmin)
    
    # 当 defect_colors 传入时，直接使用传入的颜色
    if defect_colors is not None:
        supplied = np.asarray(defect_colors, dtype=np.float32).reshape(-1, 3)
        if len(supplied) == len(pts):
            heat = np.clip(supplied[defect], 0.0, 1.0)
        elif len(supplied) == len(values):
            heat = np.clip(supplied, 0.0, 1.0)
        else:
            raise ValueError('defect_colors must align with points or defects')
    else:
        # 仅当未传入 defect_colors 时才使用默认映射
        # 统一配色 - 青→黄→橙→红
        t = np.clip((values - vmin) / (vmax - vmin + 1e-12), 0, 1)
        
        heat = np.zeros((len(values), 3), dtype=np.float32)

        mask1 = t <= 0.33
        tt1 = t[mask1] / 0.33
        heat[mask1, 0] = 0.0 + 1.0 * tt1
        heat[mask1, 1] = 0.75 + 0.25 * tt1
        heat[mask1, 2] = 1.0 - 1.0 * tt1

        mask2 = (t > 0.33) & (t <= 0.66)
        tt2 = (t[mask2] - 0.33) / 0.33
        heat[mask2, 0] = 1.0
        heat[mask2, 1] = 1.0 - 0.5 * tt2
        heat[mask2, 2] = 0.0

        mask3 = t > 0.66
        tt3 = (t[mask3] - 0.66) / 0.34
        heat[mask3, 0] = 1.0
        heat[mask3, 1] = 0.5 - 0.5 * tt3
        heat[mask3, 2] = 0.0
    
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    pixel_values = np.full(h * w, -np.inf, dtype=float)
    if values.size:
        pixel_ids = flat[np.flatnonzero(defect)]
        order = np.argsort(values)
        pixel_ids = pixel_ids[order]
        heat_u8 = np.clip(heat[order] * 255.0, 0, 255).astype(np.uint8)
        values_sorted = values[order]
        overlay_flat = overlay.reshape(-1, 4)
        overlay_flat[pixel_ids, :3] = heat_u8
        overlay_flat[pixel_ids, 3] = 255
        pixel_values[pixel_ids] = values_sorted
    
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