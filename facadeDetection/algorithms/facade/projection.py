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
    
    # ==================== 使用有符号距离区分凹陷/凸起 ====================
    if defect_values is None:
        # 使用传入的 signed_gap（有符号距离）
        abs_gap = np.abs(gaps)
        defect = np.isfinite(abs_gap) & (abs_gap > float(flatness_limit))
        values = gaps[defect]  # 保留有符号值
        abs_values = np.abs(values)
        vmax = max(float(np.percentile(abs_values, 98)) if abs_values.size else float(flatness_limit) * 1.05,
                   float(flatness_limit) * 1.05)
        
        # 颜色映射：凹陷(负) = 蓝色系，凸起(正) = 红色系，合格 = 绿色/透明
        # t: 0 = 刚好不合格, 1 = 严重不合格
        t = np.clip((abs_values - float(flatness_limit)) / (vmax - float(flatness_limit) + 1e-12), 0, 1)
        
        # 凹陷（负值）：蓝 -> 青
        # 凸起（正值）：黄 -> 红
        is_recessed = values < 0
        heat = np.zeros((len(values), 3), dtype=np.float32)
        
        # 凸起：黄色(0.2) -> 橙色 -> 红色(1.0)
        protrude_mask = ~is_recessed
        heat[protrude_mask, 0] = np.clip(0.5 + 0.5 * t[protrude_mask], 0, 1)  # R: 0.5->1.0
        heat[protrude_mask, 1] = np.clip(0.8 - 0.8 * t[protrude_mask], 0, 1)  # G: 0.8->0.0
        heat[protrude_mask, 2] = np.clip(0.2 - 0.2 * t[protrude_mask], 0, 1)  # B: 0.2->0.0
        
        # 凹陷：蓝色(0.2) -> 青色 -> 绿色(1.0)
        heat[is_recessed, 0] = np.clip(0.2 - 0.2 * t[is_recessed], 0, 1)      # R: 0.2->0.0
        heat[is_recessed, 1] = np.clip(0.5 + 0.3 * t[is_recessed], 0, 1)      # G: 0.5->0.8
        heat[is_recessed, 2] = np.clip(0.8 + 0.2 * t[is_recessed], 0, 1)       # B: 0.8->1.0
        
        defect_indices = np.flatnonzero(defect)
    else:
        values = np.asarray(defect_values, dtype=float).reshape(-1)
        defect_indices = np.flatnonzero(np.isfinite(values))
        values = values[defect_indices]
        heat = np.asarray(defect_colors, dtype=float).reshape(-1, 3)[defect_indices] if defect_colors is not None else np.zeros((len(values), 3))
        vmax = float(vmax if vmax is not None else (np.max(np.abs(values)) if len(values) else flatness_limit))

    vmin = float(flatness_limit if vmin is None else vmin)
    
    # ==================== 像素级聚合：保留最差值（按绝对值）====================
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    pixel_values = np.full(h*w, -np.inf, dtype=float)  # 存储绝对值用于比较
    pixel_signed = np.full(h*w, np.nan, dtype=float)   # 存储有符号值
    pixel_ids = flat[defect_indices]
    
    for p, value, colour in zip(pixel_ids, values, heat):
        if abs(value) > pixel_values[p]:
            pixel_values[p] = abs(value)
            pixel_signed[p] = value
            overlay.reshape(-1, 4)[p, :3] = np.clip(colour, 0, 1) * 255
            overlay.reshape(-1, 4)[p, 3] = 255
    
    return {
        'base_rgb': (np.clip(base, 0, 1) * 255).astype(np.uint8),
        'overlay_rgba': overlay,
        'u_axis': u, 'v_axis': v, 'origin': origin, 'bounds': np.r_[lo, hi],
        'pixel_size': size, 'vmin': vmin, 'vmax': vmax, 'count': count,
        'defect_mask': np.isfinite(gaps) & (np.abs(gaps) > float(flatness_limit)),
        'uv': uv,
        'pixel_defect_values': pixel_signed.reshape(h, w),  # 有符号值
        'pixel_defect_abs': pixel_values.reshape(h, w),     # 绝对值
        'polarity': 'signed: negative=recessed, positive=protruding'
    }