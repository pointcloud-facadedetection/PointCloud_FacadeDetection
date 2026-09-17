from __future__ import annotations
import numpy as np
import cv2
from algorithms.geometry import plane_axes


def rasterize_facade(points, colors, plane_model, defect_values, defect_limit,
                     pixel_size=0.01, max_size=6000, defect_colors=None,
                     vmin=None, vmax=None, base_points=None, base_colors=None,
                     projection_origin=None, projection_u_axis=None,
                     projection_v_axis=None, physical_cell_m=None,
                     gaussian_sigma=None,
                     min_splat_px=4,
                     uv_range=None):
    """将点云 + 缺陷值栅格化为底图 + 热力叠加图。

    Parameters
    ----------
    uv_range : (u_min, v_min, u_max, v_max) or None
        显式画布 UV 边界。若为 None，回退到 base_points 的鲁棒分位数
        （0.1% / 99.9%）。**绝不要用 min/max**：扫描点云中的地面、天空、
        远处建筑等离群点会把边界框拉大到数百米，导致 max_size 降采样后
        立面分辨率严重下降，warp 到照片后 splat 变成巨块。
    min_splat_px : int
        每个缺陷 splat 的最小像素半径，保证可视化可读性。
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    frame_pts = (pts if base_points is None
                 else np.asarray(base_points, dtype=float).reshape(-1, 3))
    gaps = np.asarray(defect_values, dtype=float).reshape(-1)
    rgb = np.asarray(
        colors if colors is not None else np.full((len(pts), 3), .75),
        dtype=float)
    frame_rgb = np.asarray(
        base_colors if base_colors is not None
        else np.full((len(frame_pts), 3), .75),
        dtype=float).reshape(-1, 3)
    if len(frame_rgb) != len(frame_pts):
        frame_rgb = np.full((len(frame_pts), 3), .75, dtype=float)
    if len(pts) == 0 or len(gaps) != len(pts):
        raise ValueError('points and defect_values must have equal non-zero length')

    # ---- 平面法向与局部坐标轴 ----
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
              if projection_origin is not None
              else np.mean(frame_pts, axis=0))

    uv = np.column_stack(((pts - origin) @ u, (pts - origin) @ v))
    frame_uv = np.column_stack(((frame_pts - origin) @ u,
                                (frame_pts - origin) @ v))
    finite_frame = np.all(np.isfinite(frame_uv), axis=1)
    frame_uv = frame_uv[finite_frame]
    frame_rgb = frame_rgb[finite_frame]
    if len(frame_uv) == 0:
        raise ValueError('base_points contains no finite facade points')

    # ---- 画布边界框：显式 uv_range > 鲁棒分位数（绝不用 min/max）----
    if uv_range is not None:
        uv_range = np.asarray(uv_range, dtype=float).reshape(4)
        lo = uv_range[:2].copy()
        hi = uv_range[2:].copy()
    else:
        # 0.1% / 99.9% 分位数：剔除离群点（地面、天空、远处建筑），
        # 同时保留立面边缘。对 2800 万点约有 2.8 万点被剔除。
        lo = np.quantile(frame_uv, 0.001, axis=0)
        hi = np.quantile(frame_uv, 0.999, axis=0)

    # 保证边界框至少包含所有缺陷点，避免裁剪掉真实缺陷
    if len(uv) > 0:
        finite_uv = np.all(np.isfinite(uv), axis=1)
        if np.any(finite_uv):
            lo = np.minimum(lo, np.quantile(uv[finite_uv], 0.0005, axis=0))
            hi = np.maximum(hi, np.quantile(uv[finite_uv], 0.9995, axis=0))

    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        raise ValueError('invalid uv bounds after robust estimation')
    if np.any(hi <= lo):
        raise ValueError(f'degenerate uv bounds lo={lo} hi={hi}')

    # ---- 画布尺寸（鲁棒边界 + 目标像素步长）----
    size = max(float(pixel_size), 1e-4)
    shape = np.maximum(np.ceil((hi - lo) / size).astype(int) + 1, 1)
    if max(shape) > max_size:
        size = max(size, float(np.max(hi - lo)) / max(max_size - 1, 1))
        shape = np.maximum(np.ceil((hi - lo) / size).astype(int) + 1, 1)

    h, w = int(shape[1]), int(shape[0])

    # ---- 底图（点云）栅格化：离群点会被 clip 到边缘，不会影响分辨率 ----
    x = np.clip(((uv[:, 0] - lo[0]) / size).astype(int), 0, w - 1)
    y = np.clip((h - 1 - (uv[:, 1] - lo[1]) / size).astype(int), 0, h - 1)
    flat = y * w + x
    count = np.bincount(flat, minlength=h * w).reshape(h, w).astype(np.int32)

    base_flat = np.zeros((h * w, 3), dtype=np.float32)
    frame_x = np.clip(((frame_uv[:, 0] - lo[0]) / size).astype(int), 0, w - 1)
    frame_y = np.clip((h - 1 - (frame_uv[:, 1] - lo[1]) / size).astype(int),
                      0, h - 1)
    frame_flat = frame_y * w + frame_x
    np.add.at(base_flat, frame_flat, frame_rgb.astype(np.float32))
    base_flat /= np.maximum(
        np.bincount(frame_flat, minlength=h * w).reshape(-1, 1), 1)
    base = base_flat.reshape(h, w, 3)
    facade_mask = np.bincount(frame_flat, minlength=h * w).reshape(h, w) > 0

    # ==================== 缺陷热力图 ====================
    abs_gap = np.abs(gaps)
    defect = np.isfinite(abs_gap) & (abs_gap > float(defect_limit))
    values = abs_gap[defect]

    if vmax is not None:
        vmax = float(vmax)
    else:
        vmax = max(
            float(np.percentile(values, 98)) if values.size
            else float(defect_limit) * 1.05,
            float(defect_limit) * 1.05,
        )
    vmin = float(defect_limit if vmin is None else vmin)

    # ---- 颜色映射 ----
    if defect_colors is not None:
        supplied = np.asarray(defect_colors, dtype=np.float32).reshape(-1, 3)
        if len(supplied) == len(pts):
            heat = np.clip(supplied[defect], 0.0, 1.0)
        elif len(supplied) == len(values):
            heat = np.clip(supplied, 0.0, 1.0)
        else:
            raise ValueError('defect_colors must align with points or defects')
    else:
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

    defect_indices = np.flatnonzero(defect)

    # ==================== 缺陷热力 physical-footprint 渲染 ====================
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    pixel_values = np.full(h * w, -np.inf, dtype=float)
    pixel_ids = flat[defect_indices]

    if len(pixel_ids):
        np.maximum.at(pixel_values, pixel_ids, values)

    if len(pixel_ids) > 0:
        if len(pixel_ids) > 1:
            order = np.argsort(-np.abs(values), kind="stable")
            sorted_px = pixel_ids[order]
            _, first = np.unique(sorted_px, return_index=True)
            keep_sel = order[first]
            pixel_ids = pixel_ids[keep_sel]
            values = values[keep_sel]
            heat = heat[keep_sel]

        vmax_d = float(np.max(values)) if len(values) else 1.0
        vmin_d = float(np.min(values)) if len(values) else 0.0
        value_range = max(vmax_d - vmin_d, 1e-6)

        # ---- splat 半径：min_splat_px 保底；physical_cell_m 提供上限 ----
        min_radius = max(1, int(min_splat_px))
        if physical_cell_m is not None and physical_cell_m > 0:
            cell_px_val = max(float(physical_cell_m) / max(float(size), 1e-6),
                              1.0)
            radius_px = max(cell_px_val * 0.5, float(min_radius))
        else:
            sigma = max(float(gaussian_sigma or 3.0), 0.5)
            radius_px = max(float(min_radius), sigma * 1.5)
        radius_x = max(1, int(np.ceil(radius_px)))
        radius_y = max(1, int(np.ceil(radius_px)))

        # ---- 物理网格边界：仅在格宽能容纳 radius 时才启用 ----
        if physical_cell_m is not None and physical_cell_m > 0:
            cell_px = max(1, int(round(physical_cell_m / max(float(size), 1e-6))))
            if cell_px < 2 * radius_x or cell_px < 2 * radius_y:
                cell_px = None
        else:
            cell_px = None

        importance = np.clip(
            (values - vmin_d) / value_range, 0.1, 1.0
        ).astype(np.float32)
        support = np.zeros((h, w), dtype=np.uint8)
        rgb_canvas = np.zeros((h, w, 3), dtype=np.float32)
        weight_canvas = np.zeros((h, w), dtype=np.float32)
        py_arr = (pixel_ids // w).astype(np.int32)
        px_arr = (pixel_ids % w).astype(np.int32)

        for px, py, colour, weight in zip(px_arr, py_arr, heat, importance):
            if cell_px is not None:
                grid_x0 = (int(px) // cell_px) * cell_px
                grid_x1 = min(grid_x0 + cell_px, w)
                grid_y0 = (int(py) // cell_px) * cell_px
                grid_y1 = min(grid_y0 + cell_px, h)
                x0 = max(grid_x0, int(px) - radius_x)
                x1 = min(grid_x1, int(px) + radius_x + 1)
                y0 = max(grid_y0, int(py) - radius_y)
                y1 = min(grid_y1, int(py) + radius_y + 1)
            else:
                x0 = max(0, int(px) - radius_x)
                x1 = min(w, int(px) + radius_x + 1)
                y0 = max(0, int(py) - radius_y)
                y1 = min(h, int(py) + radius_y + 1)

            if x1 <= x0 or y1 <= y0:
                continue

            region = weight_canvas[y0:y1, x0:x1]
            take = weight >= region
            if np.any(take):
                region[take] = weight
                rgb_canvas[y0:y1, x0:x1][take] = colour
            support[y0:y1, x0:x1] = 1

        if gaussian_sigma is not None and np.any(support):
            kernel_size = 3
            softened = cv2.GaussianBlur(
                rgb_canvas, (kernel_size, kernel_size),
                min(float(gaussian_sigma), 0.8))
            rgb_canvas[support > 0] = softened[support > 0]

        support = support & facade_mask.astype(np.uint8)

        overlay[:, :, :3] = np.clip(rgb_canvas * 255, 0, 255).astype(np.uint8)
        alpha_vals = np.clip(weight_canvas * 255, 0, 255)
        alpha_vals = np.where(alpha_vals > 0, np.maximum(alpha_vals, 210), 0)
        overlay[:, :, 3] = (alpha_vals * support).astype(np.uint8)
        overlay[overlay[:, :, 3] < 8] = 0

    # ---- 边缘羽化：仅对半透明边缘平滑 ----
    alpha_u8 = overlay[:, :, 3]
    if np.any(alpha_u8 > 0):
        edge_mask = (alpha_u8 > 0) & (alpha_u8 < 48)
        if np.any(edge_mask):
            alpha_smooth = cv2.GaussianBlur(
                alpha_u8.astype(np.float32), (3, 3), sigmaX=0.6)
            alpha_f = alpha_u8.astype(np.float32)
            alpha_f[edge_mask] = alpha_smooth[edge_mask]
            overlay[:, :, 3] = np.clip(alpha_f, 0, 255).astype(np.uint8)

    return {
        'base_rgb': (np.clip(base, 0, 1) * 255).astype(np.uint8),
        'facade_mask': facade_mask,
        'overlay_rgba': overlay,
        'u_axis': u, 'v_axis': v, 'origin': origin,
        'bounds': np.r_[lo, hi],
        'pixel_size': size,
        'vmin': vmin, 'vmax': vmax,
        'count': count,
        'defect_mask': defect,
        'uv': uv,
        'pixel_defect_values': pixel_values.reshape(h, w),
        'pixel_defect_abs': pixel_values.reshape(h, w),
    }