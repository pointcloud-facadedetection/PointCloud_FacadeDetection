"""质量热力图三维点取色：rasterize_facade 光栅 + UV 投影查询。

与 ViewportRenderService.apply_quality_colors 使用同一套窗口筛选、值域
映射、色标与投影轴约定；区别仅在于这里直接对任意三维点数组取色
（无代理索引层），供离线导出（带热力图的下采样 PLY）使用。
"""
from __future__ import annotations

import numpy as np

from algorithms.facade.projection import rasterize_facade
from config.settings import Config
from services.heatmap_spec import (
    defect_colormap,
    heatmap_spec,
    normalize_heatmap_mode,
)


def rasterize_quality_heatmap(quality, mode):
    """把单份质量结果按指定模式光栅化为 UV 热力图；数据不足返回 None。"""
    if not isinstance(quality, dict):
        return None
    spec = heatmap_spec(normalize_heatmap_mode(mode))
    method = spec.get('method', 'ruler')
    metric = spec.get('metric', 'flatness')

    comparison = quality.get('quality_comparison') or {}
    method_data = (comparison.get('methods') or {}).get(method) or {}
    metric_data = method_data.get(metric) or {}
    windows = metric_data.get('windows') or []
    if not isinstance(windows, list) or not windows:
        return None

    centers = np.asarray([r.get('center_xyz', [np.nan] * 3) for r in windows],
                         dtype=np.float32).reshape(-1, 3)
    values = np.asarray([r.get(spec['value_key'], np.nan) for r in windows],
                        dtype=np.float32).reshape(-1)
    failed = np.asarray([not bool(r.get(spec['pass_key'], True))
                         for r in windows], dtype=bool)
    valid = np.isfinite(centers).all(axis=1) & np.isfinite(values) & failed
    if not np.any(valid):
        return None
    centers = centers[valid]
    values = values[valid]

    limit = float((quality.get('thresholds') or {}).get(
        spec['limit_key'], 4.0))

    plane = np.asarray((quality.get('overall') or {}).get('plane_model')
                       or quality.get('plane_model') or [], dtype=float)
    if plane.size != 4:
        return None
    plane = plane / (np.linalg.norm(plane[:3]) + 1e-12)

    u_axis = np.asarray(quality.get('projection_u_axis', []), dtype=np.float64)
    v_axis = np.asarray(quality.get('projection_v_axis', []), dtype=np.float64)
    if u_axis.size != 3 or v_axis.size != 3:
        from algorithms.geometry import classify_plane, plane_axes
        facade_type, _, _, _ = classify_plane(plane[:3])
        u_axis, v_axis = plane_axes(plane[:3], facade_type)

    origin = np.asarray(quality.get('projection_origin',
                                    np.mean(centers, axis=0)),
                        dtype=np.float64).reshape(3)

    excess = np.maximum(np.abs(values) - limit, 0.0)
    scale = max(
        float(np.percentile(excess[excess > 0], 98)) if np.any(excess > 0) else 0.0,
        limit * 0.15, 1e-6)
    t = np.clip(excess / scale, 0.0, 1.0)
    heat_colors = defect_colormap(t)

    values_m = (values / 1000.0).astype(np.float64)
    limit_m = float(limit) / 1000.0
    global_vmax_m = float((limit + scale) / 1000.0)

    return rasterize_facade(
        centers.astype(np.float64),
        np.full((len(centers), 3), 0.7),
        plane,
        values_m,
        limit_m,
        pixel_size=0.01,
        defect_colors=heat_colors,
        vmin=limit_m,
        vmax=global_vmax_m,
        max_size=2400,
        projection_origin=origin,
        projection_u_axis=u_axis,
        projection_v_axis=v_axis)


def sample_heatmap_colors(raster, points):
    """把三维点投影到光栅 UV 平面查询热力颜色。

    返回 ``(colors, covered)``：colors 为 (N,3) float32 [0,1]；covered 标记
    落在画布内且对应像素不透明的点。画布外的点直接判未覆盖而不做边缘
    截断，避免相邻立面的点被错染到本立面画布边缘的热力色。
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    overlay = raster['overlay_rgba']
    lo = np.asarray(raster['bounds'][:2], dtype=np.float64)
    hi = np.asarray(raster['bounds'][2:], dtype=np.float64)
    size = float(raster['pixel_size'])
    h_pix, w_pix = overlay.shape[:2]

    rel = pts - np.asarray(raster['origin'], dtype=np.float64).reshape(3)
    dom_u = rel @ np.asarray(raster['u_axis'], dtype=np.float64)
    dom_v = rel @ np.asarray(raster['v_axis'], dtype=np.float64)
    inside = ((dom_u >= lo[0]) & (dom_u <= hi[0])
              & (dom_v >= lo[1]) & (dom_v <= hi[1]))

    px = np.clip(((dom_u - lo[0]) / size).astype(int), 0, w_pix - 1)
    py = np.clip((h_pix - 1 - (dom_v - lo[1]) / size).astype(int), 0, h_pix - 1)
    pix = overlay[py, px]
    covered = inside & (pix[:, 3] > 0)
    return pix[:, :3].astype(np.float32) / 255.0, covered


def colorize_points_with_quality_reports(points, base_colors, quality_reports,
                                         mode):
    """按各立面质量报告依次为点云叠加热力色。

    多份报告顺序叠加，后者覆盖前者（与 render_quality_reports 一致）。
    base_colors 为 None 时未覆盖点填统一灰（Config.FACADE_BASE_COLOR，
    与视口立面基准色一致），即"纯灰点云 + 热力染色"的导出语义；显式
    传入 base_colors 时未覆盖点保留各色。
    返回 ``(colors, stats)``；stats 含 applied_facades 与 colored_points。
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if base_colors is None:
        colors = np.full((len(pts), 3), Config.FACADE_BASE_COLOR,
                         dtype=np.float32)
    else:
        colors = np.asarray(base_colors, dtype=np.float32).reshape(-1, 3).copy()
        if colors.size and float(colors.max()) > 1.0 + 1e-6:
            colors /= 255.0
    covered_any = np.zeros(len(pts), dtype=bool)
    applied = 0
    for quality in quality_reports or []:
        raster = rasterize_quality_heatmap(quality, mode)
        if raster is None:
            continue
        heat, covered = sample_heatmap_colors(raster, pts)
        if not np.any(covered):
            continue
        colors[covered] = np.clip(heat[covered], 0.0, 1.0)
        covered_any |= covered
        applied += 1
    stats = {'applied_facades': int(applied),
             'colored_points': int(np.count_nonzero(covered_any))}
    return colors, stats
