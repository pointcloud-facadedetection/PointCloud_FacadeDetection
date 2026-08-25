"""立面正射图服务层：对接检测立面、组装 API 响应。

职责：从局部点云 + facade 字典提取立面点、调用 generate_facade_orthophoto、
颜色着色、PNG 编码、输出 mapping（index_map 保留 numpy，避免巨型 Python list）。
几何与坐标换算逻辑均在 facade_render.py。
"""

import base64

import cv2
import numpy as np

from facadeDetection.algorithms.geometry import fit_plane_svd

from .facade_render import (
    FacadeOrthorenderer,
    _apply_canny_edge_enhancement,
    compute_ortho_region_for_facade,
    DEFAULT_ORTHO_POINT_FRACTION,
    generate_facade_orthophoto,
    infer_facade_point_layout,
    normalize_plane_params,
    try_read_ply_intensity,
)

GLOBAL_ORTHO_MAX_PIXELS = 2_000_000


def _encode_png_b64(rgb_uint8):
    if rgb_uint8.ndim == 2:
        rgb_uint8 = cv2.cvtColor(rgb_uint8, cv2.COLOR_GRAY2RGB)
    success, buf = cv2.imencode(
        '.png',
        cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR),
    )
    if not success:
        raise ValueError("正射图 PNG 编码失败")
    return base64.b64encode(buf.tobytes()).decode('ascii')


def _serialize_frame(meta, plane_model=None):
    frame = {
        'origin': np.asarray(meta['origin'], dtype=float).tolist(),
        'u_axis': np.asarray(meta['u_axis'], dtype=float).tolist(),
        'v_axis': np.asarray(meta['v_axis'], dtype=float).tolist(),
        'normal': np.asarray(meta['normal'], dtype=float).tolist(),
        'u_min': float(meta['u_min']),
        'u_max': float(meta['u_max']),
        'v_min': float(meta['v_min']),
        'v_max': float(meta['v_max']),
        'resolution_m': float(meta['resolution']),
        'width_px': int(meta['width']),
        'height_px': int(meta['height']),
    }
    if plane_model is not None:
        frame['plane_model'] = normalize_plane_params(plane_model).tolist()
    return frame


def _build_point_mappings(facade_points, global_indices, meta, index_map):
    """将 render 元数据整理为 API 用的 3D 点 ↔ 像素列表（仅可见点）。"""
    del index_map
    point_px = meta['point_px']
    point_py = meta['point_py']
    proj_u = meta['proj_u']
    proj_v = meta['proj_v']
    proj_d = meta['proj_d']
    visible = (point_px >= 0) & (point_py >= 0)
    mappings = []
    for local_i in np.flatnonzero(visible):
        local_i = int(local_i)
        mappings.append({
            'local_index': local_i,
            'global_index': int(global_indices[local_i]),
            'px': int(point_px[local_i]),
            'py': int(point_py[local_i]),
            'visible': True,
            'uv': [float(proj_u[local_i]), float(proj_v[local_i])],
            'depth_m': float(proj_d[local_i]),
            'xyz': facade_points[local_i].tolist(),
        })
    return mappings


def prepare_ortho_api_payload(result, include_index_map=False):
    """HTTP 响应用：移除 numpy 内部字段，并精简超大 mapping。"""
    payload = {k: v for k, v in result.items() if k != '_match_image_rgb'}
    mapping = dict(payload.get('mapping') or {})
    mapping['points'] = []
    if not include_index_map:
        mapping.pop('index_map', None)
    inlier = mapping.get('inlier_indices')
    if inlier is not None and hasattr(inlier, 'tolist'):
        mapping['inlier_indices'] = inlier.tolist()
    payload['mapping'] = mapping
    return payload


def _compose_rgb(render_gray, index_map, facade_colors, meta):
    height, width = render_gray.shape
    del height, width
    rgb = np.full(render_gray.shape + (3,), 28, dtype=np.uint8)
    mask = index_map >= 0
    if np.any(mask):
        rgb[mask] = (np.clip(facade_colors[index_map[mask]], 0.0, 1.0) * 255.0).astype(np.uint8)
    return rgb


def _enhance_ortho_for_matching(bgr_uint8, valid_mask=None):
    """
    为 2D 特征匹配增强正射图：Canny 边缘 + CLAHE 局部对比度，并保留部分颜色。
    GlueStick / LightGlue 与现场照片匹配时使用此版本。
    """
    bgr = np.asarray(bgr_uint8)
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    enhanced_bgr, enhanced_gray = _apply_canny_edge_enhancement(bgr)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(enhanced_gray)
    cl_bgr = cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR).astype(np.float32)
    color_f = enhanced_bgr.astype(np.float32)
    out_bgr = np.clip(color_f * 0.55 + cl_bgr * 0.45, 0, 255).astype(np.uint8)
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != out_rgb.shape[:2]:
            raise ValueError("正射有效像素掩码尺寸不一致")
        out_rgb[~mask] = 28
    return out_rgb


def render_facade_orthographic(
    points,
    facade,
    colors=None,
    intensity=None,
    ply_path=None,
    resolution_m=0.015,
    render_mode='color',
    padding_m=None,
    margin_m=None,
    point_fraction=DEFAULT_ORTHO_POINT_FRACTION,
    depth_band_m=None,
    margin_left_ratio=None,
    margin_right_ratio=None,
    margin_bottom_ratio=None,
    margin_top_ratio=None,
    include_point_mappings=True,
):
    """
    生成立面正射图及 3D↔2D 映射（供 HTTP API / 自动匹配使用）。

    沿选定立面法向投影；以立面包围盒为核心向外扩展局部范围，并保证包含
    该立面 inlier。``point_fraction`` 仅保留用于兼容旧 API。仅支持颜色模式。
    """
    del padding_m, render_mode, depth_band_m, margin_m

    if facade is None:
        raise ValueError("未指定立面")
    if resolution_m <= 0:
        raise ValueError("resolution_m 必须大于 0")
    requested_facade_id = int(facade.get('id', -1))
    if requested_facade_id < 0:
        raise ValueError("立面缺少有效 id")

    inlier_raw = facade.get('inlier_indices')
    facade_indices = np.asarray([] if inlier_raw is None else inlier_raw, dtype=np.int32)
    if facade_indices.size < 3:
        raise ValueError("立面有效点数不足，无法生成正射图")

    points = np.asarray(points, dtype=float)
    facade_points = points[facade_indices]

    region = compute_ortho_region_for_facade(
        points,
        facade,
        facade_indices,
        target_fraction=float(point_fraction),
        margin_left_ratio=margin_left_ratio,
        margin_right_ratio=margin_right_ratio,
        margin_bottom_ratio=margin_bottom_ratio,
        margin_top_ratio=margin_top_ratio,
    )
    uv_bounds = region['uv_bounds']
    frame = region['frame']
    selected = region['selected_indices']
    del frame

    render_points = points[selected]
    render_indices = selected.astype(np.int32, copy=False)

    if colors is not None:
        colors = np.asarray(colors, dtype=float)
        if colors.shape[0] != points.shape[0]:
            raise ValueError("颜色数量与点云点数不一致")
        render_colors = colors[selected]
    else:
        render_colors = None

    if intensity is not None:
        intensity = np.asarray(intensity, dtype=float).reshape(-1)
        if intensity.shape[0] != points.shape[0]:
            raise ValueError("强度数量与点云点数不一致")
        render_intensity = intensity[selected]
    elif ply_path:
        full_intensity = try_read_ply_intensity(ply_path, len(points))
        render_intensity = full_intensity[selected] if full_intensity is not None else None
    else:
        render_intensity = None

    plane_model = facade.get('plane_model') or fit_plane_svd(facade_points)
    plane_params = normalize_plane_params(plane_model)

    points_input, gen_mode, channel_type = infer_facade_point_layout(
        render_points,
        colors=render_colors,
        intensity=render_intensity,
    )
    if channel_type == 'xyz':
        gen_mode = 'depth'

    orthophoto_bgr, index_map, meta, render_gray = generate_facade_orthophoto(
        points_input,
        resolution=float(resolution_m),
        render_mode=gen_mode,
        facade=facade,
        uv_bounds=uv_bounds,
        max_pixels=(
            GLOBAL_ORTHO_MAX_PIXELS
            if region.get('region_mode') == 'global'
            else 12_000_000
        ),
    )
    effective_resolution_m = float(meta['resolution'])

    display_colors = render_colors
    if display_colors is None:
        display_colors = np.full((len(render_points), 3), 0.72, dtype=float)

    valid_mask = index_map >= 0
    if channel_type == 'rgb':
        match_image_rgb = _enhance_ortho_for_matching(
            orthophoto_bgr, valid_mask=valid_mask
        )
        rgb = match_image_rgb
    elif channel_type == 'intensity':
        match_image_rgb = _enhance_ortho_for_matching(
            orthophoto_bgr, valid_mask=valid_mask
        )
        rgb = cv2.cvtColor(orthophoto_bgr, cv2.COLOR_BGR2RGB)
    else:
        base_bgr = cv2.cvtColor(
            _compose_rgb(render_gray, index_map, display_colors, meta),
            cv2.COLOR_RGB2BGR,
        )
        match_image_rgb = _enhance_ortho_for_matching(
            base_bgr, valid_mask=valid_mask
        )
        rgb = match_image_rgb

    if include_point_mappings:
        point_mappings = _build_point_mappings(render_points, render_indices, meta, index_map)
        visible_count = int(len(point_mappings))
    else:
        point_mappings = []
        visible_count = int(np.count_nonzero(meta['point_px'] >= 0))

    frame = _serialize_frame(meta, plane_params)
    filled_pixels = int(np.sum(index_map >= 0))

    return {
        'facade_id': requested_facade_id,
        'requested_facade_id': requested_facade_id,
        'width_px': frame['width_px'],
        'height_px': frame['height_px'],
        'resolution_m': effective_resolution_m,
        'physical_width_m': float(meta['u_max'] - meta['u_min']),
        'physical_height_m': float(meta['v_max'] - meta['v_min']),
        'render_mode': 'color',
        'point_channel': channel_type,
        'edge_enhanced': bool(meta.get('edge_enhanced')),
        'point_count': int(len(render_points)),
        'total_point_count': int(len(points)),
        'facade_point_count': int(len(facade_indices)),
        'selected_point_count': int(len(render_points)),
        'target_point_fraction': region.get('target_fraction'),
        'selected_point_fraction': float(region['selected_fraction']),
        'region_scale': float(region['region_scale']),
        'region_depth_m': float(region['region_depth_m']),
        'region_mode': region.get('region_mode', 'local'),
        'margin_u_m': float(region['margin_u_m']),
        'margin_v_m': float(region['margin_v_m']),
        'margin_left_m': float(region['margin_left_m']),
        'margin_right_m': float(region['margin_right_m']),
        'margin_bottom_m': float(region['margin_bottom_m']),
        'margin_top_m': float(region['margin_top_m']),
        'margin_left_ratio': float(region['margin_left_ratio']),
        'margin_right_ratio': float(region['margin_right_ratio']),
        'margin_bottom_ratio': float(region['margin_bottom_ratio']),
        'margin_top_ratio': float(region['margin_top_ratio']),
        'core_width_m': float(region['core_width_m']),
        'core_height_m': float(region['core_height_m']),
        'visible_point_count': visible_count,
        'filled_pixel_count': filled_pixels,
        'image_base64': _encode_png_b64(rgb),
        'image_mime': 'image/png',
        '_match_image_rgb': match_image_rgb,
        'match_image_mode': 'color_enhanced',
        'plane': {
            'plane_model': plane_params.tolist(),
            'center': frame['origin'],
        },
        'mapping': {
            'index_map': index_map.reshape(-1),
            'inlier_indices': render_indices,
            'frame': frame,
            'points': point_mappings,
        },
    }


__all__ = ['render_facade_orthographic', 'prepare_ortho_api_payload']
