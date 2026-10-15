"""复核页交互式靠尺：在视口拾取点处实时放置单根靠尺。

只编排已有算法积木（prepare_surface / ruler_at），不引入新检测算法。
全部函数为纯计算（无 Qt / 无视口依赖），几何输出为世界坐标渲染规格，
由视口层负责转成 VTK actor。
"""
from __future__ import annotations

import numpy as np

from algorithms.facade.ruler_flatness_3d import prepare_surface, ruler_at
from algorithms.geometry import classify_plane, plane_axes

# 尺面相对墙面接触线的法向抬升，避免与点云 z-fighting
RULER_LIFT_M = 0.003


def ray_plane_intersect(origin, direction, plane_model):
    """射线与平面 ax+by+cz+d=0 求交，返回世界坐标交点或 None。

    平行（|n·dir|≈0）或交点在射线反方向（t<0）时返回 None。
    """
    plane = np.asarray(plane_model, dtype=np.float64).reshape(-1)
    if plane.size != 4:
        return None
    normal = plane[:3]
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return None
    normal = normal / norm
    d = float(plane[3]) / norm
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    denom = float(normal @ direction)
    if abs(denom) < 1e-12:
        return None
    t = -(float(normal @ np.asarray(origin, dtype=np.float64).reshape(3)) + d) / denom
    if t < 0:
        return None
    return np.asarray(origin, dtype=np.float64).reshape(3) + t * direction


def point_in_facade_bounds(point, geometry):
    """交点投影到立面 UV 坐标系，判定是否落在四边面 uv_bounds 内。

    ``geometry`` 为 facade_geometry dict（也可直接传 quality 报告，
    自动取其 facade_geometry 字段）。返回 (u, v, inside)；几何数据
    缺失时 inside 为 None（表示无法判定，调用方自行取舍）。
    """
    if isinstance(geometry, dict) and isinstance(
            geometry.get('facade_geometry'), dict):
        geometry = geometry['facade_geometry']
    if not isinstance(geometry, dict):
        return None, None, None
    try:
        origin = np.asarray(geometry['origin'], dtype=np.float64).reshape(3)
        u_axis = np.asarray(geometry['u_axis'], dtype=np.float64).reshape(3)
        v_axis = np.asarray(geometry['v_axis'], dtype=np.float64).reshape(3)
        u0, v0, u1, v1 = np.asarray(geometry['uv_bounds'],
                                    dtype=np.float64).reshape(4)
    except (KeyError, ValueError, TypeError):
        return None, None, None
    rel = np.asarray(point, dtype=np.float64).reshape(3) - origin
    u, v = float(rel @ u_axis), float(rel @ v_axis)
    return u, v, bool(u0 <= u <= u1 and v0 <= v <= v1)


def resolve_ruler_direction(facade):
    """靠尺方向：优先取质量结果的投影 u 轴，否则由平面法向推导。

    返回单位向量；立面信息不足时返回 None（prepare_surface 走 PCA 主方向）。
    """
    facade = facade or {}
    report = facade.get('quality_report')
    if isinstance(report, dict):
        axis = np.asarray(report.get('projection_u_axis') or [], dtype=float)
        if axis.shape == (3,) and np.linalg.norm(axis) > 1e-9:
            return axis / np.linalg.norm(axis)
    plane = np.asarray(facade.get('plane_model') or [], dtype=float).reshape(-1)
    if plane.size < 3 or np.linalg.norm(plane[:3]) < 1e-12:
        return None
    facade_type, _, _, _ = classify_plane(plane[:3])
    u_axis, _ = plane_axes(plane[:3], facade_type)
    return np.asarray(u_axis, dtype=float)


def extract_facade_points(facade, processed_raw_points, index=None):
    """按立面索引域从全量点云取出立面点（与批量导出链路同口径）。

    ``index`` 为 FacadeIndexService 的索引对象（has_source_mapping /
    proxy_to_source_ids），可为 None。点数不足时返回 None。
    """
    points = np.asarray(processed_raw_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        return None
    ids = np.asarray((facade or {}).get('proxy_indices')
                     or (facade or {}).get('inlier_indices') or [],
                     dtype=np.int64)
    if len(ids) and index is not None and index.has_source_mapping():
        ids = np.asarray(index.proxy_to_source_ids(ids, deduplicate=True),
                         dtype=np.int64)
    ids = ids[(ids >= 0) & (ids < len(points))]
    if len(ids) < 3:
        return None
    return points[ids]


def build_interactive_surface(points, plane_model, plane_origin=None,
                              ruler_dir=None, select_band=0.01,
                              hole_band=0.02, bin_size=0.04, top_q=1.0,
                              sor=None):
    """为交互靠尺准备 Surface；一次构建后可反复 place_ruler。"""
    return prepare_surface(
        points, ruler_dir=ruler_dir, select_band=select_band,
        hole_band=hole_band, bin_size=bin_size, top_q=top_q, sor=sor,
        plane_model=plane_model, plane_origin=plane_origin)


def place_ruler(surface, click_point, ruler_length=2.0, ruler_width=0.055,
                threshold_mm=None, max_hole_ratio=0.2):
    """以 click_point（世界坐标）在尺方向上的投影为尺心放置靠尺。

    返回 dict：ok/reason、gap_mm、exceeded（threshold_mm 给出时）、
    u_center，以及视口渲染规格 render（corners/outline/defect_segments/
    depression_xyz，均为世界坐标）。
    """
    click = np.asarray(click_point, dtype=np.float64).reshape(3)
    u_center = float((click - surface.centroid) @ surface.e_u)
    # 尺心钳制在实区范围内，拾取偏差落在立面外时贴边放置
    lo, hi = surface.u_span
    u_center = float(np.clip(u_center, lo + 1e-9, hi - 1e-9))

    result = ruler_at(surface, u_center, ruler_length, max_hole_ratio,
                      threshold_mm=threshold_mm)
    if not result.get('ok'):
        return {'ok': False, 'reason': result.get('reason', '靠尺放置失败'),
                'u_center': u_center}

    slope, intercept = surface.ref_line

    def _contact(u):
        return surface.to_world(u, slope * u + intercept)

    half_l = ruler_length / 2.0
    half_w = ruler_width / 2.0
    lift = surface.normal * RULER_LIFT_M
    u0, u1 = u_center - half_l, u_center + half_l
    corners = np.array([
        _contact(u0) - surface.e_v * half_w + lift,
        _contact(u1) - surface.e_v * half_w + lift,
        _contact(u1) + surface.e_v * half_w + lift,
        _contact(u0) + surface.e_v * half_w + lift,
    ])

    segments = []
    grid = result.get('ruler_defect_grid') or {}
    for cell in grid.get('grids') or []:
        if int(cell.get('point_count', 0)) <= 0:
            continue
        defect_mm = float(cell.get('max_defect_mm', 0.0))
        segments.append({
            'start': _contact(float(cell['u_start_m'])) + lift,
            'end': _contact(float(cell['u_end_m'])) + lift,
            'defect_mm': defect_mm,
            'exceed': (threshold_mm is not None
                       and abs(defect_mm) > float(threshold_mm)),
        })

    gap_mm = float(result.get('gap_mm', 0.0))
    return {
        'ok': True,
        'u_center': u_center,
        'gap_mm': gap_mm,
        'signed_gap_mm': float(result.get('signed_gap_mm', gap_mm)),
        'exceeded': (threshold_mm is not None
                     and gap_mm > float(threshold_mm)),
        'coverage_valid': bool(result.get('coverage_valid', True)),
        'hole_ratio': float(result.get('hole_ratio', 0.0)),
        'angle_deg': float(result.get('angle_deg', 0.0)),
        'render': {
            'corners': corners,
            'body_line': np.vstack((_contact(u0) + lift,
                                    _contact(u1) + lift)),
            'defect_segments': segments,
            'depression_xyz': np.asarray(result.get('depression_xyz'),
                                         dtype=np.float64),
        },
    }


def transparent_png_uv_rect(points, quality, pixel_size=0.01, max_size=3200):
    """重建 transparent PNG（裁剪后）对应的 UV 矩形。

    导出链路（heatmap_renderer.render_transparent_heatmap）的画布与裁剪：
    画布 = quality domain 外扩 5%，像素步长 = max(pixel_size, 长边/(max_size-1))，
    再按 facade_mask（全量点投影覆盖率 ≥2% 的行列）裁剪。
    本函数用同一份 processed_raw_points 复刻该过程，返回裁剪框的 UV 矩形。
    与模式无关（四种模式的画布/裁剪共用同一 facade_mask）。

    返回 dict：uv_bounds=(u0,v0,u1,v1)、canvas_wh=(W,H)、crop_xy=(x0,y0,x1,y1)、
    pixel_size；domain 缺失或点不足时返回 None。
    """
    if not isinstance(quality, dict):
        return None
    domain = (quality.get('quality_comparison') or {}).get('domain') \
        or quality.get('quality_domain') or {}
    try:
        u0 = float(domain['u_min_m']); v0 = float(domain['v_min_m'])
        u1 = float(domain['u_max_m']); v1 = float(domain['v_max_m'])
    except (KeyError, TypeError, ValueError):
        return None
    if not (u1 > u0 and v1 > v0):
        return None
    origin = np.asarray(quality.get('projection_origin') or [],
                        dtype=np.float64).reshape(-1)
    u_axis = np.asarray(quality.get('projection_u_axis') or [],
                        dtype=np.float64).reshape(-1)
    v_axis = np.asarray(quality.get('projection_v_axis') or [],
                        dtype=np.float64).reshape(-1)
    if origin.size != 3 or u_axis.size != 3 or v_axis.size != 3:
        return None
    u_axis = u_axis / np.linalg.norm(u_axis)
    v_axis = v_axis / np.linalg.norm(v_axis)

    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        return None

    du, dv = (u1 - u0) * 0.05, (v1 - v0) * 0.05
    lo = np.array([u0 - du, v0 - dv])
    hi = np.array([u1 + du, v1 + dv])
    size = max(float(pixel_size), float(np.max(hi - lo)) / max(max_size - 1, 1))
    width = int(np.ceil((hi[0] - lo[0]) / size)) + 1
    height = int(np.ceil((hi[1] - lo[1]) / size)) + 1

    counts = np.zeros(height * width, dtype=np.uint32)
    chunk = 4_000_000
    for start in range(0, len(pts), chunk):
        block = np.asarray(pts[start:start + chunk], dtype=np.float32)
        rel = block - origin.astype(np.float32)
        fu = rel @ u_axis.astype(np.float32)
        fv = rel @ v_axis.astype(np.float32)
        fx = np.clip(((fu - lo[0]) / size).astype(np.int64), 0, width - 1)
        fy = np.clip((height - 1 - (fv - lo[1]) / size).astype(np.int64),
                     0, height - 1)
        counts += np.bincount(fy * width + fx,
                              minlength=height * width).astype(np.uint32)
    mask = counts.reshape(height, width) > 0
    col_cov = mask.sum(axis=0) / float(height)
    row_cov = mask.sum(axis=1) / float(width)
    cols = np.where(col_cov >= 0.02)[0]
    rows = np.where(row_cov >= 0.02)[0]
    if len(cols) == 0 or len(rows) == 0:
        return None
    x0, x1 = int(cols[0]), int(cols[-1])
    y0, y1 = int(rows[0]), int(rows[-1])
    return {
        'uv_bounds': (float(lo[0] + x0 * size), float(hi[1] - y1 * size),
                      float(lo[0] + x1 * size), float(hi[1] - y0 * size)),
        'canvas_wh': (width, height),
        'crop_xy': (x0, y0, x1, y1),
        'pixel_size': float(size),
    }


def ruler_direction_from_points(center_point, second_point, plane_model):
    """两次平面交点推导靠尺方向：差向量投影到立面平面内并单位化。

    两点过近（退化方向，平面内分量 < 1mm）返回 None，由调用方提示重点。
    """
    plane = np.asarray(plane_model, dtype=np.float64).reshape(-1)
    if plane.size != 4 or np.linalg.norm(plane[:3]) < 1e-12:
        return None
    normal = plane[:3] / np.linalg.norm(plane[:3])
    d = (np.asarray(second_point, dtype=np.float64).reshape(3)
         - np.asarray(center_point, dtype=np.float64).reshape(3))
    d = d - float(d @ normal) * normal
    norm = np.linalg.norm(d)
    if norm < 1e-3:
        return None
    return d / norm

def place_ruler_on_strip(points, plane_model, center, direction,
                         ruler_length=2.0, ruler_width=0.055,
                         threshold_mm=None, max_hole_ratio=0.2,
                         select_band=0.01, hole_band=0.02, bin_size=0.04,
                         top_q=1.0, sor=None):
    """尺心 + 方向 → 尺宽条带取样 → prepare_surface / ruler_at。

    与批量链路同物理语义：剖面只取尺线两侧 ±ruler_width/2 内的点
    （全立面宽剖面会把局部缺陷稀释掉——缺陷在条带外方向上被
    同 bin 的正常点顶掉）。条带方向垂直分量由法向叉乘方向得到。
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    plane = np.asarray(plane_model, dtype=np.float64).reshape(-1)
    if plane.size != 4 or np.linalg.norm(plane[:3]) < 1e-12:
        return {'ok': False, 'reason': '立面平面参数无效'}
    normal = plane[:3] / np.linalg.norm(plane[:3])
    center = np.asarray(center, dtype=np.float64).reshape(3)
    u = np.asarray(direction, dtype=np.float64).reshape(3)
    u = u - float(u @ normal) * normal
    if np.linalg.norm(u) < 1e-9:
        return {'ok': False, 'reason': '靠尺方向与立面法向平行'}
    u = u / np.linalg.norm(u)
    across = np.cross(normal, u)
    dv = (points - center) @ across
    strip = points[np.abs(dv) <= float(ruler_width) / 2.0]
    if len(strip) < 3:
        return {'ok': False, 'reason': '靠尺条带内点数不足，请换个位置'}
    try:
        surface = build_interactive_surface(
            strip, plane_model, plane_origin=center, ruler_dir=u,
            select_band=select_band, hole_band=hole_band,
            bin_size=bin_size, top_q=top_q, sor=sor)
    except ValueError as exc:
        return {'ok': False, 'reason': str(exc)}
    result = place_ruler(surface, center, ruler_length, ruler_width,
                         threshold_mm, max_hole_ratio)
    result['strip_point_count'] = int(len(strip))
    return result
