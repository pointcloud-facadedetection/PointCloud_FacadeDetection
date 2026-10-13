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
            'defect_segments': segments,
            'depression_xyz': np.asarray(result.get('depression_xyz'),
                                         dtype=np.float64),
        },
    }
