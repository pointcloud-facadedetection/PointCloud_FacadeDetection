"""立面局部几何：由检测 bbox 计算四角 3D 坐标。"""

from __future__ import annotations

import numpy as np

from facadeDetection.algorithms.geometry import plane_axes


def compute_facade_bbox_corners_3d(facade, points=None):
    """
    根据立面 bbox_2d 计算局部坐标系下矩形四角的世界坐标 (Z-up)。

    顺序：左下 → 右下 → 右上 → 左上（沿 u 水平、v 竖直）。
    """
    center = np.asarray(facade.get('center', [0, 0, 0]), dtype=float)
    bbox = facade.get('bbox_2d') or {}
    u_axis = np.asarray(bbox.get('u_axis', []), dtype=float)
    v_axis = np.asarray(bbox.get('v_axis', []), dtype=float)

    if u_axis.size != 3 or v_axis.size != 3:
        plane = np.asarray(facade.get('plane_model', [0, 0, 1, 0]), dtype=float)
        normal = plane[:3] / (np.linalg.norm(plane[:3]) + 1e-12)
        u_axis, v_axis = plane_axes(normal, facade.get('type'))

    u_min = bbox.get('u_min')
    u_max = bbox.get('u_max')
    v_min = bbox.get('v_min')
    v_max = bbox.get('v_max')

    if None in (u_min, u_max, v_min, v_max) and points is not None:
        inlier_indices = np.asarray(facade.get('inlier_indices', []), dtype=int)
        pts = np.asarray(points, dtype=float)[inlier_indices]
        local_u = (pts - center) @ u_axis
        local_v = (pts - center) @ v_axis
        u_min, u_max = float(np.min(local_u)), float(np.max(local_u))
        v_min, v_max = float(np.min(local_v)), float(np.max(local_v))
    else:
        u_min = float(u_min)
        u_max = float(u_max)
        v_min = float(v_min)
        v_max = float(v_max)

    if u_max <= u_min or v_max <= v_min:
        raise ValueError('立面 bbox 无效，无法计算四角顶点')

    corner_spec = [
        ('左下', u_min, v_min),
        ('右下', u_max, v_min),
        ('右上', u_max, v_max),
        ('左上', u_min, v_max),
    ]
    corners = []
    for label, u, v in corner_spec:
        xyz = center + u * u_axis + v * v_axis
        corners.append({
            'label': label,
            'uv': [float(u), float(v)],
            'xyz': xyz.astype(float).tolist(),
        })
    return corners


__all__ = ['compute_facade_bbox_corners_3d']
