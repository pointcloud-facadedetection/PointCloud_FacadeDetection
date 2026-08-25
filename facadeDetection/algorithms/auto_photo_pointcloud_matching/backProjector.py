"""步骤 3：正射图像素 → 3D 物理坐标回溯（自动建立 2D-3D 映射）。"""

from __future__ import annotations

import numpy as np

from .facade_render import _frame_resolution, pixel_to_world


def mapping_to_render_assets(mapping, points):
    """
    从步骤一 ortho mapping 恢复 backproject 所需数据结构。

    返回:
        index_map (H,W), facade_points (M,3), meta_info
    """
    if mapping is None:
        raise ValueError('缺少正射图 mapping')

    frame = mapping.get('frame') or {}
    width = int(frame.get('width_px', 0))
    height = int(frame.get('height_px', 0))
    if width <= 0 or height <= 0:
        raise ValueError('mapping.frame 缺少有效宽高')

    flat = mapping.get('index_map')
    if flat is None:
        raise ValueError('mapping 缺少 index_map')
    index_map = np.asarray(flat, dtype=np.int32).reshape(height, width)

    indices_raw = mapping.get('inlier_indices')
    indices = np.asarray([] if indices_raw is None else indices_raw, dtype=int)
    if indices.size == 0:
        raise ValueError('mapping 缺少 inlier_indices')

    pts = np.asarray(points, dtype=float)
    facade_points = pts[indices]

    meta_info = {
        'origin': np.asarray(frame['origin'], dtype=np.float64),
        'u_axis': np.asarray(frame['u_axis'], dtype=np.float64),
        'v_axis': np.asarray(frame['v_axis'], dtype=np.float64),
        'normal': np.asarray(frame.get('normal', [0.0, 0.0, 1.0]), dtype=np.float64),
        'plane_model': np.asarray(frame.get('plane_model', []), dtype=np.float64),
        'u_min': float(frame['u_min']),
        'u_max': float(frame.get('u_max', frame['u_min'])),
        'v_min': float(frame.get('v_min', 0.0)),
        'v_max': float(frame['v_max']),
        'resolution': _frame_resolution(frame),
        'width': width,
        'height': height,
    }
    return index_map, facade_points, meta_info


class Facade3DBackprojector:
    """
    2D 匹配点 → 3D 世界坐标。

    查找策略（依次）:
      1. index_map 精确索引
      2. 邻域窗口内最近有效点
      3. 立面 UV 平面几何解析（空洞/玻璃区域）
    """

    def __init__(self, search_window_size=5):
        self.search_window_size = int(search_window_size)
        if self.search_window_size < 1:
            raise ValueError('search_window_size 必须 >= 1')

    def backproject(
        self,
        kpts_photo,
        kpts_render,
        index_map,
        facade_points,
        meta_info,
        render_scale=1.0,
    ):
        """
        :param kpts_photo: 照片像素 (N,2)
        :param kpts_render: 正射图像素 (N,2)
        :param index_map: 像素 → 立面局部点索引 (H,W)，-1 为空
        :param facade_points: 立面点世界坐标 (M,3+)
        :param meta_info: 正射渲染元数据（render meta 或 mapping.frame）
        :param render_scale: 正射图匹配坐标相对原始正射图的缩放（默认 1.0）
        :return: dict，含 image_points / object_points / details
        """
        kpts_photo = np.asarray(kpts_photo, dtype=np.float64).reshape(-1, 2)
        kpts_render = np.asarray(kpts_render, dtype=np.float64).reshape(-1, 2)
        if len(kpts_photo) != len(kpts_render):
            raise ValueError('照片与正射图匹配点数量不一致')
        if len(kpts_photo) == 0:
            return {
                'image_points': np.empty((0, 2), dtype=np.float32),
                'object_points': np.empty((0, 3), dtype=np.float32),
                'details': [],
            }

        index_map = np.asarray(index_map, dtype=np.int32)
        facade_points = np.asarray(facade_points, dtype=np.float64)
        scale = float(render_scale) if render_scale else 1.0
        if scale <= 0:
            raise ValueError('render_scale 必须大于 0')

        map_h, map_w = index_map.shape
        half_w = self.search_window_size // 2

        pts_2d = []
        pts_3d = []
        details = []

        for i, (pt_p, pt_r) in enumerate(zip(kpts_photo, kpts_render)):
            u_r = float(pt_r[0]) / scale
            v_r = float(pt_r[1]) / scale
            px = int(np.round(u_r))
            py = int(np.round(v_r))

            target_3d = None
            method = 'geometric'
            local_index = -1

            if 0 <= py < map_h and 0 <= px < map_w:
                local_index = int(index_map[py, px])
                if local_index >= 0:
                    target_3d = facade_points[local_index, :3]
                    method = 'index_map'

            if target_3d is None and 0 <= py < map_h and 0 <= px < map_w:
                y0 = max(0, py - half_w)
                y1 = min(map_h, py + half_w + 1)
                x0 = max(0, px - half_w)
                x1 = min(map_w, px + half_w + 1)
                sub = index_map[y0:y1, x0:x1]
                valid = np.argwhere(sub >= 0)
                if valid.size > 0:
                    best_dist = np.inf
                    best_idx = -1
                    for dy, dx in valid:
                        cy = y0 + int(dy)
                        cx = x0 + int(dx)
                        dist = (cx - px) ** 2 + (cy - py) ** 2
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = int(sub[dy, dx])
                    if best_idx >= 0:
                        local_index = best_idx
                        target_3d = facade_points[best_idx, :3]
                        method = 'neighborhood'

            if target_3d is None:
                target_3d = pixel_to_world(px, py, meta_info, depth=0.0)
                plane = np.asarray(meta_info.get('plane_model', []), dtype=float)
                normal = np.asarray(meta_info['normal'], dtype=float)
                if plane.size == 4:
                    plane_norm = np.linalg.norm(plane[:3])
                    normal_norm = np.linalg.norm(normal)
                    if plane_norm > 1e-12 and normal_norm > 1e-12:
                        plane = plane / plane_norm
                        normal = normal / normal_norm
                        denominator = float(np.dot(plane[:3], normal))
                        if abs(denominator) > 1e-9:
                            signed = float(np.dot(plane[:3], target_3d) + plane[3])
                            target_3d = target_3d - normal * (signed / denominator)
                method = 'geometric'

            pts_2d.append([float(pt_p[0]), float(pt_p[1])])
            pts_3d.append(np.asarray(target_3d, dtype=np.float64).reshape(3))
            details.append({
                'index': int(i),
                'lookup_method': method,
                'ortho_px': [u_r, v_r],
                'local_index': int(local_index) if local_index >= 0 else None,
                'xyz': np.asarray(target_3d, dtype=float).tolist(),
            })

        return {
            'image_points': np.asarray(pts_2d, dtype=np.float32),
            'object_points': np.asarray(pts_3d, dtype=np.float32),
            'details': details,
        }


def backproject_matches_to_3d(
    kpts_photo,
    kpts_render,
    mapping,
    points,
    render_scale=1.0,
    search_window_size=5,
):
    """
    函数式入口：2D 匹配对 → 2D-3D 对应点。

    :param mapping: 步骤一 render_facade_orthographic 返回的 mapping 字典
    :param points: 全点云 N×3（Z-up）
    """
    index_map, facade_points, meta_info = mapping_to_render_assets(mapping, points)
    projector = Facade3DBackprojector(search_window_size=search_window_size)
    result = projector.backproject(
        kpts_photo,
        kpts_render,
        index_map,
        facade_points,
        meta_info,
        render_scale=render_scale,
    )

    method_counts = {}
    for item in result['details']:
        method_counts[item['lookup_method']] = method_counts.get(item['lookup_method'], 0) + 1

    return {
        **result,
        'pair_count': int(len(result['image_points'])),
        'lookup_method_counts': method_counts,
    }


__all__ = [
    'Facade3DBackprojector',
    'mapping_to_render_assets',
    'backproject_matches_to_3d',
]
