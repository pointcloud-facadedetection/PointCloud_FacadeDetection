"""立面正射渲染核心：几何建系、Z-Buffer 栅格化、像素↔3D 坐标换算。

本模块只处理「给定立面点 + 平面参数 → 栅格图 + index_map + meta」；
不依赖 Flask、立面检测结果结构或 PNG/base64 编码。
上层编排见 ortho_renderer.py。
"""

import numpy as np
import cv2


from facadeDetection.algorithms.geometry import plane_axes


def _build_local_frame_from_plane(plane_params, points_3d):
    """由立面检测平面方程构建立部局部坐标系 (U, V, N)。"""
    plane = np.asarray(plane_params, dtype=np.float64).reshape(4)
    normal = plane[:3].copy()
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        raise ValueError('立面平面法向无效')
    normal /= norm

    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(normal, up)) > 0.99:
        up = np.array([0.0, 1.0, 0.0])

    u_axis = np.cross(normal, up)
    u_axis /= np.linalg.norm(u_axis) + 1e-12
    v_axis = np.cross(u_axis, normal)
    v_axis /= np.linalg.norm(v_axis) + 1e-12
    if v_axis[2] < 0:
        v_axis = -v_axis
        u_axis = -u_axis
    return u_axis, v_axis, normal


def resolve_facade_ortho_frame(facade, pts_xyz):
    """
    优先使用立面检测结果的 center / bbox_2d / plane_model 建系，
    与验证四角、热力图等模块保持一致。
    """
    pts_xyz = np.asarray(pts_xyz, dtype=np.float64)
    facade = facade or {}
    center = facade.get('center')
    origin = (
        np.asarray(center, dtype=float)
        if center is not None
        else np.mean(pts_xyz, axis=0)
    )

    bbox = facade.get('bbox_2d') or {}
    u_axis = np.asarray(bbox.get('u_axis', []), dtype=float)
    v_axis = np.asarray(bbox.get('v_axis', []), dtype=float)
    plane = np.asarray(facade.get('plane_model', [0, 0, 1, 0]), dtype=float)
    normal = plane[:3] / (np.linalg.norm(plane[:3]) + 1e-12)

    if u_axis.size == 3 and v_axis.size == 3:
        u_axis = u_axis / (np.linalg.norm(u_axis) + 1e-12)
        v_axis = v_axis / (np.linalg.norm(v_axis) + 1e-12)
    else:
        u_axis, v_axis = plane_axes(normal, facade.get('type'))

    if v_axis[2] < 0:
        v_axis = -v_axis
        u_axis = -u_axis

    return origin, u_axis, v_axis, normal


DEFAULT_ORTHO_MARGIN_M = 15.0
DEFAULT_ORTHO_DEPTH_BAND_M = 10.0
DEFAULT_ORTHO_COPLANAR_BAND_M = 0.5
DEFAULT_ORTHO_POINT_FRACTION = 2.0 / 3.0
DEFAULT_ORTHO_MIN_MARGIN_M = 3.0
DEFAULT_ORTHO_MAX_MARGIN_M = 15.0
DEFAULT_ORTHO_MARGIN_RATIO = 0.25
MAX_ORTHO_PIXELS = 12_000_000
MAX_ORTHO_DIMENSION_PX = 8192


def compute_facade_uv_bounds(facade_points, facade, margin_m=DEFAULT_ORTHO_MARGIN_M):
    """以立面 inlier 的 UV 包围盒加 margin 作为正射画布范围。"""
    pts_xyz = np.asarray(facade_points, dtype=np.float64)[:, :3]
    origin, u_axis, v_axis, normal = resolve_facade_ortho_frame(facade, pts_xyz)
    centered = pts_xyz - origin
    fu = centered @ u_axis
    fv = centered @ v_axis
    uv_bounds = (
        float(fu.min()) - margin_m,
        float(fu.max()) + margin_m,
        float(fv.min()) - margin_m,
        float(fv.max()) + margin_m,
    )
    return uv_bounds, (origin, u_axis, v_axis, normal)


def _project_points_to_facade_uv(points, frame):
    origin, u_axis, v_axis, normal = frame
    pts_xyz = np.asarray(points, dtype=np.float64)[:, :3]
    centered = pts_xyz - origin
    return (
        centered @ u_axis,
        centered @ v_axis,
        centered @ normal,
    )


def compute_ortho_region_for_facade(
    points,
    facade,
    facade_indices,
    target_fraction=DEFAULT_ORTHO_POINT_FRACTION,
    min_margin_m=DEFAULT_ORTHO_MIN_MARGIN_M,
    max_margin_m=DEFAULT_ORTHO_MAX_MARGIN_M,
    margin_ratio=DEFAULT_ORTHO_MARGIN_RATIO,
    margin_left_ratio=None,
    margin_right_ratio=None,
    margin_bottom_ratio=None,
    margin_top_ratio=None,
    depth_band_m=DEFAULT_ORTHO_DEPTH_BAND_M,
):
    """
    以选中立面的 UV 包围盒为核心，按立面宽高向外扩展局部匹配范围。

    扩展量由立面自身尺寸决定，而不是由全点云点数决定，避免大场景中为了
    凑足固定点数比例而把远处建筑、地面和其他立面投进正射图。
    ``target_fraction`` 仅为兼容旧调用保留，不再参与范围计算。
    """
    del target_fraction
    points = np.asarray(points, dtype=np.float64)
    facade_indices = np.asarray(facade_indices, dtype=np.int32).reshape(-1)
    n_total = int(points.shape[0])
    if n_total < 3:
        raise ValueError('点云点数不足')
    if facade_indices.size < 3:
        raise ValueError('立面 inlier 不足')

    valid_inliers = facade_indices[(facade_indices >= 0) & (facade_indices < n_total)]
    if valid_inliers.size < 3:
        raise ValueError('立面有效 inlier 索引不足')
    facade_points = points[valid_inliers]
    core_bounds, frame = compute_facade_uv_bounds(facade_points, facade, margin_m=0.0)
    pu, pv, pd = _project_points_to_facade_uv(points, frame)
    global_bounds = (
        float(np.min(pu)),
        float(np.max(pu)),
        float(np.min(pv)),
        float(np.max(pv)),
    )

    core_width = max(float(core_bounds[1] - core_bounds[0]), 1e-6)
    core_height = max(float(core_bounds[3] - core_bounds[2]), 1e-6)
    directional_ratios = (
        margin_left_ratio,
        margin_right_ratio,
        margin_bottom_ratio,
        margin_top_ratio,
    )
    global_mode = all(
        ratio is not None and float(ratio) >= 1.0
        for ratio in directional_ratios
    )

    if global_mode:
        uv_bounds = global_bounds
        selected = np.arange(n_total, dtype=np.int32)
        margin_left = max(0.0, float(core_bounds[0] - uv_bounds[0]))
        margin_right = max(0.0, float(uv_bounds[1] - core_bounds[1]))
        margin_bottom = max(0.0, float(core_bounds[2] - uv_bounds[2]))
        margin_top = max(0.0, float(uv_bounds[3] - core_bounds[3]))
        region_depth_m = float(np.max(np.abs(pd))) if pd.size else 0.0
    else:
        def resolve_margin(size, directional_ratio, distance_to_global):
            if directional_ratio is not None:
                ratio = float(np.clip(directional_ratio, 0.0, 1.0))
                if ratio >= 1.0:
                    return max(0.0, float(distance_to_global))
                return size * ratio
            ratio = float(margin_ratio)
            if ratio <= 0:
                return 0.0
            return float(np.clip(
                size * ratio, float(min_margin_m), float(max_margin_m)
            ))

        margin_left = resolve_margin(
            core_width, margin_left_ratio, core_bounds[0] - global_bounds[0]
        )
        margin_right = resolve_margin(
            core_width, margin_right_ratio, global_bounds[1] - core_bounds[1]
        )
        margin_bottom = resolve_margin(
            core_height, margin_bottom_ratio, core_bounds[2] - global_bounds[2]
        )
        margin_top = resolve_margin(
            core_height, margin_top_ratio, global_bounds[3] - core_bounds[3]
        )
        uv_bounds = (
            float(core_bounds[0] - margin_left),
            float(core_bounds[1] + margin_right),
            float(core_bounds[2] - margin_bottom),
            float(core_bounds[3] + margin_top),
        )

        inlier_mask = np.zeros(n_total, dtype=bool)
        inlier_mask[valid_inliers] = True
        u_min, u_max, v_min, v_max = uv_bounds
        local_mask = (
            (pu >= u_min) & (pu <= u_max)
            & (pv >= v_min) & (pv <= v_max)
            & (np.abs(pd) <= float(depth_band_m))
        )
        selected = np.flatnonzero(local_mask | inlier_mask)
        region_depth_m = float(depth_band_m)

    if selected.size < 3:
        raise ValueError('正射区域有效点数不足')

    return {
        'uv_bounds': uv_bounds,
        'frame': frame,
        'selected_indices': selected.astype(np.int32, copy=False),
        'target_fraction': None,
        'selected_fraction': float(selected.size / n_total),
        'target_point_count': int(selected.size),
        'total_point_count': n_total,
        'region_scale': float(
            (core_width + margin_left + margin_right)
            * (core_height + margin_bottom + margin_top)
            / (core_width * core_height)
        ),
        'region_depth_m': region_depth_m,
        'region_mode': 'global' if global_mode else 'local',
        'margin_u_m': max(margin_left, margin_right),
        'margin_v_m': max(margin_bottom, margin_top),
        'margin_left_m': margin_left,
        'margin_right_m': margin_right,
        'margin_bottom_m': margin_bottom,
        'margin_top_m': margin_top,
        'margin_left_ratio': float(
            margin_ratio if margin_left_ratio is None else margin_left_ratio
        ),
        'margin_right_ratio': float(
            margin_ratio if margin_right_ratio is None else margin_right_ratio
        ),
        'margin_bottom_ratio': float(
            margin_ratio if margin_bottom_ratio is None else margin_bottom_ratio
        ),
        'margin_top_ratio': float(
            margin_ratio if margin_top_ratio is None else margin_top_ratio
        ),
        'core_width_m': core_width,
        'core_height_m': core_height,
    }


def select_points_in_ortho_region(points, frame, uv_bounds, depth_band_m=DEFAULT_ORTHO_DEPTH_BAND_M):
    """筛选立面周围局部点云：UV 在 bbox 内且距立面平面不超过 depth_band_m。"""
    origin, u_axis, v_axis, normal = frame
    u_min, u_max, v_min, v_max = uv_bounds
    pts_xyz = np.asarray(points, dtype=np.float64)[:, :3]
    centered = pts_xyz - origin
    pu = centered @ u_axis
    pv = centered @ v_axis
    pd = centered @ normal
    mask = (
        (pu >= u_min) & (pu <= u_max)
        & (pv >= v_min) & (pv <= v_max)
        & (np.abs(pd) <= depth_band_m)
    )
    return np.flatnonzero(mask)


def select_facade_ortho_points(
    points,
    facade_indices,
    frame,
    uv_bounds,
    coplanar_band_m=DEFAULT_ORTHO_COPLANAR_BAND_M,
):
    """
    正射渲染点集：必选当前立面 inlier，另可含 UV 范围内且贴近该平面的共面点。

    不用大 depth_band 扫全厚度，避免把垂直相邻立面（如立面 1 的大面）投影进来。
    """
    facade_indices = np.asarray(facade_indices, dtype=np.int32).reshape(-1)
    origin, u_axis, v_axis, normal = frame
    u_min, u_max, v_min, v_max = uv_bounds
    n_total = int(np.asarray(points).shape[0])

    inlier_mask = np.zeros(n_total, dtype=bool)
    valid_inliers = facade_indices[(facade_indices >= 0) & (facade_indices < n_total)]
    inlier_mask[valid_inliers] = True

    pts_xyz = np.asarray(points, dtype=np.float64)[:, :3]
    centered = pts_xyz - origin
    pu = centered @ u_axis
    pv = centered @ v_axis
    pd_abs = np.abs(centered @ normal)
    in_uv = (pu >= u_min) & (pu <= u_max) & (pv >= v_min) & (pv <= v_max)
    coplanar_extra = in_uv & (pd_abs <= float(coplanar_band_m))
    return np.flatnonzero(inlier_mask | coplanar_extra)


def cap_ortho_resolution(
    resolution,
    uv_bounds,
    max_pixels=MAX_ORTHO_PIXELS,
    max_dim=MAX_ORTHO_DIMENSION_PX,
):
    """画布过大时自动加粗分辨率，避免分配 GB 级数组。"""
    u_min, u_max, v_min, v_max = uv_bounds
    res = float(resolution)
    w = h = 0
    for _ in range(32):
        w = max(1, int(np.ceil((u_max - u_min) / res)) + 1)
        h = max(1, int(np.ceil((v_max - v_min) / res)) + 1)
        if w * h <= max_pixels and max(w, h) <= max_dim:
            return res, w, h
        scale = max(np.sqrt((w * h) / max_pixels), max(w, h) / max_dim, 1.05)
        res *= scale
    raise ValueError(
        f'正射区域过大（约 {w}×{h} px），请减小 margin 或增大 resolution_m（当前 {resolution} m/px）'
    )


def normalize_plane_params(plane_model):
    plane = np.asarray(plane_model, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(plane[:3])
    if norm < 1e-12:
        raise ValueError("立面平面模型无效")
    return np.array([
        plane[0] / norm,
        plane[1] / norm,
        plane[2] / norm,
        plane[3] / norm,
    ], dtype=float)


def _frame_resolution(meta):
    """读取 meta / 序列化 frame 中的米/像素分辨率。"""
    if 'resolution' in meta:
        return float(meta['resolution'])
    if 'resolution_m' in meta:
        return float(meta['resolution_m'])
    raise KeyError('meta 缺少 resolution / resolution_m')


class FacadeOrthorenderer:
    """
    将 3D 立面点云正射渲染为 2D 图像，并构建像素与 3D 点的映射表。

    - index_map[py, px] = 立面局部点索引（-1 表示无点）
    - meta_info 含 origin / u_axis / v_axis / normal，可用于 UV ↔ 世界坐标换算
    """

    def __init__(self, resolution=0.02):
        """resolution: 米/像素，0.02 表示 2 cm/px。"""
        self.resolution = float(resolution)

    def _build_local_frame(self, plane_params, points_3d):
        """由平面方程或 SVD 构建立面局部坐标系 (U, V, N)。"""
        if plane_params is not None:
            return _build_local_frame_from_plane(plane_params, points_3d)
        mean = np.mean(points_3d[:, :3], axis=0)
        _, _, vh = np.linalg.svd(points_3d[:, :3] - mean, full_matrices=False)
        normal = vh[2]
        normal /= np.linalg.norm(normal) + 1e-12
        u_axis = vh[0]
        u_axis /= np.linalg.norm(u_axis) + 1e-12
        v_axis = np.cross(normal, u_axis)
        v_axis /= np.linalg.norm(v_axis) + 1e-12
        if v_axis[2] < 0:
            v_axis = -v_axis
            u_axis = -u_axis
        return u_axis, v_axis, normal

    def render(self, facade_points, plane_params=None, use_intensity=True):
        """
        :param facade_points: (N, 3) 或 (N, 4)[含强度]
        :param plane_params: [A, B, C, D] 平面方程 Ax+By+Cz+D=0
        :param use_intensity: 使用第 4 通道作为灰度；否则用 signed depth
        :return: (render_img, index_map, meta_info)
        """
        pts_xyz = np.asarray(facade_points[:, :3], dtype=np.float64)
        if pts_xyz.shape[0] < 3:
            raise ValueError("立面点数不足，无法正射渲染")

        has_intensity = facade_points.shape[1] >= 4 and use_intensity
        intensity = np.asarray(facade_points[:, 3], dtype=np.float64) if has_intensity else None

        u_axis, v_axis, normal = self._build_local_frame(plane_params, pts_xyz)
        origin = np.mean(pts_xyz, axis=0)

        centered = pts_xyz - origin
        proj_u = centered @ u_axis
        proj_v = centered @ v_axis
        proj_d = centered @ normal

        u_min, u_max = float(proj_u.min()), float(proj_u.max())
        v_min, v_max = float(proj_v.min()), float(proj_v.max())

        width = max(1, int(np.ceil((u_max - u_min) / self.resolution)) + 1)
        height = max(1, int(np.ceil((v_max - v_min) / self.resolution)) + 1)

        canvas = np.zeros((height, width), dtype=np.float32)
        z_buffer = np.full((height, width), np.inf, dtype=np.float32)
        index_map = np.full((height, width), -1, dtype=np.int32)

        px = np.clip(((proj_u - u_min) / self.resolution).astype(np.int32), 0, width - 1)
        py = np.clip(((v_max - proj_v) / self.resolution).astype(np.int32), 0, height - 1)

        point_px = np.full(len(pts_xyz), -1, dtype=np.int32)
        point_py = np.full(len(pts_xyz), -1, dtype=np.int32)

        for i in range(len(pts_xyz)):
            x, y = int(px[i]), int(py[i])
            d = abs(float(proj_d[i]))
            if d < z_buffer[y, x]:
                z_buffer[y, x] = d
                index_map[y, x] = i
                if has_intensity:
                    canvas[y, x] = float(intensity[i])
                else:
                    canvas[y, x] = float(proj_d[i])

        for y in range(height):
            for x in range(width):
                local_idx = int(index_map[y, x])
                if local_idx >= 0:
                    point_px[local_idx] = x
                    point_py[local_idx] = y

        valid_mask = index_map >= 0
        if np.any(valid_mask):
            c_min = float(canvas[valid_mask].min())
            c_max = float(canvas[valid_mask].max())
            if c_max > c_min:
                render_img = ((canvas - c_min) / (c_max - c_min) * 255.0).astype(np.uint8)
            else:
                render_img = np.zeros((height, width), dtype=np.uint8)
                render_img[valid_mask] = 128
        else:
            render_img = np.zeros((height, width), dtype=np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        render_img = cv2.morphologyEx(render_img, cv2.MORPH_CLOSE, kernel)

        meta_info = {
            'origin': origin,
            'u_axis': u_axis,
            'v_axis': v_axis,
            'normal': normal,
            'u_min': u_min,
            'u_max': u_max,
            'v_min': v_min,
            'v_max': v_max,
            'resolution': self.resolution,
            'width': width,
            'height': height,
            'point_px': point_px,
            'point_py': point_py,
            'proj_u': proj_u,
            'proj_v': proj_v,
            'proj_d': proj_d,
        }

        return render_img, index_map, meta_info


def pixel_to_uv(px, py, meta):
    """正射图像素 → 立面 UV（米）。"""
    res = _frame_resolution(meta)
    u = float(meta['u_min']) + float(px) * res
    v = float(meta['v_max']) - float(py) * res
    return u, v


def uv_to_pixel(u, v, meta):
    """立面 UV（米）→ 正射图像素（最近整数）。"""
    res = _frame_resolution(meta)
    px = int(round((float(u) - float(meta['u_min'])) / res))
    py = int(round((float(meta['v_max']) - float(v)) / res))
    return px, py


def uv_to_world(u, v, meta, depth=0.0):
    """立面 UV + 沿法向深度 → 世界坐标 (Z-up)。"""
    origin = np.asarray(meta['origin'], dtype=np.float64)
    u_axis = np.asarray(meta['u_axis'], dtype=np.float64)
    v_axis = np.asarray(meta['v_axis'], dtype=np.float64)
    normal = np.asarray(meta['normal'], dtype=np.float64)
    return origin + u * u_axis + v * v_axis + float(depth) * normal


def pixel_to_world(px, py, meta, depth=0.0):
    """正射图像素 → 世界坐标。"""
    u, v = pixel_to_uv(px, py, meta)
    return uv_to_world(u, v, meta, depth=depth)


def world_to_pixel(xyz, meta):
    """世界坐标 → 正射图像素（浮点）。"""
    point = np.asarray(xyz, dtype=np.float64).reshape(3)
    origin = np.asarray(meta['origin'], dtype=np.float64)
    u_axis = np.asarray(meta['u_axis'], dtype=np.float64)
    v_axis = np.asarray(meta['v_axis'], dtype=np.float64)
    res = _frame_resolution(meta)
    centered = point - origin
    u = float(centered @ u_axis)
    v = float(centered @ v_axis)
    px = (u - float(meta['u_min'])) / res
    py = (float(meta['v_max']) - v) / res
    return px, py, u, v


def colors_are_meaningful(colors, atol=0.02):
    """判断点云 RGB 是否为真实颜色（非统一灰度占位）。"""
    if colors is None:
        return False
    cols = np.asarray(colors, dtype=np.float64)
    if cols.ndim != 2 or cols.shape[1] < 3 or cols.shape[0] == 0:
        return False
    if float(np.max(cols) - np.min(cols)) < atol:
        return False
    channel_std = np.std(cols, axis=0)
    if float(np.max(channel_std)) < atol:
        return False
    return True


def try_read_ply_intensity(ply_path, expected_count):
    """尽力从 PLY 读取 per-vertex 强度（Open3D tensor 或 ASCII 头解析）。"""
    if not ply_path:
        return None
    try:
        import open3d as o3d

        pcd_t = o3d.t.io.read_point_cloud(str(ply_path))
        for name in ('intensity', 'Intensity', 'scalar_intensity', 'reflectance', 'scalar'):
            if name in pcd_t.point:
                arr = np.asarray(pcd_t.point[name].numpy(), dtype=np.float64).reshape(-1)
                if len(arr) == int(expected_count):
                    return arr
    except Exception:
        pass

    try:
        with open(ply_path, 'rb') as handle:
            header_bytes = b''
            while True:
                line = handle.readline()
                if not line:
                    break
                header_bytes += line
                if line.strip() == b'end_header':
                    break
        header = header_bytes.decode('ascii', errors='ignore')
        if 'format ascii' not in header.lower():
            return None

        prop_names = []
        for raw_line in header.splitlines():
            parts = raw_line.strip().split()
            if len(parts) == 3 and parts[0] == 'property':
                prop_names.append(parts[2].lower())

        intensity_idx = None
        for candidate in ('intensity', 'scalar_intensity', 'reflectance', 'scalar'):
            if candidate in prop_names:
                intensity_idx = prop_names.index(candidate)
                break
        if intensity_idx is None:
            return None

        values = []
        with open(ply_path, 'r', encoding='utf-8', errors='ignore') as handle:
            in_body = False
            for raw_line in handle:
                line = raw_line.strip()
                if not in_body:
                    if line == 'end_header':
                        in_body = True
                    continue
                if not line:
                    continue
                parts = line.split()
                if len(parts) <= intensity_idx:
                    continue
                values.append(float(parts[intensity_idx]))
                if len(values) >= int(expected_count):
                    break
        if len(values) != int(expected_count):
            return None
        return np.asarray(values, dtype=np.float64)
    except Exception:
        return None


def infer_facade_point_layout(facade_xyz, colors=None, intensity=None):
    """
    根据点云维度推断正射渲染输入与强制 render_mode。

    返回:
        points_input: (N,3|4|6)
        render_mode: 'rgb' | 'intensity' | 'depth'
        channel_type: 同 render_mode
    """
    facade_xyz = np.asarray(facade_xyz, dtype=np.float64)
    if facade_xyz.ndim != 2 or facade_xyz.shape[1] < 3:
        raise ValueError('立面点坐标无效')

    if colors_are_meaningful(colors):
        cols = np.asarray(colors, dtype=np.float64)
        if float(cols.max()) <= 1.0 + 1e-6:
            rgb255 = np.clip(cols[:, :3] * 255.0, 0, 255)
        else:
            rgb255 = np.clip(cols[:, :3], 0, 255)
        return (
            np.column_stack([facade_xyz, rgb255]),
            'rgb',
            'rgb',
        )

    if intensity is not None:
        intensity = np.asarray(intensity, dtype=np.float64).reshape(-1)
        if intensity.shape[0] == facade_xyz.shape[0]:
            i_min, i_max = float(intensity.min()), float(intensity.max())
            if i_max > i_min + 1e-9:
                return (
                    np.column_stack([facade_xyz, intensity]),
                    'intensity',
                    'intensity',
                )

    return facade_xyz[:, :3], 'depth', 'xyz'


def _rasterize_index_map(px, py, depth, height, width):
    """向量化 Z-Buffer：按深度从远到近写入，最近点覆盖。"""
    n = len(px)
    depth = np.abs(np.asarray(depth, dtype=np.float32))
    order = np.argsort(-depth, kind='stable')
    px_s = px[order]
    py_s = py[order]
    idx_s = np.arange(n, dtype=np.int32)[order]

    index_map = np.full((height, width), -1, dtype=np.int32)
    index_map[py_s, px_s] = idx_s

    z_buffer = np.full((height, width), np.inf, dtype=np.float32)
    z_buffer[py_s, px_s] = depth[order]
    return index_map, z_buffer, order


def _fill_point_pixel_lookup(index_map, n_points):
    """由 index_map 反查每个点的可见像素坐标。"""
    point_px = np.full(int(n_points), -1, dtype=np.int32)
    point_py = np.full(int(n_points), -1, dtype=np.int32)
    vis_y, vis_x = np.nonzero(index_map >= 0)
    if vis_y.size == 0:
        return point_px, point_py
    local_idx = index_map[vis_y, vis_x]
    point_px[local_idx] = vis_x.astype(np.int32)
    point_py[local_idx] = vis_y.astype(np.int32)
    return point_px, point_py


def _apply_canny_edge_enhancement(orthophoto_bgr, render_gray=None):
    """对纯几何深度正射图叠加 Canny 边缘，增强窗框/楼层线特征。"""
    orthophoto_bgr = np.asarray(orthophoto_bgr)
    if orthophoto_bgr.ndim == 2:
        gray = orthophoto_bgr.astype(np.uint8)
        orthophoto_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        gray = cv2.cvtColor(orthophoto_bgr, cv2.COLOR_BGR2GRAY)

    if render_gray is not None:
        gray = np.asarray(render_gray, dtype=np.uint8)

    edges = cv2.Canny(gray, 50, 150)
    edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    enhanced_bgr = cv2.addWeighted(orthophoto_bgr, 0.6, edge_bgr, 0.4, 0)
    enhanced_gray = cv2.addWeighted(gray, 0.6, edges, 0.4, 0)
    return enhanced_bgr, enhanced_gray


def generate_facade_orthophoto(
    facade_points,
    resolution=0.015,
    render_mode='auto',
    facade=None,
    uv_bounds=None,
    max_pixels=MAX_ORTHO_PIXELS,
):
    """
    从选中的 3D 立面点云生成 2D 立面正射图（检测坐标系 + Z-Buffer + 形态学闭运算）。

    :param facade_points: (N, 3) XYZ；(N, 4) 含强度；(N, 6) 含 RGB (0–255 或 0–1)
    :param facade: 可选，立面检测字典（center/bbox_2d/plane_model），用于与全链路对齐
    :return: orthophoto_bgr, index_map, meta_info, render_gray
    """
    facade_points = np.asarray(facade_points, dtype=np.float64)
    pts_xyz = facade_points[:, :3]
    num_cols = facade_points.shape[1]
    if pts_xyz.shape[0] < 3:
        raise ValueError('立面点数不足，无法生成正射图')

    if facade is not None:
        mean_center, u_axis, v_axis, normal = resolve_facade_ortho_frame(facade, pts_xyz)
    else:
        mean_center = np.mean(pts_xyz, axis=0)
        centered_xyz = pts_xyz - mean_center
        _, _, vh = np.linalg.svd(centered_xyz, full_matrices=False)
        normal = vh[2]
        normal /= np.linalg.norm(normal) + 1e-12
        u_axis = vh[0]
        u_axis /= np.linalg.norm(u_axis) + 1e-12
        v_axis = np.cross(normal, u_axis)
        v_axis /= np.linalg.norm(v_axis) + 1e-12
        if v_axis[2] < 0:
            v_axis = -v_axis
            u_axis = -u_axis

    centered_xyz = pts_xyz - mean_center
    proj_u = centered_xyz @ u_axis
    proj_v = centered_xyz @ v_axis
    proj_d = centered_xyz @ normal

    if uv_bounds is not None:
        u_min, u_max, v_min, v_max = [float(x) for x in uv_bounds]
    else:
        u_min, u_max = float(proj_u.min()), float(proj_u.max())
        v_min, v_max = float(proj_v.min()), float(proj_v.max())

    resolution, width, height = cap_ortho_resolution(
        resolution,
        (u_min, u_max, v_min, v_max),
        max_pixels=max_pixels,
    )

    px = np.clip(((proj_u - u_min) / resolution).astype(np.int32), 0, width - 1)
    py = np.clip(((v_max - proj_v) / resolution).astype(np.int32), 0, height - 1)

    mode = (render_mode or 'auto').lower()
    if mode == 'auto':
        if num_cols >= 6:
            mode = 'rgb'
        elif num_cols >= 4:
            mode = 'intensity'
        else:
            mode = 'depth'
    elif mode not in ('rgb', 'intensity', 'depth'):
        raise ValueError("render_mode 必须是 rgb、intensity、depth 或 auto")

    index_map, z_buffer, order = _rasterize_index_map(px, py, proj_d, height, width)
    px_w = px[order]
    py_w = py[order]
    ortho_gray = np.zeros((height, width), dtype=np.float32)
    orthophoto = np.zeros((height, width, 3), dtype=np.uint8)

    if mode == 'rgb':
        rgb_data = facade_points[:, 3:6].astype(np.float64)
        if rgb_data.max() <= 1.0 + 1e-6:
            rgb_data = np.clip(rgb_data * 255.0, 0, 255)
        bgr_data = rgb_data[:, [2, 1, 0]].astype(np.uint8)
        orthophoto[py_w, px_w] = bgr_data[order]
        ortho_gray[py_w, px_w] = (
            0.299 * rgb_data[:, 0] + 0.587 * rgb_data[:, 1] + 0.114 * rgb_data[:, 2]
        )[order].astype(np.float32)

    elif mode == 'intensity':
        intensity_data = facade_points[:, 3].astype(np.float64)
        ortho_gray[py_w, px_w] = intensity_data[order].astype(np.float32)
        mask = z_buffer != np.inf
        render_u8 = np.zeros((height, width), dtype=np.uint8)
        if np.any(mask):
            i_min, i_max = ortho_gray[mask].min(), ortho_gray[mask].max()
            render_u8[mask] = np.clip(
                (ortho_gray[mask] - i_min) / (i_max - i_min + 1e-6) * 255,
                0, 255,
            ).astype(np.uint8)
        orthophoto = cv2.cvtColor(render_u8, cv2.COLOR_GRAY2BGR)
        ortho_gray = render_u8.astype(np.float32)

    else:
        ortho_gray[py_w, px_w] = np.abs(proj_d[order]).astype(np.float32)
        mask = z_buffer != np.inf
        render_u8 = np.zeros((height, width), dtype=np.uint8)
        if np.any(mask):
            d_min, d_max = ortho_gray[mask].min(), ortho_gray[mask].max()
            render_u8[mask] = np.clip(
                (ortho_gray[mask] - d_min) / (d_max - d_min + 1e-6) * 255,
                0, 255,
            ).astype(np.uint8)
        orthophoto = cv2.cvtColor(render_u8, cv2.COLOR_GRAY2BGR)
        ortho_gray = render_u8.astype(np.float32)

    point_px, point_py = _fill_point_pixel_lookup(index_map, len(pts_xyz))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    orthophoto = cv2.morphologyEx(orthophoto, cv2.MORPH_CLOSE, kernel)

    valid_mask = index_map >= 0
    render_gray = np.zeros((height, width), dtype=np.uint8)
    if np.any(valid_mask):
        g_min = float(ortho_gray[valid_mask].min())
        g_max = float(ortho_gray[valid_mask].max())
        if g_max > g_min:
            render_gray[valid_mask] = np.clip(
                (ortho_gray[valid_mask] - g_min) / (g_max - g_min) * 255,
                0, 255,
            ).astype(np.uint8)
        else:
            render_gray[valid_mask] = 128
    render_gray = cv2.morphologyEx(render_gray, cv2.MORPH_CLOSE, kernel)

    edge_enhanced = False
    if mode == 'depth':
        orthophoto, render_gray = _apply_canny_edge_enhancement(orthophoto, render_gray)
        edge_enhanced = True

    meta_info = {
        'origin': mean_center,
        'u_axis': u_axis,
        'v_axis': v_axis,
        'normal': normal,
        'u_min': u_min,
        'u_max': u_max,
        'v_min': v_min,
        'v_max': v_max,
        'resolution': float(resolution),
        'width': width,
        'height': height,
        'point_px': point_px,
        'point_py': point_py,
        'proj_u': proj_u,
        'proj_v': proj_v,
        'proj_d': proj_d,
        'render_mode': mode,
        'edge_enhanced': edge_enhanced,
    }

    return orthophoto, index_map, meta_info, render_gray

