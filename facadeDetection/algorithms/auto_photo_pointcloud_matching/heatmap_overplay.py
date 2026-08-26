"""步骤 5：3D 立面平整度热力图 → 2D 照片 Alpha 融合叠加。"""

from __future__ import annotations

# OpenCV Python 接口由二进制扩展动态导出，Pylint 无法静态识别成员。
# pylint: disable=no-member

import base64

import cv2
import numpy as np

from algorithms.geometry import signed_plane_distance


def _default_distortion(dist_coeffs=None):
    if dist_coeffs is None:
        return np.zeros((5, 1), dtype=np.float64)
    dist = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
    if dist.shape[0] < 5:
        padded = np.zeros((5, 1), dtype=np.float64)
        padded[: dist.shape[0]] = dist
        return padded
    return dist[:5]


def _encode_png_b64(bgr_uint8):
    ok, buf = cv2.imencode('.png', bgr_uint8)
    if not ok:
        raise ValueError('热力图 PNG 编码失败')
    return base64.b64encode(buf.tobytes()).decode('ascii')


def compute_facade_deviation_scalars(points_3d, plane_model, unit='mm'):
    """计算立面点到拟合平面的有符号距离（正=沿法向一侧，负=另一侧）。"""
    points = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    plane = np.asarray(plane_model, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(plane[:3])
    if norm < 1e-12:
        raise ValueError('立面平面模型无效')
    plane = plane / norm
    signed_m = signed_plane_distance(points, plane)
    if unit == 'mm':
        return (signed_m * 1000.0).astype(np.float32)
    return signed_m.astype(np.float32)


def orient_plane_toward_camera(plane_model, facade_center, camera_center):
    """法向朝向相机，便于「正偏差=朝向相机凸起」语义。"""
    plane = np.asarray(plane_model, dtype=np.float64).reshape(4).copy()
    norm = np.linalg.norm(plane[:3])
    if norm < 1e-12:
        return plane
    plane /= norm
    center = np.asarray(facade_center, dtype=np.float64).reshape(3)
    cam = np.asarray(camera_center, dtype=np.float64).reshape(3)
    if np.dot(plane[:3], cam - center) < 0:
        plane *= -1.0
    return plane


class FacadeHeatmapOverlay:
    """3D 标量场正向投影到照片并 Alpha 融合。"""

    def __init__(self, alpha=0.5, point_radius=6, blur_size=15):
        self.alpha = float(alpha)
        self.point_radius = int(point_radius)
        self.blur_size = int(blur_size)

    def overlay(
        self,
        photo_img,
        points_3d,
        scalars,
        R,
        T,
        K,
        dist_coeffs=None,
        val_range=None,
        draw_colorbar=True,
        grid_heatmap=None,
    ):
        """
        :param photo_img: BGR uint8 (H,W,3)
        :return: (blended_bgr, heatmap_bgr, meta)
        """
        photo = np.asarray(photo_img)
        if photo.ndim == 2:
            photo = cv2.cvtColor(photo, cv2.COLOR_GRAY2BGR)
        h, w = photo.shape[:2]

        dist = _default_distortion(dist_coeffs)
        scalars = np.asarray(scalars, dtype=np.float32).reshape(-1)
        pts_3d = np.ascontiguousarray(points_3d, dtype=np.float32).reshape(-1, 3)
        if len(scalars) != len(pts_3d):
            raise ValueError('标量数量与 3D 点数量不一致')

        if val_range is not None:
            min_v, max_v = float(val_range[0]), float(val_range[1])
        else:
            min_v = float(np.percentile(scalars, 5))
            max_v = float(np.percentile(scalars, 95))
        if max_v <= min_v:
            max_v = min_v + 1e-6

        norm_scalars = np.clip(
            (scalars - min_v) / (max_v - min_v) * 255.0,
            0,
            255,
        ).astype(np.uint8)

        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        T = np.asarray(T, dtype=np.float64).reshape(3, 1)
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        rvec, _ = cv2.Rodrigues(R)

        if grid_heatmap:
            return self._overlay_grid(
                photo,
                grid_heatmap,
                rvec,
                T,
                K,
                dist,
                val_range,
                draw_colorbar,
            )

        projected, _ = cv2.projectPoints(pts_3d, rvec, T, K, dist)
        projected = projected.reshape(-1, 2)

        cam_pts = (R @ pts_3d.T + T).T
        depths = cam_pts[:, 2]

        valid = np.where(
            (depths > 0)
            & (projected[:, 0] >= 0)
            & (projected[:, 0] < w)
            & (projected[:, 1] >= 0)
            & (projected[:, 1] < h)
        )[0]
        valid = valid[np.argsort(-depths[valid])]

        mask_gray = np.zeros((h, w), dtype=np.uint8)
        mask_weight = np.zeros((h, w), dtype=np.float32)

        for idx in valid:
            u, v = int(projected[idx, 0]), int(projected[idx, 1])
            val = int(norm_scalars[idx])
            cv2.circle(mask_gray, (u, v), self.point_radius, val, -1)
            cv2.circle(mask_weight, (u, v), self.point_radius, 1.0, -1)

        if self.blur_size > 0:
            ksize = self.blur_size if self.blur_size % 2 == 1 else self.blur_size + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_CLOSE, kernel)
            mask_gray = cv2.GaussianBlur(mask_gray, (ksize, ksize), 0)
            mask_weight = cv2.GaussianBlur(mask_weight, (ksize, ksize), 0)
            mask_weight = np.clip(mask_weight, 0.0, 1.0)

        heatmap_color = cv2.applyColorMap(mask_gray, cv2.COLORMAP_JET)
        weight_3d = np.repeat(mask_weight[:, :, np.newaxis], 3, axis=2) * self.alpha
        blended = (
            photo.astype(np.float32) * (1.0 - weight_3d)
            + heatmap_color.astype(np.float32) * weight_3d
        ).astype(np.uint8)

        if draw_colorbar:
            blended = self._draw_colorbar(blended, min_v, max_v)

        meta = {
            'value_min': min_v,
            'value_max': max_v,
            'projected_point_count': int(len(valid)),
            'total_point_count': int(len(pts_3d)),
            'alpha': self.alpha,
            'unit': 'mm',
        }
        return blended, heatmap_color, meta

    def _overlay_grid(
        self,
        photo,
        grid_heatmap,
        rvec,
        translation,
        camera_matrix,
        distortion,
        val_range,
        draw_colorbar,
    ):
        """将同一份 3D 网格逐格透视投影，避免点扩散造成形变不真实。"""
        vertices = np.asarray(grid_heatmap.get('vertices') or [], dtype=np.float32)
        colors = np.asarray(grid_heatmap.get('colors') or [], dtype=np.float32)
        triangles = np.asarray(grid_heatmap.get('triangles') or [], dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
            raise ValueError('网格热力图缺少有效 3D 顶点')
        if colors.shape != vertices.shape:
            raise ValueError('网格热力图颜色与顶点数量不一致')

        h, w = photo.shape[:2]
        projected, _ = cv2.projectPoints(
            np.ascontiguousarray(vertices),
            rvec,
            translation,
            camera_matrix,
            distortion,
        )
        projected = projected.reshape(-1, 2)
        rotation, _ = cv2.Rodrigues(rvec)
        depths = (rotation @ vertices.T + translation).T[:, 2]

        heatmap_color = np.zeros((h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)
        cell_count = len(vertices) // 4
        visible_cells = []
        for cell_index in range(cell_count):
            start = cell_index * 4
            stop = start + 4
            cell_depths = depths[start:stop]
            if len(cell_depths) != 4 or np.any(cell_depths <= 0):
                continue
            polygon = projected[start:stop]
            if not np.all(np.isfinite(polygon)):
                continue
            if (
                np.max(polygon[:, 0]) < 0
                or np.min(polygon[:, 0]) >= w
                or np.max(polygon[:, 1]) < 0
                or np.min(polygon[:, 1]) >= h
            ):
                continue
            visible_cells.append((float(np.mean(cell_depths)), start, polygon))

        # 先画远处网格，近处网格覆盖，保持正确的透视遮挡顺序。
        visible_cells.sort(key=lambda item: item[0], reverse=True)
        for _, start, polygon in visible_cells:
            polygon_i32 = np.rint(polygon).astype(np.int32)
            rgb = np.clip(colors[start] * 255.0, 0, 255).astype(np.uint8)
            bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
            cv2.fillConvexPoly(heatmap_color, polygon_i32, bgr, lineType=cv2.LINE_AA)
            cv2.fillConvexPoly(mask, polygon_i32, 255, lineType=cv2.LINE_AA)

        if not visible_cells:
            raise ValueError('所选立面不在照片视野内，请检查相机位姿')

        weight = (mask.astype(np.float32) / 255.0 * self.alpha)[:, :, np.newaxis]
        blended = (
            photo.astype(np.float32) * (1.0 - weight)
            + heatmap_color.astype(np.float32) * weight
        ).astype(np.uint8)

        min_v, max_v = val_range if val_range is not None else (
            float(grid_heatmap.get('min_deviation_m', 0.0)) * 1000.0,
            float(grid_heatmap.get('max_deviation_m', 0.0)) * 1000.0,
        )
        if draw_colorbar:
            blended = self._draw_colorbar(blended, float(min_v), float(max_v))

        meta = {
            'value_min': float(min_v),
            'value_max': float(max_v),
            'projected_point_count': int(np.count_nonzero(depths > 0)),
            'total_point_count': int(len(vertices)),
            'projected_cell_count': int(len(visible_cells)),
            'total_cell_count': int(cell_count),
            'alpha': self.alpha,
            'unit': 'mm',
            'projection_mode': 'perspective_grid',
            'triangle_count': int(len(triangles) // 3),
        }
        return blended, heatmap_color, meta

    def _draw_colorbar(self, img, min_v, max_v):
        h, w = img.shape[:2]
        bar_w, bar_h = 20, max(60, int(h * 0.3))
        x_start = max(10, w - 60)
        y_start = int(h * 0.35)

        gradient = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(-1, 1)
        gradient_img = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
        gradient_img = cv2.resize(gradient_img, (bar_w, bar_h))

        y_end = min(h, y_start + bar_h)
        bar_h = y_end - y_start
        gradient_img = cv2.resize(gradient_img, (bar_w, bar_h))
        img[y_start:y_end, x_start:x_start + bar_w] = gradient_img
        cv2.rectangle(
            img,
            (x_start, y_start),
            (x_start + bar_w, y_end),
            (255, 255, 255),
            1,
        )
        cv2.putText(
            img,
            f'{max_v:.1f}mm',
            (max(5, x_start - 55), y_start + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f'{min_v:.1f}mm',
            (max(5, x_start - 55), y_end - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return img


def create_facade_heatmap_overlay(
    photo_path,
    facade,
    points,
    pose,
    alpha=0.5,
    point_radius=6,
    blur_size=15,
    val_range_mm=None,
    percentile=98.0,
    grid_heatmap=None,
):
    """
    步骤 5 服务入口：读取照片 + 相机位姿，将立面平整度热力图贴到 2D 照片上。

    :param pose: 含 rotation_matrix, translation_vector, camera_matrix, distortion_coefficients
    """
    photo_bgr = cv2.imread(str(photo_path))
    if photo_bgr is None:
        raise FileNotFoundError(f'无法读取照片: {photo_path}')

    if facade is None:
        raise ValueError('未指定立面')
    indices = np.asarray(facade.get('inlier_indices') or [], dtype=int)
    indices = indices[(indices >= 0) & (indices < len(points))]
    if indices.size < 3:
        raise ValueError('立面有效点数不足')

    pts = np.asarray(points, dtype=float)
    facade_points = pts[indices]
    plane = facade.get('plane_model')
    if plane is None:
        raise ValueError('立面缺少 plane_model')

    R = np.asarray(pose['rotation_matrix'], dtype=float)
    T = np.asarray(pose['translation_vector'], dtype=float).reshape(3, 1)
    K = np.asarray(pose['camera_matrix'], dtype=float)
    dist = pose.get('distortion_coefficients')
    camera_center = (-R.T @ T.reshape(3)).reshape(3)

    center = np.asarray(facade.get('center', np.mean(facade_points, axis=0)), dtype=float)
    oriented_plane = orient_plane_toward_camera(plane, center, camera_center)
    scalars_mm = compute_facade_deviation_scalars(facade_points, oriented_plane, unit='mm')

    if val_range_mm is None:
        limit = float(np.percentile(np.abs(scalars_mm), percentile))
        limit = max(limit, 4.0)
        val_range_mm = (-limit, limit)

    engine = FacadeHeatmapOverlay(
        alpha=alpha,
        point_radius=point_radius,
        blur_size=blur_size,
    )
    blended, heatmap_layer, meta = engine.overlay(
        photo_bgr,
        facade_points,
        scalars_mm,
        R,
        T,
        K,
        dist_coeffs=dist,
        val_range=val_range_mm,
        grid_heatmap=grid_heatmap,
    )

    return {
        'facade_id': int(facade.get('id', -1)),
        'image_base64': _encode_png_b64(blended),
        'heatmap_base64': _encode_png_b64(heatmap_layer),
        'image_mime': 'image/png',
        'width_px': int(blended.shape[1]),
        'height_px': int(blended.shape[0]),
        'overlay_meta': meta,
        'deviation_limit_mm': float(max(abs(val_range_mm[0]), abs(val_range_mm[1]))),
        'point_count': int(len(facade_points)),
    }


__all__ = [
    'FacadeHeatmapOverlay',
    'compute_facade_deviation_scalars',
    'create_facade_heatmap_overlay',
]
