"""将三维立面偏差热力图透视投影到二维照片。"""

from __future__ import annotations

import cv2
import numpy as np


def compute_facade_deviation_scalars(points_3d, plane_model, unit='mm'):
    points = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    plane = np.asarray(plane_model, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(plane[:3]))
    if norm < 1e-12:
        raise ValueError('立面拟合平面法向量无效')
    distances = (points @ plane[:3] + plane[3]) / norm
    if unit == 'mm':
        distances *= 1000.0
    return distances.astype(np.float32)


def orient_plane_toward_camera(plane_model, facade_center, camera_center):
    plane = np.asarray(plane_model, dtype=np.float64).reshape(4).copy()
    norm = float(np.linalg.norm(plane[:3]))
    if norm < 1e-12:
        raise ValueError('立面拟合平面法向量无效')
    plane /= norm
    toward_camera = (
        np.asarray(camera_center, dtype=np.float64).reshape(3)
        - np.asarray(facade_center, dtype=np.float64).reshape(3)
    )
    if float(np.dot(plane[:3], toward_camera)) < 0.0:
        plane *= -1.0
    return plane


def _draw_metre_grid(image, pixel_size):
    height, width = image.shape[:2]
    step = max(1, int(round(1.0 / float(pixel_size))))
    color = (175, 175, 175)
    for grid_x in range(0, width, step):
        cv2.line(
            image,
            (grid_x, 0),
            (grid_x, height - 1),
            color,
            1,
            cv2.LINE_AA,
        )
    for grid_y in range(height - 1, -1, -step):
        cv2.line(
            image,
            (0, grid_y),
            (width - 1, grid_y),
            color,
            1,
            cv2.LINE_AA,
        )


def build_plane_grid_heatmap(
    points,
    values_mm,
    plane,
    neutral_mm,
    vmin_mm,
    vmax_mm=None,
):
    """生成与侧栏网格图相同的立面热力栅格，并给出可贴到照片上的像素与四角。"""
    from algorithms.facade.heatmap_colors import (
        draw_signed_colorbar,
        signed_deviation_colors,
    )
    from algorithms.geometry import plane_axes

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    values = np.asarray(values_mm, dtype=np.float32).reshape(-1)
    normal = np.asarray(plane[:3], dtype=np.float64).reshape(3)
    facade_type = (
        'horizontal'
        if abs(float(normal[2])) > 0.85
        else 'vertical_facade'
    )
    u_axis, v_axis = plane_axes(normal, facade_type)
    origin = np.mean(points, axis=0)
    uv = np.column_stack(
        ((points - origin) @ u_axis, (points - origin) @ v_axis)
    )
    lower = np.min(uv, axis=0)
    upper = np.max(uv, axis=0)
    span = np.maximum(upper - lower, 0.1)
    pixel_size = max(
        0.02,
        float(span[0]) / 600.0,
        float(span[1]) / 1000.0,
    )
    width = max(24, int(np.ceil(span[0] / pixel_size)) + 1)
    height = max(24, int(np.ceil(span[1] / pixel_size)) + 1)
    x = np.clip(
        ((uv[:, 0] - lower[0]) / pixel_size).astype(np.int64),
        0,
        width - 1,
    )
    y = np.clip(
        height - 1
        - ((uv[:, 1] - lower[1]) / pixel_size).astype(np.int64),
        0,
        height - 1,
    )
    flat = y * width + x
    order = np.lexsort((np.abs(values), flat))
    sorted_flat = flat[order]
    keep = np.r_[
        sorted_flat[1:] != sorted_flat[:-1],
        True,
    ]
    selected = order[keep]
    if vmax_mm is None:
        bound = abs(float(vmin_mm))
        vmin_mm, vmax_mm = -bound, bound
    cell_colors = signed_deviation_colors(
        values[selected],
        threshold=neutral_mm,
        vmin=float(vmin_mm),
        vmax=float(vmax_mm),
    )

    margin = 36
    legend_width = 168
    canvas_rgb = np.full(
        (height + margin * 2, width + margin * 2 + legend_width, 3),
        245,
        dtype=np.uint8,
    )
    facade_rgb = np.full((height, width, 3), 225, dtype=np.uint8)
    facade_rgb.reshape(-1, 3)[flat[selected]] = np.clip(
        cell_colors * 255.0,
        0,
        255,
    ).astype(np.uint8)
    patch_mask = np.zeros((height, width), dtype=np.uint8)
    patch_mask.reshape(-1)[flat[selected]] = 255
    patch_rgb = facade_rgb.copy()
    _draw_metre_grid(patch_rgb, pixel_size)
    canvas_rgb[margin:margin + height, margin:margin + width] = patch_rgb
    legend_height = min(height, max(120, min(300, int(height * 0.42))))
    draw_signed_colorbar(
        canvas_rgb,
        margin + width + 18,
        margin + height - legend_height,
        22,
        legend_height,
        neutral_mm,
        vmin=float(vmin_mm),
        vmax=float(vmax_mm),
        text_color=(40, 40, 40),
        font_scale=0.62,
        thickness=1,
        output_bgr=False,
    )
    u0, v0 = float(lower[0]), float(lower[1])
    u1 = u0 + (width - 1) * float(pixel_size)
    v1 = v0 + (height - 1) * float(pixel_size)
    corners_3d = np.vstack(
        (
            origin + u0 * u_axis + v0 * v_axis,
            origin + u1 * u_axis + v0 * v_axis,
            origin + u1 * u_axis + v1 * v_axis,
            origin + u0 * u_axis + v1 * v_axis,
        )
    )
    return {
        'canvas_bgr': np.ascontiguousarray(canvas_rgb[:, :, ::-1]),
        'patch_bgr': np.ascontiguousarray(patch_rgb[:, :, ::-1]),
        'patch_mask': np.ascontiguousarray(patch_mask),
        'corners_3d': np.ascontiguousarray(corners_3d),
        'pixel_size': float(pixel_size),
    }


class FacadeHeatmapOverlay:
    """按相机内外参投影点状热力图，并与原照片 Alpha 融合。"""

    def __init__(self, alpha=0.58, point_radius=6, blur_size=15):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.point_radius = max(1, int(point_radius))
        self.blur_size = max(0, int(blur_size))

    @staticmethod
    def _distortion(dist_coeffs):
        if dist_coeffs is None:
            return np.zeros((5, 1), dtype=np.float64)
        values = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
        if len(values) >= 5:
            return values
        return np.pad(values, ((0, 5 - len(values)), (0, 0)))

    def overlay(
        self,
        photo_img,
        points_3d,
        scalars,
        rotation,
        translation,
        camera_matrix,
        *,
        dist_coeffs=None,
        val_range=None,
        threshold=None,
        draw_colorbar=True,
        boundary_points_3d=None,
        draw_boundary=False,
        border_color=(0, 0, 255),
        border_thickness=3,
    ):
        photo = np.asarray(photo_img, dtype=np.uint8)
        if photo.ndim == 2:
            photo = cv2.cvtColor(photo, cv2.COLOR_GRAY2BGR)
        if photo.ndim != 3 or photo.shape[2] != 3:
            raise ValueError('照片图像格式无效')

        points = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
        values = np.asarray(scalars, dtype=np.float32).reshape(-1)
        if len(points) != len(values) or len(points) == 0:
            raise ValueError('立面点与热力值数量不一致')
        rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        translation = np.asarray(translation, dtype=np.float64).reshape(3, 1)
        camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        rvec, _ = cv2.Rodrigues(rotation)
        projected, _ = cv2.projectPoints(
            np.ascontiguousarray(points),
            rvec,
            translation,
            camera_matrix,
            self._distortion(dist_coeffs),
        )
        pixels = projected.reshape(-1, 2)
        depths = (rotation @ points.T + translation).T[:, 2]
        height, width = photo.shape[:2]
        valid = (
            np.isfinite(pixels).all(axis=1)
            & np.isfinite(values)
            & (depths > 1e-6)
            & (pixels[:, 0] >= -self.point_radius)
            & (pixels[:, 0] < width + self.point_radius)
            & (pixels[:, 1] >= -self.point_radius)
            & (pixels[:, 1] < height + self.point_radius)
        )
        if not np.any(valid):
            raise ValueError('所选立面不在照片视野内，请检查匹配矩阵')

        from algorithms.facade.heatmap_colors import (
            GRAY_BGR,
            compute_heatmap_scale,
            signed_deviation_colors_bgr,
        )

        if val_range is None:
            scale_threshold, min_value, max_value = compute_heatmap_scale(
                values[valid]
            )
        else:
            min_value, max_value = (float(val_range[0]), float(val_range[1]))
            scale_threshold, min_value, max_value = compute_heatmap_scale(
                (min_value, max_value),
            )
        if threshold is not None:
            scale_threshold = float(threshold)
        point_colors = signed_deviation_colors_bgr(
            values,
            scale_threshold,
            min_value,
            max_value,
        )

        heatmap = np.full_like(photo, GRAY_BGR)
        mask = np.zeros((height, width), dtype=np.uint8)
        valid_indices = np.flatnonzero(valid)
        if len(valid_indices) > 200_000:
            stride = int(np.ceil(len(valid_indices) / 200_000))
            valid_indices = valid_indices[::stride]
        # 先画远点、后画近点，使同一像素由更靠近相机的立面点覆盖。
        order = valid_indices[np.argsort(depths[valid_indices])[::-1]]
        for index in order:
            center = tuple(np.rint(pixels[index]).astype(int))
            color = tuple(int(channel) for channel in point_colors[index])
            cv2.circle(
                heatmap,
                center,
                self.point_radius,
                color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
            cv2.circle(
                mask,
                center,
                self.point_radius,
                255,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

        if self.blur_size > 1:
            kernel = self.blur_size | 1
            heatmap = cv2.GaussianBlur(heatmap, (kernel, kernel), 0)
            mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)

        boundary_pixels = None
        if boundary_points_3d is not None:
            boundary = np.asarray(
                boundary_points_3d,
                dtype=np.float64,
            ).reshape(-1, 3)
            if len(boundary) >= 4:
                projected_boundary, _ = cv2.projectPoints(
                    np.ascontiguousarray(boundary),
                    rvec,
                    translation,
                    camera_matrix,
                    self._distortion(dist_coeffs),
                )
                candidate = projected_boundary.reshape(-1, 2)
                boundary_depths = (
                    rotation @ boundary.T + translation
                ).T[:, 2]
                visible = (
                    np.isfinite(candidate).all(axis=1)
                    & np.isfinite(boundary_depths)
                    & (boundary_depths > 1e-6)
                )
                if int(visible.sum()) >= 2:
                    boundary_pixels = candidate
                    # 红框只标出检测点 UV 包围盒，不再用四边形裁掉两端热力。
        weight = (
            mask.astype(np.float32)[:, :, np.newaxis]
            / 255.0
            * self.alpha
        )
        blended = np.clip(
            photo.astype(np.float32) * (1.0 - weight)
            + heatmap.astype(np.float32) * weight,
            0,
            255,
        ).astype(np.uint8)
        if draw_boundary and boundary_pixels is not None:
            cv2.polylines(
                blended,
                [np.rint(boundary_pixels).astype(np.int32)],
                True,
                tuple(int(channel) for channel in border_color),
                max(1, int(border_thickness)),
                cv2.LINE_AA,
            )
        if draw_colorbar:
            self._draw_colorbar(
                blended,
                min_value,
                max_value,
                scale_threshold,
            )
        meta = {
            'projected_point_count': int(len(order)),
            'visible_point_count': int(np.sum(valid)),
            'total_point_count': int(len(points)),
            'alpha': self.alpha,
            'unit': 'mm',
        }
        if boundary_pixels is not None:
            meta['facade_boundary_pixels'] = boundary_pixels.tolist()
        return blended, heatmap, meta

    def overlay_grid(
        self,
        photo_img,
        grid_bgr,
        grid_mask,
        corners_3d,
        rotation,
        translation,
        camera_matrix,
        *,
        dist_coeffs=None,
        draw_colorbar=False,
        val_range=None,
        threshold=None,
    ):
        """把侧栏网格热力图原图像素透视贴到照片或点云映射图上。"""
        photo = np.asarray(photo_img, dtype=np.uint8)
        if photo.ndim == 2:
            photo = cv2.cvtColor(photo, cv2.COLOR_GRAY2BGR)
        if photo.ndim != 3 or photo.shape[2] != 3:
            raise ValueError('照片图像格式无效')
        grid = np.ascontiguousarray(grid_bgr, dtype=np.uint8)
        if grid.ndim != 3 or grid.shape[2] != 3:
            raise ValueError('网格热力图格式无效')
        mask = np.asarray(grid_mask, dtype=np.uint8)
        if mask.shape[:2] != grid.shape[:2]:
            raise ValueError('网格热力图与有效像素掩膜尺寸不一致')
        corners = np.asarray(corners_3d, dtype=np.float64).reshape(4, 3)
        rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        translation = np.asarray(translation, dtype=np.float64).reshape(3, 1)
        camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        depths = (rotation @ corners.T + translation).T[:, 2]
        if np.any(~np.isfinite(depths)) or np.any(depths <= 1e-6):
            raise ValueError('所选立面不在照片视野内，请检查匹配矩阵')
        rvec, _ = cv2.Rodrigues(rotation)
        projected, _ = cv2.projectPoints(
            np.ascontiguousarray(corners),
            rvec,
            translation,
            camera_matrix,
            self._distortion(dist_coeffs),
        )
        destination = projected.reshape(4, 2).astype(np.float32)
        if not np.isfinite(destination).all():
            raise ValueError('网格热力图投影坐标无效')
        height, width = grid.shape[:2]
        source = np.asarray(
            (
                (0.0, height - 1.0),
                (width - 1.0, height - 1.0),
                (width - 1.0, 0.0),
                (0.0, 0.0),
            ),
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(source, destination)
        photo_h, photo_w = photo.shape[:2]
        warped = cv2.warpPerspective(
            grid,
            matrix,
            (photo_w, photo_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        warped_mask = cv2.warpPerspective(
            mask,
            matrix,
            (photo_w, photo_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        weight = (
            warped_mask.astype(np.float32)[:, :, np.newaxis]
            / 255.0
            * self.alpha
        )
        blended = np.clip(
            photo.astype(np.float32) * (1.0 - weight)
            + warped.astype(np.float32) * weight,
            0,
            255,
        ).astype(np.uint8)
        if draw_colorbar and val_range is not None and threshold is not None:
            self._draw_colorbar(
                blended,
                float(val_range[0]),
                float(val_range[1]),
                float(threshold),
            )
        meta = {
            'projected_point_count': int(np.count_nonzero(mask)),
            'visible_point_count': int(np.count_nonzero(warped_mask)),
            'total_point_count': int(np.count_nonzero(mask)),
            'alpha': self.alpha,
            'unit': 'mm',
            'source': 'grid',
        }
        return blended, warped, meta

    @classmethod
    def _draw_colorbar(cls, image, min_value, max_value, threshold):
        from algorithms.facade.heatmap_colors import draw_signed_colorbar

        height, width = image.shape[:2]
        bar_height = max(200, min(380, int(height * 0.48)))
        bar_width = 22
        panel_width = 148
        x0 = max(8, width - panel_width - 10)
        y0 = max(24, height - bar_height - 28)
        y1 = min(height - 8, y0 + bar_height + 16)
        x1 = width - 6
        panel = image[y0 - 16:y1, x0 - 8:x1]
        if panel.size:
            image[y0 - 16:y1, x0 - 8:x1] = cv2.addWeighted(
                panel,
                0.22,
                np.full_like(panel, 245),
                0.78,
                0,
            )
        draw_signed_colorbar(
            image,
            x0,
            y0,
            bar_width,
            bar_height,
            threshold,
            vmin=float(min_value),
            vmax=float(max_value),
            text_color=(40, 40, 40),
            font_scale=0.68,
            thickness=2,
            output_bgr=True,
        )
