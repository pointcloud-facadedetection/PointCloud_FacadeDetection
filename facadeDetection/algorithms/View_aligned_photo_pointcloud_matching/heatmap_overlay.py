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
        draw_colorbar=True,
        boundary_points_3d=None,
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

        if val_range is None:
            limit = max(4.0, float(np.percentile(np.abs(values[valid]), 98.0)))
            min_value, max_value = -limit, limit
        else:
            min_value, max_value = (float(val_range[0]), float(val_range[1]))
        point_colors = self._signed_colors_bgr(
            values,
            min_value,
            max_value,
        )

        heatmap = np.zeros_like(photo)
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
                if (
                    np.isfinite(candidate).all()
                    and np.all(boundary_depths > 1e-6)
                ):
                    boundary_pixels = candidate
                    polygon = np.rint(candidate).astype(np.int32)
                    facade_mask = np.zeros_like(mask)
                    cv2.fillConvexPoly(
                        facade_mask,
                        polygon,
                        255,
                        lineType=cv2.LINE_AA,
                    )
                    mask = cv2.bitwise_and(mask, facade_mask)
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
        if boundary_pixels is not None:
            cv2.polylines(
                blended,
                [np.rint(boundary_pixels).astype(np.int32)],
                True,
                tuple(int(channel) for channel in border_color),
                max(1, int(border_thickness)),
                cv2.LINE_AA,
            )
        if draw_colorbar:
            self._draw_colorbar(blended, min_value, max_value)
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

    @staticmethod
    def _signed_colors_bgr(values, min_value, max_value):
        """凹陷为蓝色、平整为灰色、凸起为红色。"""
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        limit = max(abs(float(min_value)), abs(float(max_value)), 1e-6)
        neutral = min(2.0, limit * 0.25)
        rgb = np.tile(
            np.asarray((0.50, 0.52, 0.52), dtype=np.float32),
            (len(values), 1),
        )
        negative = values < -neutral
        positive = values > neutral
        if np.any(negative):
            t = np.clip(
                (-values[negative] - neutral) / max(limit - neutral, 1e-6),
                0.0,
                1.0,
            )
            rgb[negative] = (
                np.asarray((0.25, 0.45, 1.0))[None, :] * (1.0 - t[:, None])
                + np.asarray((0.00, 0.00, 0.82))[None, :] * t[:, None]
            )
        if np.any(positive):
            t = np.clip(
                (values[positive] - neutral) / max(limit - neutral, 1e-6),
                0.0,
                1.0,
            )
            rgb[positive] = (
                np.asarray((1.0, 0.38, 0.38))[None, :] * (1.0 - t[:, None])
                + np.asarray((0.85, 0.00, 0.00))[None, :] * t[:, None]
            )
        return np.clip(rgb[:, ::-1] * 255.0, 0, 255).astype(np.uint8)

    @classmethod
    def _draw_colorbar(cls, image, min_value, max_value):
        height, width = image.shape[:2]
        bar_height = max(80, min(260, int(height * 0.34)))
        bar_width = max(14, min(24, width // 40))
        x0 = max(8, width - bar_width - 48)
        y0 = max(8, height - bar_height - 26)
        gradient = np.linspace(
            max_value,
            min_value,
            bar_height,
            dtype=np.float32,
        )
        colors = cls._signed_colors_bgr(
            gradient,
            min_value,
            max_value,
        ).reshape(bar_height, 1, 3)
        colors = np.repeat(colors, bar_width, axis=1)
        image[y0:y0 + bar_height, x0:x0 + bar_width] = colors
        cv2.rectangle(
            image,
            (x0, y0),
            (x0 + bar_width, y0 + bar_height),
            (255, 255, 255),
            1,
        )
        cv2.putText(
            image,
            f'{max_value:+.1f}',
            (x0 + bar_width + 4, y0 + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f'{min_value:+.1f}',
            (x0 + bar_width + 4, y0 + bar_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
