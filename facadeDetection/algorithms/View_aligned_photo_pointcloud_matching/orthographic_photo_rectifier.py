"""基于横、竖双消失点的建筑立面近似正投影校正。"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .photo_loader import read_bgr_image


class PhotoPerspectiveRectifier:
    """将目标立面的两组 Manhattan 方向恢复为正交平行方向。"""

    def __init__(
        self,
        target_max_dim: int = 2048,
        correction_strength: float = 1.0,
        keep_original_aspect_ratio: bool = True,
    ):
        self.target_max_dim = int(target_max_dim)
        self.correction_strength = float(correction_strength)
        self.keep_original_aspect_ratio = bool(keep_original_aspect_ratio)
        self.last_method = 'unknown'
        self.last_info = {}
        self.last_src_pts = None
        self.last_dst_pts = None
        self.last_structure_points = None
        self.last_vp = None
        self.last_crop_box = None

    @staticmethod
    def _line_from_segment(segment):
        x1, y1, x2, y2 = segment
        line = np.cross([x1, y1, 1.0], [x2, y2, 1.0])
        norm = math.hypot(float(line[0]), float(line[1]))
        return line / max(norm, 1e-12)

    def _detect_lines(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        found = cv2.createLineSegmentDetector(
            cv2.LSD_REFINE_STD
        ).detect(gray)[0]
        if found is None:
            raise RuntimeError('未检测到建筑结构线。')
        height, width = gray.shape
        min_length = 0.025 * min(height, width)
        vertical, horizontal = [], []
        for raw in found[:, 0, :]:
            segment = np.asarray(raw, dtype=np.float64)
            dx, dy = segment[2] - segment[0], segment[3] - segment[1]
            length = float(math.hypot(dx, dy))
            if length < min_length:
                continue
            mid_x = 0.5 * (segment[0] + segment[2])
            mid_y = 0.5 * (segment[1] + segment[3])
            if not (
                0.08 * width <= mid_x <= 0.96 * width
                and 0.02 * height <= mid_y <= 0.98 * height
            ):
                continue
            angle = math.degrees(math.atan2(dy, dx)) % 180.0
            item = {
                'segment': segment,
                'line': self._line_from_segment(segment),
                'length': length,
            }
            if abs(angle - 90.0) <= 42.0:
                vertical.append(item)
            if min(angle, 180.0 - angle) <= 38.0:
                horizontal.append(item)
        if len(vertical) < 2 or len(horizontal) < 2:
            raise RuntimeError('目标立面的横向或竖向结构线不足。')
        return vertical, horizontal

    @staticmethod
    def _vp_residuals(vp, items):
        segments = np.asarray([item['segment'] for item in items])
        directions = segments[:, 2:4] - segments[:, 0:2]
        directions /= np.maximum(
            np.linalg.norm(directions, axis=1, keepdims=True), 1e-12
        )
        if abs(float(vp[2])) > 1e-10:
            point = vp[:2] / vp[2]
            mids = 0.5 * (segments[:, 0:2] + segments[:, 2:4])
            rays = point - mids
            rays /= np.maximum(
                np.linalg.norm(rays, axis=1, keepdims=True), 1e-12
            )
        else:
            ray = vp[:2] / max(float(np.linalg.norm(vp[:2])), 1e-12)
            rays = np.repeat(ray.reshape(1, 2), len(items), axis=0)
        cosine = np.abs(np.sum(directions * rays, axis=1))
        return np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))

    def _estimate_vp(self, items, seed):
        rng = np.random.default_rng(seed)
        lengths = np.asarray([item['length'] for item in items])
        probabilities = lengths / lengths.sum()
        lines = np.asarray([item['line'] for item in items])
        best = None
        best_score = -1.0
        for _ in range(4000):
            i, j = rng.choice(len(items), 2, replace=False, p=probabilities)
            vp = np.cross(lines[i], lines[j])
            norm = float(np.linalg.norm(vp))
            if norm <= 1e-12:
                continue
            vp /= norm
            residuals = self._vp_residuals(vp, items)
            inliers = residuals <= 2.8
            score = float(lengths[inliers].sum())
            if score > best_score:
                best = inliers
                best_score = score
        if best is None or int(np.sum(best)) < 2:
            raise RuntimeError('无法稳定估计建筑消失点。')
        weighted = lines[best] * np.sqrt(lengths[best])[:, None]
        _u, _s, vh = np.linalg.svd(weighted, full_matrices=False)
        vp = vh[-1]
        if abs(float(vp[2])) > 1e-10:
            vp /= vp[2]
        else:
            vp /= max(float(np.linalg.norm(vp[:2])), 1e-12)
        residuals = self._vp_residuals(vp, items)
        inliers = residuals <= 2.8
        return vp, inliers, residuals

    @staticmethod
    def _transform(matrix, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        hom = np.column_stack([points, np.ones(len(points))])
        mapped = (matrix @ hom.T).T
        result = np.full((len(points), 2), np.nan)
        valid = np.abs(mapped[:, 2]) > 1e-10
        result[valid] = mapped[valid, :2] / mapped[valid, 2:3]
        return result

    def _dominant_direction(self, projective, items):
        directions, weights = [], []
        for item in items:
            segment = item['segment']
            mapped = self._transform(
                projective,
                [[segment[0], segment[1]], [segment[2], segment[3]]],
            )
            vector = mapped[1] - mapped[0]
            norm = float(np.linalg.norm(vector))
            if not np.isfinite(norm) or norm <= 1e-9:
                continue
            vector /= norm
            if vector[0] < 0:
                vector = -vector
            directions.append(vector)
            weights.append(item['length'])
        if not directions:
            raise RuntimeError('校正后的结构方向退化。')
        direction = np.average(directions, axis=0, weights=weights)
        return direction / max(float(np.linalg.norm(direction)), 1e-12)

    def _rectification(self, horizontal, vertical, vp_h, vp_v):
        line_at_infinity = np.cross(vp_h, vp_v)
        if abs(float(line_at_infinity[2])) < 1e-10:
            raise RuntimeError('立面消失线不稳定。')
        line_at_infinity /= line_at_infinity[2]
        projective = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            line_at_infinity,
        ])
        dir_h = self._dominant_direction(projective, horizontal)
        dir_v = self._dominant_direction(projective, vertical)
        if dir_v[1] < 0:
            dir_v = -dir_v
        basis = np.column_stack([dir_h, dir_v])
        if abs(float(np.linalg.det(basis))) < 0.08:
            raise RuntimeError('横竖方向无法恢复为稳定的正交坐标。')
        affine = np.linalg.inv(basis)
        metric = np.array([
            [affine[0, 0], affine[0, 1], 0.0],
            [affine[1, 0], affine[1, 1], 0.0],
            [0.0, 0.0, 1.0],
        ])
        return metric @ projective

    def _warp_full(self, image, matrix):
        height, width = image.shape[:2]
        corners = np.array([
            [0, 0], [width - 1, 0],
            [width - 1, height - 1], [0, height - 1],
        ])
        mapped = self._transform(matrix, corners)
        if not np.all(np.isfinite(mapped)):
            raise RuntimeError('正投影后的图像边界无效。')
        minimum, maximum = mapped.min(axis=0), mapped.max(axis=0)
        span = maximum - minimum
        if np.any(span <= 1) or max(span) > 30.0 * max(width, height):
            raise RuntimeError('正投影变换过强，请先裁剪到目标立面。')
        scale = min(1.0, self.target_max_dim / max(span))
        translate = np.array([
            [scale, 0.0, -scale * minimum[0]],
            [0.0, scale, -scale * minimum[1]],
            [0.0, 0.0, 1.0],
        ])
        full = translate @ matrix
        out_w = max(1, int(math.ceil(span[0] * scale)))
        out_h = max(1, int(math.ceil(span[1] * scale)))
        warped = cv2.warpPerspective(
            image,
            full,
            (out_w, out_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return warped, full

    def rectify(self, photo_img):
        image = np.asarray(photo_img)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image = image[:, :, :3].copy()
        try:
            vertical, horizontal = self._detect_lines(image)
            vp_v, inliers_v, residuals_v = self._estimate_vp(vertical, 0)
            vp_h, inliers_h, residuals_h = self._estimate_vp(horizontal, 1)
            selected_v = [x for x, keep in zip(vertical, inliers_v) if keep]
            selected_h = [x for x, keep in zip(horizontal, inliers_h) if keep]
            matrix = self._rectification(
                selected_h, selected_v, vp_h, vp_v
            )
            rectified, matrix = self._warp_full(image, matrix)
            self.last_method = 'facade_frontoparallel_dual_vp'
            self.last_vp = vp_v.copy()
            self.last_structure_points = np.asarray([
                point
                for item in selected_h + selected_v
                for point in (
                    item['segment'][:2],
                    item['segment'][2:],
                )
            ], dtype=np.float32)
            self.last_info = {
                'projection': 'fronto_parallel',
                'horizontal_vp': vp_h.tolist(),
                'vertical_vp': vp_v.tolist(),
                'horizontal_inlier_count': len(selected_h),
                'vertical_inlier_count': len(selected_v),
                'horizontal_residual_median_deg': float(
                    np.median(residuals_h[inliers_h])
                ),
                'vertical_residual_median_deg': float(
                    np.median(residuals_v[inliers_v])
                ),
            }
        except Exception as exc:
            rectified = image.copy()
            matrix = np.eye(3)
            self.last_method = 'identity_fallback'
            self.last_info = {'fallback_reason': str(exc)}
        out_h, out_w = rectified.shape[:2]
        self.last_src_pts = np.float32([
            [0, 0], [image.shape[1] - 1, 0],
            [image.shape[1] - 1, image.shape[0] - 1],
            [0, image.shape[0] - 1],
        ])
        self.last_dst_pts = np.float32([
            [0, 0], [out_w - 1, 0],
            [out_w - 1, out_h - 1], [0, out_h - 1],
        ])
        self.last_crop_box = (0, 0, out_w, out_h)
        return rectified, matrix, np.linalg.inv(matrix)


def rectify_photo_perspective(
    photo_path: str | Path,
    *,
    target_max_dim: int = 2048,
    correction_strength: float = 1.0,
    keep_original_aspect_ratio: bool = True,
):
    path = Path(photo_path).expanduser().resolve()
    image = read_bgr_image(path)
    rectifier = PhotoPerspectiveRectifier(
        target_max_dim=target_max_dim,
        correction_strength=correction_strength,
        keep_original_aspect_ratio=keep_original_aspect_ratio,
    )
    rectified, matrix, inverse = rectifier.rectify(image)
    result = {
        'photo_path': str(path),
        'rectified_bgr': rectified,
        'H': matrix,
        'H_inv': inverse,
        'method': rectifier.last_method,
        'original_size': (image.shape[1], image.shape[0]),
        'rectified_size': (rectified.shape[1], rectified.shape[0]),
        'content_mean': float(rectified.mean()),
        'rectify_info': rectifier.last_info,
        'src_quad': rectifier.last_src_pts.tolist(),
        'dst_quad': rectifier.last_dst_pts.tolist(),
        'crop_box': rectifier.last_crop_box,
    }
    if rectifier.last_structure_points is not None:
        result['structure_points'] = rectifier.last_structure_points.tolist()
    if rectifier.last_vp is not None:
        result['vertical_vp'] = rectifier.last_vp.tolist()
    return result


__all__ = ['PhotoPerspectiveRectifier', 'rectify_photo_perspective']
