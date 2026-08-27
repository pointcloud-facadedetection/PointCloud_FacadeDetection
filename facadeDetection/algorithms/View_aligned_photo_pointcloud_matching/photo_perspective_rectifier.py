"""2D 现场照片竖直视角矫正。

该实现来自 test/gpt_zheng_whole_v4.py 的核心流程：只校正竖直消失点，
让建筑竖线站直，同时尽量保留左右立面的原始观察关系。
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .photo_loader import read_bgr_image


class PhotoPerspectiveRectifier:
    """基于竖直消失点的整栋建筑照片摆正器。"""

    MIN_LINE_LENGTH_RATIO = 0.025
    MAX_VERTICAL_ANGLE_DEVIATION_DEG = 45.0
    ROI_X_MIN = 0.10
    ROI_X_MAX = 0.95
    ROI_Y_MIN = 0.00
    ROI_Y_MAX = 1.00
    VP_RANSAC_ITERATIONS = 5000
    VP_RANSAC_ANGLE_THRESHOLD_DEG = 2.2
    BUILDING_CLUSTER_GAP_RATIO = 0.09
    ANCHOR_Y_MIN_RATIO = 0.45
    ANCHOR_Y_MAX_RATIO = 0.72
    STRUCTURE_LOW_Q = 0.01
    STRUCTURE_HIGH_Q = 0.99
    CROP_MARGIN_X_RATIO = 0.08
    CROP_MARGIN_Y_RATIO = 0.035
    MASK_LEFT_QUANTILE = 0.94
    MASK_RIGHT_QUANTILE = 0.06
    CROP_INSET_PX = 2
    ASPECT_SEARCH_STEPS = 61
    ASPECT_CENTER_PENALTY = 0.03

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
        self.last_src_pts: np.ndarray | None = None
        self.last_dst_pts: np.ndarray | None = None
        self.last_structure_points: np.ndarray | None = None
        self.last_vp: np.ndarray | None = None
        self.last_crop_box: tuple[int, int, int, int] | None = None
        self.last_info: dict = {}

    @staticmethod
    def _normalize_line(line: np.ndarray) -> np.ndarray:
        line = np.asarray(line, dtype=np.float64)
        norm = math.hypot(float(line[0]), float(line[1]))
        if norm < 1e-12:
            return line
        return line / norm

    @classmethod
    def _segment_to_line(cls, x1, y1, x2, y2) -> np.ndarray:
        p1 = np.array([x1, y1, 1.0], dtype=np.float64)
        p2 = np.array([x2, y2, 1.0], dtype=np.float64)
        return cls._normalize_line(np.cross(p1, p2))

    @staticmethod
    def _line_intersection(line_a: np.ndarray, line_b: np.ndarray) -> np.ndarray | None:
        point = np.cross(line_a, line_b)
        if abs(float(point[2])) < 1e-12:
            return None
        point = point / point[2]
        if not np.all(np.isfinite(point)):
            return None
        return point

    @staticmethod
    def _transform_points(matrix: np.ndarray, points) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(pts) == 0:
            return np.empty((0, 2), dtype=np.float64)
        hom = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
        out = (np.asarray(matrix, dtype=np.float64) @ hom.T).T
        valid = np.abs(out[:, 2]) > 1e-12
        result = np.full((len(pts), 2), np.nan, dtype=np.float64)
        result[valid] = out[valid, :2] / out[valid, 2:3]
        return result

    @staticmethod
    def _weighted_median(values, weights) -> float:
        values = np.asarray(values, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        order = np.argsort(values)
        values = values[order]
        weights = weights[order]
        cumsum = np.cumsum(weights)
        if cumsum[-1] <= 0:
            return float(np.median(values))
        idx = int(np.searchsorted(cumsum, 0.5 * cumsum[-1]))
        return float(values[min(max(idx, 0), len(values) - 1)])

    def _detect_vertical_lines(self, image: np.ndarray) -> list[dict]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        detected = detector.detect(gray)[0]
        if detected is None:
            raise RuntimeError('未检测到直线段，无法自动摆正照片。')

        h, w = image.shape[:2]
        min_length = min(h, w) * self.MIN_LINE_LENGTH_RATIO
        rx0 = self.ROI_X_MIN * w
        rx1 = self.ROI_X_MAX * w
        ry0 = self.ROI_Y_MIN * h
        ry1 = self.ROI_Y_MAX * h

        vertical = []
        for seg in detected[:, 0, :]:
            x1, y1, x2, y2 = map(float, seg)
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length < min_length:
                continue
            mid_x = 0.5 * (x1 + x2)
            mid_y = 0.5 * (y1 + y2)
            if not (rx0 <= mid_x <= rx1 and ry0 <= mid_y <= ry1):
                continue
            angle = math.degrees(math.atan2(dy, dx)) % 180.0
            if abs(angle - 90.0) > self.MAX_VERTICAL_ANGLE_DEVIATION_DEG:
                continue
            vertical.append({
                'segment': np.array([x1, y1, x2, y2], dtype=np.float64),
                'line': self._segment_to_line(x1, y1, x2, y2),
                'length': float(length),
                'angle': float(angle),
                'mid_x': float(mid_x),
                'mid_y': float(mid_y),
            })

        if len(vertical) < 2:
            raise RuntimeError('竖向结构线太少，无法稳定估计竖直消失点。')
        return vertical

    def _estimate_vertical_vp(self, lines: list[dict], image_shape, seed=0):
        h, w = image_shape[:2]
        count = len(lines)
        rng = np.random.default_rng(seed)
        lengths = np.array([item['length'] for item in lines], dtype=np.float64)
        probs = lengths / max(float(lengths.sum()), 1e-12)
        line_matrix = np.array([item['line'] for item in lines], dtype=np.float64)
        segments = np.array([item['segment'] for item in lines], dtype=np.float64)
        mids = np.column_stack([
            0.5 * (segments[:, 0] + segments[:, 2]),
            0.5 * (segments[:, 1] + segments[:, 3]),
        ])
        dirs = np.column_stack([
            segments[:, 2] - segments[:, 0],
            segments[:, 3] - segments[:, 1],
        ])
        dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
        cos_threshold = math.cos(math.radians(self.VP_RANSAC_ANGLE_THRESHOLD_DEG))

        best_score = -1.0
        best_inliers = None
        best_vp = None
        for _ in range(self.VP_RANSAC_ITERATIONS):
            i, j = rng.choice(count, 2, replace=False, p=probs)
            vp = self._line_intersection(line_matrix[i], line_matrix[j])
            if vp is None:
                continue
            if abs(float(vp[0])) > 300 * w or abs(float(vp[1])) > 300 * h:
                continue
            delta = vp[:2][None, :] - mids
            dist = np.linalg.norm(delta, axis=1)
            good = dist > 1e-12
            cosang = np.zeros(count, dtype=np.float64)
            cosang[good] = np.abs(np.sum(dirs[good] * delta[good], axis=1) / dist[good])
            inliers = cosang >= cos_threshold
            score = float(lengths[inliers].sum())
            if score > best_score:
                best_score = score
                best_inliers = inliers
                best_vp = vp

        if best_inliers is None or int(best_inliers.sum()) < 2:
            raise RuntimeError('无法稳定估计竖直消失点。')

        selected_lines = line_matrix[best_inliers]
        weights = np.sqrt(lengths[best_inliers])[:, None]
        _u, _s, vt = np.linalg.svd(selected_lines * weights, full_matrices=False)
        vp = vt[-1]
        if abs(float(vp[2])) < 1e-12:
            vp = best_vp
        else:
            vp = vp / vp[2]

        delta = vp[:2][None, :] - mids
        dist = np.linalg.norm(delta, axis=1)
        good = dist > 1e-12
        cosang = np.zeros(count, dtype=np.float64)
        cosang[good] = np.abs(np.sum(dirs[good] * delta[good], axis=1) / dist[good])
        residuals = np.degrees(np.arccos(np.clip(cosang, 0.0, 1.0)))
        inliers = residuals <= self.VP_RANSAC_ANGLE_THRESHOLD_DEG
        return vp, inliers, residuals

    def _select_main_building_cluster(self, lines: list[dict], inliers, image_shape):
        _h, w = image_shape[:2]
        candidates = [item for item, flag in zip(lines, inliers) if flag]
        if len(candidates) < 2:
            raise RuntimeError('竖直内点太少，无法选择主建筑结构线。')

        candidates = sorted(candidates, key=lambda item: item['mid_x'])
        gap_threshold = self.BUILDING_CLUSTER_GAP_RATIO * w
        clusters = []
        current = [candidates[0]]
        for item in candidates[1:]:
            if item['mid_x'] - current[-1]['mid_x'] > gap_threshold:
                clusters.append(current)
                current = [item]
            else:
                current.append(item)
        clusters.append(current)

        center_x = 0.5 * (w - 1)
        best = clusters[0]
        best_score = -1.0
        for cluster in clusters:
            total_length = sum(item['length'] for item in cluster)
            xs = np.array([item['mid_x'] for item in cluster], dtype=np.float64)
            cmean = float(np.average(xs, weights=[item['length'] for item in cluster]))
            center_bonus = 0.75 + 0.25 * math.exp(-((cmean - center_x) / (0.35 * w)) ** 2)
            score = float(total_length * center_bonus)
            if score > best_score:
                best_score = score
                best = cluster
        return best, clusters

    def _build_minimal_vertical_homography(self, image_shape, vp_vertical, main_lines):
        h, w = image_shape[:2]
        cx = 0.5 * (w - 1)
        cy = 0.5 * (h - 1)
        dx = float(vp_vertical[0] - cx)
        dy = float(vp_vertical[1] - cy)
        if abs(dx) + abs(dy) < 1e-12:
            raise RuntimeError('竖直消失点过于接近图像中心，几何不稳定。')

        current_angle = math.atan2(dy, dx)
        target_angle = -math.pi / 2 if dy < 0 else math.pi / 2
        theta = target_angle - current_angle
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        t0 = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
        t1 = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]])
        roll = np.array([[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]])
        h_roll = t1 @ roll @ t0

        vp_rot = h_roll @ np.asarray(vp_vertical, dtype=np.float64)
        vp_rot = vp_rot / vp_rot[2]
        vp_y_centered = float(vp_rot[1] - cy)
        if abs(vp_y_centered) < 0.15 * h:
            raise RuntimeError('竖直消失点离图像中心太近，照片竖向透视不足以稳定校正。')

        mids = []
        weights = []
        for item in main_lines:
            x1, y1, x2, y2 = item['segment']
            mids.append([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
            weights.append(item['length'])
        mids_rot = self._transform_points(h_roll, mids)
        anchor_y = self._weighted_median(mids_rot[:, 1], weights)
        anchor_y = float(np.clip(
            anchor_y,
            self.ANCHOR_Y_MIN_RATIO * (h - 1),
            self.ANCHOR_Y_MAX_RATIO * (h - 1),
        ))

        strength = float(np.clip(self.correction_strength, 0.0, 1.0))
        p = -strength / vp_y_centered
        anchor_den = 1.0 + p * float(anchor_y - cy)
        h_proj = np.array([
            [anchor_den, 0.0, 0.0],
            [0.0, anchor_den, 0.0],
            [0.0, p, 1.0],
        ], dtype=np.float64)
        h_base = t1 @ h_proj @ t0 @ h_roll
        if abs(float(h_base[2, 2])) > 1e-12:
            h_base = h_base / h_base[2, 2]
        return h_base, {
            'roll_deg': math.degrees(theta),
            'anchor_y': anchor_y,
            'vp_after_roll': vp_rot.tolist(),
            'vp_y_centered': vp_y_centered,
        }

    def _make_full_warp(self, image: np.ndarray, h_base: np.ndarray):
        h, w = image.shape[:2]
        sample_count = 500
        xs = np.linspace(0, w - 1, sample_count)
        ys = np.linspace(0, h - 1, sample_count)
        border = np.vstack([
            np.column_stack([xs, np.zeros_like(xs)]),
            np.column_stack([xs, np.full_like(xs, h - 1)]),
            np.column_stack([np.zeros_like(ys), ys]),
            np.column_stack([np.full_like(ys, w - 1), ys]),
        ])
        warped_border = self._transform_points(h_base, border)
        warped_border = warped_border[np.all(np.isfinite(warped_border), axis=1)]
        if len(warped_border) == 0:
            raise RuntimeError('矫正后的图像边界无效。')

        min_xy = np.quantile(warped_border, 0.001, axis=0)
        max_xy = np.quantile(warped_border, 0.999, axis=0)
        span = max_xy - min_xy
        if np.any(span <= 1):
            raise RuntimeError('矫正后的输出范围退化。')
        max_side = self.target_max_dim if self.target_max_dim > 0 else max(span)
        scale = min(1.0, float(max_side) / float(max(span)))
        translate = np.array([
            [scale, 0.0, -scale * min_xy[0]],
            [0.0, scale, -scale * min_xy[1]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        h_full = translate @ h_base
        out_w = max(1, int(math.ceil(span[0] * scale)))
        out_h = max(1, int(math.ceil(span[1] * scale)))
        warped = cv2.warpPerspective(
            image,
            h_full,
            (out_w, out_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        src_mask = np.full((h, w), 255, dtype=np.uint8)
        mask = cv2.warpPerspective(
            src_mask,
            h_full,
            (out_w, out_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return warped, mask, h_full

    @staticmethod
    def _main_structure_points(main_lines: list[dict]) -> np.ndarray:
        points = []
        for item in main_lines:
            x1, y1, x2, y2 = item['segment']
            points.append([x1, y1])
            points.append([x2, y2])
        return np.asarray(points, dtype=np.float64)

    def _building_aware_crop_box(self, mask: np.ndarray, h_full: np.ndarray, main_lines: list[dict]):
        height, width = mask.shape[:2]
        src_points = self._main_structure_points(main_lines)
        dst_points = self._transform_points(h_full, src_points)
        dst_points = dst_points[np.all(np.isfinite(dst_points), axis=1)]
        if len(dst_points) < 4:
            return 0, 0, width, height

        qlo = np.quantile(dst_points, self.STRUCTURE_LOW_Q, axis=0)
        qhi = np.quantile(dst_points, self.STRUCTURE_HIGH_Q, axis=0)
        span = np.maximum(qhi - qlo, 1.0)
        y0 = int(math.floor(qlo[1] - self.CROP_MARGIN_Y_RATIO * span[1]))
        y1 = int(math.ceil(qhi[1] + self.CROP_MARGIN_Y_RATIO * span[1]))
        y0 = max(0, min(y0, height - 1))
        y1 = max(y0 + 1, min(y1, height))

        valid = mask > 0
        lefts, rights = [], []
        for y in range(y0, y1):
            xs = np.where(valid[y])[0]
            if len(xs) > 0:
                lefts.append(xs[0])
                rights.append(xs[-1])
        if len(lefts) < 4:
            return 0, 0, width, height

        x0_mask = int(math.ceil(np.quantile(lefts, self.MASK_LEFT_QUANTILE)))
        x1_mask = int(math.floor(np.quantile(rights, self.MASK_RIGHT_QUANTILE))) + 1
        x0_struct = int(math.floor(qlo[0] - self.CROP_MARGIN_X_RATIO * span[0]))
        x1_struct = int(math.ceil(qhi[0] + self.CROP_MARGIN_X_RATIO * span[0]))
        core_x0 = int(math.floor(np.quantile(dst_points[:, 0], 0.05)))
        core_x1 = int(math.ceil(np.quantile(dst_points[:, 0], 0.95)))

        x0 = max(0, x0_struct)
        x1 = min(width, x1_struct)
        if x0_mask <= core_x0:
            x0 = max(x0, x0_mask)
        if x1_mask >= core_x1:
            x1 = min(x1, x1_mask)
        if x1 <= x0 + 20:
            x0, x1 = x0_mask, x1_mask

        x0 += self.CROP_INSET_PX
        x1 -= self.CROP_INSET_PX
        y0 += self.CROP_INSET_PX
        y1 -= self.CROP_INSET_PX
        x0 = max(0, min(x0, width - 1))
        x1 = max(x0 + 1, min(x1, width))
        y0 = max(0, min(y0, height - 1))
        y1 = max(y0 + 1, min(y1, height))
        return x0, y0, x1, y1

    @staticmethod
    def _rect_sum(integral, x0, y0, x1, y1):
        return integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]

    def _adjust_crop_box_to_aspect(self, mask: np.ndarray, base_box, target_ratio: float):
        height, width = mask.shape[:2]
        bx0, by0, bx1, by1 = map(int, base_box)
        bw = max(1, bx1 - bx0)
        bh = max(1, by1 - by0)
        if target_ratio <= 0:
            return bx0, by0, bx1, by1

        target_w = bw
        target_h = int(math.ceil(target_w / target_ratio))
        if target_h < bh:
            target_h = bh
            target_w = int(math.ceil(target_h * target_ratio))

        if target_w > width or target_h > height:
            scale = min(width / max(target_w, 1), height / max(target_h, 1))
            target_w = max(bw, int(math.floor(target_w * scale)))
            target_h = max(bh, int(round(target_w / target_ratio)))
            if target_h > height:
                target_h = height
                target_w = int(round(target_h * target_ratio))
            if target_w > width:
                target_w = width
                target_h = int(round(target_w / target_ratio))
            if target_w < bw or target_h < bh or target_w > width or target_h > height:
                return bx0, by0, bx1, by1

        x0_min = max(0, bx1 - target_w)
        x0_max = min(bx0, width - target_w)
        y0_min = max(0, by1 - target_h)
        y0_max = min(by0, height - target_h)
        if x0_min > x0_max or y0_min > y0_max:
            return bx0, by0, bx1, by1

        valid = (mask > 0).astype(np.uint8)
        integral = cv2.integral(valid, sdepth=cv2.CV_64F)
        base_cx = 0.5 * (bx0 + bx1)
        base_cy = 0.5 * (by0 + by1)
        nx = max(1, min(self.ASPECT_SEARCH_STEPS, x0_max - x0_min + 1))
        ny = max(1, min(self.ASPECT_SEARCH_STEPS, y0_max - y0_min + 1))
        xs = np.unique(np.round(np.linspace(x0_min, x0_max, nx)).astype(int))
        ys = np.unique(np.round(np.linspace(y0_min, y0_max, ny)).astype(int))

        best = None
        best_score = -1e30
        area = float(target_w * target_h)
        for y0 in ys:
            y1 = y0 + target_h
            for x0 in xs:
                x1 = x0 + target_w
                valid_ratio = float(self._rect_sum(integral, x0, y0, x1, y1)) / area
                center_dist = math.hypot(
                    (x0 + 0.5 * target_w - base_cx) / max(width, 1),
                    (y0 + 0.5 * target_h - base_cy) / max(height, 1),
                )
                score = valid_ratio - self.ASPECT_CENTER_PENALTY * center_dist
                if score > best_score:
                    best_score = score
                    best = (int(x0), int(y0), int(x1), int(y1))
        return best if best is not None else (bx0, by0, bx1, by1)

    @staticmethod
    def _crop_with_box(warped: np.ndarray, mask: np.ndarray, h_full: np.ndarray, crop_box):
        x0, y0, x1, y1 = map(int, crop_box)
        cropped = warped[y0:y1, x0:x1].copy()
        cropped_mask = mask[y0:y1, x0:x1].copy()
        crop_matrix = np.array([
            [1.0, 0.0, -x0],
            [0.0, 1.0, -y0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        return cropped, cropped_mask, crop_matrix @ h_full

    @staticmethod
    def _resize_if_needed(image: np.ndarray, matrix: np.ndarray, target_max_dim: int):
        height, width = image.shape[:2]
        max_dim = max(width, height)
        if target_max_dim <= 0 or max_dim <= target_max_dim:
            return image, matrix
        scale = float(target_max_dim) / float(max_dim)
        resized = cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        scale_matrix = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]])
        return resized, scale_matrix @ matrix

    def _rectify_vertical_only(self, bgr: np.ndarray):
        lines = self._detect_vertical_lines(bgr)
        vp, inliers, residuals = self._estimate_vertical_vp(lines, bgr.shape, seed=0)
        main_lines, clusters = self._select_main_building_cluster(lines, inliers, bgr.shape)
        h_base, info = self._build_minimal_vertical_homography(bgr.shape, vp, main_lines)
        warped, mask, h_full = self._make_full_warp(bgr, h_base)
        base_box = self._building_aware_crop_box(mask, h_full, main_lines)
        if self.keep_original_aspect_ratio:
            original_ratio = bgr.shape[1] / float(max(bgr.shape[0], 1))
            crop_box = self._adjust_crop_box_to_aspect(mask, base_box, original_ratio)
        else:
            crop_box = base_box
        cropped, crop_mask, h_crop = self._crop_with_box(warped, mask, h_full, crop_box)
        cropped, h_crop = self._resize_if_needed(cropped, h_crop, self.target_max_dim)

        self.last_method = 'vertical_vp_v4'
        self.last_vp = np.asarray(vp, dtype=np.float64)
        self.last_crop_box = tuple(int(v) for v in crop_box)
        self.last_src_pts = None
        self.last_structure_points = self._main_structure_points(main_lines).astype(np.float32)
        h_out, w_out = cropped.shape[:2]
        self.last_dst_pts = np.float32([[0, 0], [w_out - 1, 0], [w_out - 1, h_out - 1], [0, h_out - 1]])
        self.last_info = {
            **info,
            'vertical_line_count': len(lines),
            'vertical_inlier_count': int(np.sum(inliers)),
            'main_line_count': len(main_lines),
            'cluster_sizes': [len(cluster) for cluster in clusters],
            'median_vertical_residual_deg': (
                float(np.median(residuals[inliers])) if np.any(inliers) else None
            ),
            'base_crop_box': tuple(int(v) for v in base_box),
            'crop_box': tuple(int(v) for v in crop_box),
            'invalid_ratio': float(1.0 - np.mean(crop_mask > 0)),
        }
        return cropped, h_crop

    def _rectify_identity_fallback(self, bgr: np.ndarray):
        self.last_method = 'identity_fallback'
        h, w = bgr.shape[:2]
        matrix = np.eye(3, dtype=np.float64)
        out, matrix = self._resize_if_needed(bgr.copy(), matrix, self.target_max_dim)
        out_h, out_w = out.shape[:2]
        self.last_src_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        self.last_structure_points = None
        self.last_dst_pts = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
        self.last_crop_box = (0, 0, w, h)
        return out, matrix

    def rectify(self, photo_img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        img = np.asarray(photo_img)
        if img.ndim == 2:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] >= 3:
            bgr = img[:, :, :3].copy()
        else:
            bgr = img.copy()

        try:
            rectified, matrix = self._rectify_vertical_only(bgr)
        except Exception as exc:
            rectified, matrix = self._rectify_identity_fallback(bgr)
            self.last_info = {'fallback_reason': str(exc)}
        return rectified, matrix, np.linalg.inv(matrix)


def rectify_photo_perspective(
    photo_path: str | Path,
    *,
    target_max_dim: int = 2048,
    correction_strength: float = 1.0,
    keep_original_aspect_ratio: bool = True,
) -> dict:
    """读取照片并按 v4 竖直消失点算法摆正建筑照片。"""
    path = Path(photo_path).expanduser().resolve()
    bgr = read_bgr_image(path)
    orig_h, orig_w = bgr.shape[:2]
    rectifier = PhotoPerspectiveRectifier(
        target_max_dim=target_max_dim,
        correction_strength=correction_strength,
        keep_original_aspect_ratio=keep_original_aspect_ratio,
    )
    rect_bgr, h_mat, h_inv = rectifier.rectify(bgr)
    rect_h, rect_w = rect_bgr.shape[:2]

    result = {
        'photo_path': str(path),
        'rectified_bgr': rect_bgr,
        'H': h_mat,
        'H_inv': h_inv,
        'method': rectifier.last_method,
        'original_size': (orig_w, orig_h),
        'rectified_size': (rect_w, rect_h),
        'content_mean': float(rect_bgr.mean()),
        'rectify_info': rectifier.last_info,
    }
    if rectifier.last_src_pts is not None:
        result['src_quad'] = rectifier.last_src_pts.tolist()
    if rectifier.last_dst_pts is not None:
        result['dst_quad'] = rectifier.last_dst_pts.tolist()
    if rectifier.last_structure_points is not None:
        result['structure_points'] = rectifier.last_structure_points.tolist()
    if rectifier.last_vp is not None:
        result['vertical_vp'] = rectifier.last_vp.tolist()
    if rectifier.last_crop_box is not None:
        result['crop_box'] = tuple(int(v) for v in rectifier.last_crop_box)
    return result


__all__ = [
    'PhotoPerspectiveRectifier',
    'rectify_photo_perspective',
]
