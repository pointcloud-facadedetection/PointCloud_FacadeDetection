"""2D 现场照片透视矫正（消影点 / 单应变换），用于匹配前拉平仰角。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import base64
import cv2
import numpy as np


class PhotoPerspectiveRectifier:
    """
    2D 现场照片透视仰角自动矫正器
    利用消影点 (Vanishing Point) 或几何单应性变换，将大仰角照片“拉平”为正面视角
    """

    def __init__(self, target_max_dim=1024):
        self.target_max_dim = int(target_max_dim)
        self.last_method = 'unknown'

    def _find_vertical_vanishing_point(self, gray_img):
        """利用霍夫直线检测计算垂直方向的消影点 (VP)"""
        h, w = gray_img.shape
        edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=80,
            minLineLength=int(h * 0.15), maxLineGap=20,
        )

        if lines is None or len(lines) < 2:
            return None

        vert_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = x2 - x1, y2 - y1
            angle = np.abs(np.arctan2(dy, dx) * 180.0 / np.pi)
            if 45.0 < angle < 135.0:
                a = y2 - y1
                b = x1 - x2
                c = x2 * y1 - x1 * y2
                line_norm = float(np.hypot(a, b))
                if line_norm > 1e-9:
                    vert_lines.append((a / line_norm, b / line_norm, c / line_norm))

        if len(vert_lines) < 2:
            return None

        a_mat = np.array([[line[0], line[1]] for line in vert_lines], dtype=np.float64)
        b_vec = np.array([-line[2] for line in vert_lines], dtype=np.float64)
        try:
            return np.linalg.lstsq(a_mat, b_vec, rcond=None)[0]
        except Exception:
            return None

    def _bounded_transform(self, h_mat, width, height):
        """平移变换后的画布限制到 target_max_dim，并同步更新单应矩阵。"""
        corners = np.array(
            [[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]],
            dtype=np.float64,
        ).T
        warped = h_mat @ corners
        if (
            not np.all(np.isfinite(warped))
            or np.any(np.abs(warped[2, :]) < 1e-9)
        ):
            raise ValueError('透视矫正产生无效画布')
        warped /= warped[2, :]
        x_min, x_max = float(np.min(warped[0])), float(np.max(warped[0]))
        y_min, y_max = float(np.min(warped[1])), float(np.max(warped[1]))
        out_w = max(1, int(np.ceil(x_max - x_min)))
        out_h = max(1, int(np.ceil(y_max - y_min)))
        transform = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1],
        ], dtype=np.float64) @ h_mat
        max_dim = max(out_w, out_h)
        if self.target_max_dim > 0 and max_dim > self.target_max_dim:
            scale = float(self.target_max_dim) / float(max_dim)
            scale_mat = np.array([
                [scale, 0, 0],
                [0, scale, 0],
                [0, 0, 1],
            ], dtype=np.float64)
            transform = scale_mat @ transform
            out_w = max(1, int(round(out_w * scale)))
            out_h = max(1, int(round(out_h * scale)))
        return transform, out_w, out_h

    def rectify(self, photo_img):
        """
        对输入 2D 照片做透视校正。

        :param photo_img: BGR/RGB/灰度 numpy 图像
        :return: rect_img, H_final, H_inv
        """
        img = np.asarray(photo_img)
        if img.ndim == 2:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 3:
            bgr = img.copy()
        else:
            bgr = img[:, :, :3].copy()

        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        vp = self._find_vertical_vanishing_point(gray)

        if vp is None or vp[1] >= 0:
            self.last_method = 'mild_trapezoid'
            # 顶部适当拉宽，抵消仰拍近大远小（角点顺序必须与 src 一致：TL/TR/BR/BL）
            crop_w = w * 0.15
            src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst_pts = np.float32([
                [-crop_w, 0],
                [w + crop_w, 0],
                [w, h],
                [0, h],
            ])
            h_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
        else:
            self.last_method = 'vanishing_point'
            k = np.array([[w, 0, w / 2.0], [0, h, h / 2.0], [0, 0, 1.0]], dtype=np.float64)
            vp_homo = np.array([vp[0], vp[1], 1.0])
            v_dir = np.linalg.inv(k) @ vp_homo
            v_dir /= np.linalg.norm(v_dir)
            roll = np.arctan2(v_dir[0], -v_dir[1])
            pitch = np.arcsin(np.clip(v_dir[2], -1.0, 1.0))
            r_roll = np.array([
                [np.cos(roll), -np.sin(roll), 0],
                [np.sin(roll), np.cos(roll), 0],
                [0, 0, 1],
            ])
            r_pitch = np.array([
                [1, 0, 0],
                [0, np.cos(pitch), -np.sin(pitch)],
                [0, np.sin(pitch), np.cos(pitch)],
            ])
            r_rect = r_pitch @ r_roll
            h_mat = k @ r_rect @ np.linalg.inv(k)

        h_final, out_w, out_h = self._bounded_transform(h_mat, w, h)
        h_inv = np.linalg.inv(h_final)
        rect_img = cv2.warpPerspective(
            bgr, h_final, (out_w, out_h), flags=cv2.INTER_LINEAR,
        )

        # 消影点路径若产生异常黑图，回退到 mild 梯形校正
        if float(rect_img.mean()) < 5.0:
            self.last_method = 'mild_trapezoid_fallback'
            crop_w = w * 0.08
            src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst_pts = np.float32([
                [-crop_w, 0],
                [w + crop_w, 0],
                [w, h],
                [0, h],
            ])
            h_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
            h_final, out_w, out_h = self._bounded_transform(h_mat, w, h)
            h_inv = np.linalg.inv(h_final)
            rect_img = cv2.warpPerspective(
                bgr, h_final, (out_w, out_h), flags=cv2.INTER_LINEAR,
            )

        return rect_img, h_final, h_inv

    @staticmethod
    def map_keypoints_back(kpts_rect, h_inv):
        """将校正后图像上的匹配点映射回原始照片像素坐标。"""
        kpts_rect = np.asarray(kpts_rect, dtype=np.float64).reshape(-1, 2)
        if kpts_rect.size == 0:
            return np.empty((0, 2), dtype=np.float32)

        h_inv = np.asarray(h_inv, dtype=np.float64).reshape(3, 3)
        pts_homo = np.hstack([kpts_rect, np.ones((len(kpts_rect), 1))]).T
        orig_homo = h_inv @ pts_homo
        orig_pts = (orig_homo[:2, :] / orig_homo[2, :]).T
        return orig_pts.astype(np.float32)


def perspective_rectify_paths(photo_id: str, photo_folder: str) -> tuple[str, str, str]:
    """返回 (rectified_photo_id, rectified_path, meta_path)。"""
    stem, ext = os.path.splitext(photo_id)
    if not ext:
        ext = '.jpg'
    rect_id = f'{stem}_persp_rect{ext}'
    rect_path = os.path.join(photo_folder, rect_id)
    meta_path = os.path.join(photo_folder, f'{stem}_persp_rect.json')
    return rect_id, rect_path, meta_path


def save_perspective_rectify_meta(meta_path, *, photo_id, rectified_photo_id, h_mat, h_inv, original_size, rectified_size):
    payload = {
        'photo_id': photo_id,
        'rectified_photo_id': rectified_photo_id,
        'H': np.asarray(h_mat, dtype=np.float64).reshape(3, 3).tolist(),
        'H_inv': np.asarray(h_inv, dtype=np.float64).reshape(3, 3).tolist(),
        'original_width_px': int(original_size[0]),
        'original_height_px': int(original_size[1]),
        'rectified_width_px': int(rectified_size[0]),
        'rectified_height_px': int(rectified_size[1]),
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def load_perspective_rectify_meta(meta_path: str) -> dict:
    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)
    meta['H'] = np.asarray(meta['H'], dtype=np.float64)
    meta['H_inv'] = np.asarray(meta['H_inv'], dtype=np.float64)
    return meta


def rectify_photo_for_matching(photo_path, *, photo_folder=None, target_max_dim=1024) -> dict:
    """
    矫正照片并落盘，供后续 2D 匹配使用。

    返回 dict 含 rectified_photo_id、预览 base64、单应矩阵等。
    """
    photo_path = Path(photo_path)
    if not photo_path.is_file():
        raise FileNotFoundError(f'照片不存在: {photo_path}')

    bgr = cv2.imread(str(photo_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f'无法读取照片: {photo_path}')

    orig_h, orig_w = bgr.shape[:2]
    rectifier = PhotoPerspectiveRectifier(target_max_dim=target_max_dim)
    rect_bgr, h_mat, h_inv = rectifier.rectify(bgr)
    rect_h, rect_w = rect_bgr.shape[:2]

    photo_id = photo_path.name
    folder = str(photo_folder or photo_path.parent)
    os.makedirs(folder, exist_ok=True)
    rect_id, rect_path, meta_path = perspective_rectify_paths(photo_id, folder)

    if not cv2.imwrite(rect_path, rect_bgr):
        raise IOError(f'矫正照片保存失败: {rect_path}')

    meta = save_perspective_rectify_meta(
        meta_path,
        photo_id=photo_id,
        rectified_photo_id=rect_id,
        h_mat=h_mat,
        h_inv=h_inv,
        original_size=(orig_w, orig_h),
        rectified_size=(rect_w, rect_h),
    )

    ok, buf = cv2.imencode('.png', rect_bgr)
    if not ok:
        raise ValueError('矫正预览图编码失败')

    preview_bgr = rect_bgr
    max_preview = max(rect_h, rect_w)
    if max_preview > 1600:
        scale = 1600.0 / max_preview
        preview_bgr = cv2.resize(
            rect_bgr,
            (int(rect_w * scale), int(rect_h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    ok_preview, preview_buf = cv2.imencode('.png', preview_bgr)
    if ok_preview:
        buf = preview_buf

    return {
        **meta,
        'rectified_path': rect_path,
        'meta_path': meta_path,
        'preview_image_base64': base64.b64encode(buf.tobytes()).decode('ascii'),
        'image_mime': 'image/png',
        'method': rectifier.last_method,
        'content_mean': float(rect_bgr.mean()),
    }


__all__ = [
    'PhotoPerspectiveRectifier',
    'load_perspective_rectify_meta',
    'map_keypoints_back',
    'perspective_rectify_paths',
    'rectify_photo_for_matching',
]

# 兼容旧调用名
map_keypoints_back = PhotoPerspectiveRectifier.map_keypoints_back
