"""手动 2D-3D 匹配：点对状态与相机位姿估计。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtGui import QImage, QImageReader

from algorithms.photo_pointcloud_matching import solve_camera_pose
from algorithms.View_aligned_photo_pointcloud_matching import (
    PHOTO_SUFFIXES,
    validate_photo_path,
    rectify_photo_perspective,
    validate_scan_pose_json,
    build_scan_viewport_camera,
)


@dataclass
class MatchPair:
    photo: Optional[tuple[float, float]] = None
    cloud: Optional[tuple[float, float, float]] = None


@dataclass
class PhotoMatchState:
    photo_path: Optional[str] = None
    image_width: int = 0
    image_height: int = 0
    correspondences: list[MatchPair] = field(default_factory=list)
    annotating: bool = False
    next_is_photo: bool = True
    pose: Optional[dict] = None
    match_mode: str = 'manual'
    rectified: Optional[dict] = None
    scan_pose_path: Optional[str] = None
    scan_pose_meta: Optional[dict] = None
    annotation_space: str = 'raw'


class PhotoMatchService:
    def __init__(self):
        self.state = PhotoMatchState()

    def reset(self):
        self.state = PhotoMatchState()

    def load_photo(self, file_path: str) -> QImage:
        path = validate_photo_path(file_path)
        reader = QImageReader(str(path))
        reader.setAutoTransform(True)
        image = reader.read()
        if image.isNull():
            raise ValueError(reader.errorString() or '无法读取照片')
        self.state.photo_path = str(path)
        self.state.image_width = image.width()
        self.state.image_height = image.height()
        self.state.correspondences = []
        self.state.annotating = False
        self.state.next_is_photo = True
        self.state.pose = None
        self.state.match_mode = 'manual'
        self.state.rectified = None
        self.state.annotation_space = 'raw'
        return image

    def rectify_perspective(self) -> dict:
        """消除照片仰拍透视，使建筑上下边近似垂直。"""
        if not self.state.photo_path:
            raise ValueError('请先上传 2D 照片')
        result = rectify_photo_perspective(self.state.photo_path)
        self.state.rectified = {
            key: value
            for key, value in result.items()
            if key != 'rectified_bgr'
        }
        rect_w, rect_h = result.get('rectified_size', (self.state.image_width, self.state.image_height))
        self.state.image_width = int(rect_w)
        self.state.image_height = int(rect_h)
        self.state.correspondences = []
        self.state.annotating = False
        self.state.next_is_photo = True
        self.state.pose = None
        self.state.match_mode = 'manual'
        self.state.annotation_space = 'rectified'
        return result

    def load_scan_pose(self, file_path: str) -> dict:
        """上传并校验扫描仪位姿 JSON。"""
        path = Path(file_path).expanduser().resolve()
        meta = validate_scan_pose_json(path)
        self.state.scan_pose_path = str(path)
        self.state.scan_pose_meta = meta
        return meta

    def build_viewport_camera(self, lookat) -> dict:
        """按已上传的扫描位姿与观察目标计算视口相机。"""
        if not self.state.scan_pose_path:
            raise ValueError('请先上传扫描仪位姿文件')
        return build_scan_viewport_camera(self.state.scan_pose_path, lookat)

    def complete_pair_count(self) -> int:
        return sum(
            1
            for pair in self.state.correspondences
            if pair.photo is not None and pair.cloud is not None
        )

    def add_photo_point(self, pixel_x: float, pixel_y: float):
        if not self.state.annotating:
            raise RuntimeError('请先进入标注模式')
        if not self.state.next_is_photo:
            raise RuntimeError('请先在 3D 点云中点击对应点')
        last = self.state.correspondences[-1] if self.state.correspondences else None
        if last is not None and last.cloud is None:
            last.photo = (float(pixel_x), float(pixel_y))
        else:
            self.state.correspondences.append(
                MatchPair(photo=(float(pixel_x), float(pixel_y)))
            )
        self.state.next_is_photo = False
        self.state.pose = None
        self.state.match_mode = 'manual'

    def add_cloud_point(self, point) -> None:
        if not self.state.annotating:
            raise RuntimeError('请先进入标注模式')
        if self.state.next_is_photo:
            raise RuntimeError('请先在照片中点击对应点')
        xyz = tuple(float(v) for v in np.asarray(point, dtype=float).reshape(3))
        last = self.state.correspondences[-1] if self.state.correspondences else None
        if last is None or last.cloud is not None:
            self.state.correspondences.append(MatchPair(cloud=xyz))
        else:
            last.cloud = xyz
        self.state.next_is_photo = True
        self.state.pose = None
        self.state.match_mode = 'manual'

    def undo_last(self) -> None:
        if not self.state.correspondences:
            return
        last = self.state.correspondences[-1]
        if last.cloud is not None and last.photo is not None:
            last.cloud = None
            self.state.next_is_photo = False
        else:
            self.state.correspondences.pop()
            self.state.next_is_photo = True
        self.state.pose = None

    def cloud_points(self) -> list[tuple[float, float, float]]:
        return [
            pair.cloud
            for pair in self.state.correspondences
            if pair.cloud is not None
        ]

    def photo_points(self) -> list[tuple[float, float]]:
        return [
            pair.photo
            for pair in self.state.correspondences
            if pair.photo is not None
        ]

    def solve_pose(self) -> dict:
        pairs = [
            pair
            for pair in self.state.correspondences
            if pair.photo is not None and pair.cloud is not None
        ]
        if len(pairs) < 6:
            raise ValueError('估计相机内参至少需要 6 对完整的 2D-3D 匹配点')
        if self.state.image_width <= 0 or self.state.image_height <= 0:
            raise ValueError('缺少照片原始宽高')
        result = solve_camera_pose(
            object_points=[pair.cloud for pair in pairs],
            image_points=[pair.photo for pair in pairs],
            image_width=self.state.image_width,
            image_height=self.state.image_height,
        )
        self.state.pose = result
        self.state.match_mode = 'manual'
        return result

    @staticmethod
    def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        center = np.mean(points, axis=0)
        centered = points - center.reshape(1, 3)
        if len(points) < 3 or np.linalg.matrix_rank(centered) < 2:
            raise ValueError('反向检查至少需要 3 个分布有效的 3D 标注点')
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        return center, normal

    @staticmethod
    def _project_points(points: np.ndarray, camera_matrix, rotation, translation) -> np.ndarray:
        cam = (rotation @ points.T).T + translation.reshape(1, 3)
        z = cam[:, 2:3]
        uvw = (camera_matrix @ cam.T).T
        return uvw[:, :2] / np.maximum(z, 1e-9)

    @staticmethod
    def _apply_homography(matrix, points_xy: np.ndarray) -> np.ndarray:
        homography = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
        pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        hom = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
        mapped = (homography @ hom.T).T
        return mapped[:, :2] / np.maximum(mapped[:, 2:3], 1e-9)

    def reverse_mapping_check(self) -> dict:
        """用已求得的 A/T/M 反投影标注像素，检查回到立面平面的 3D 误差。"""
        if self.state.pose is None:
            raise ValueError('请先估算相机内外参数')
        pairs = [
            pair
            for pair in self.state.correspondences
            if pair.photo is not None and pair.cloud is not None
        ]
        if len(pairs) < 3:
            raise ValueError('反向映射检查至少需要 3 对完整标注点')

        image_points = np.asarray([pair.photo for pair in pairs], dtype=np.float64).reshape(-1, 2)
        object_points = np.asarray([pair.cloud for pair in pairs], dtype=np.float64).reshape(-1, 3)
        pose = self.state.pose
        camera_matrix = np.asarray(pose['camera_matrix'], dtype=np.float64).reshape(3, 3)
        rotation = np.asarray(pose['rotation_matrix'], dtype=np.float64).reshape(3, 3)
        translation = np.asarray(pose['translation_vector'], dtype=np.float64).reshape(3)
        plane_point, plane_normal = self._fit_plane(object_points)
        camera_center = -rotation.T @ translation
        inv_camera = np.linalg.inv(camera_matrix)

        back_points = []
        distances = []
        valid = []
        for pixel, target in zip(image_points, object_points):
            ray_camera = inv_camera @ np.array([pixel[0], pixel[1], 1.0], dtype=np.float64)
            ray_world = rotation.T @ ray_camera
            ray_world = ray_world / (np.linalg.norm(ray_world) + 1e-12)
            denom = float(np.dot(plane_normal, ray_world))
            if abs(denom) < 1e-9:
                back_points.append([float('nan'), float('nan'), float('nan')])
                distances.append(float('nan'))
                valid.append(False)
                continue
            depth = float(np.dot(plane_normal, plane_point - camera_center) / denom)
            point = camera_center + depth * ray_world
            back_points.append(point.tolist())
            distances.append(float(np.linalg.norm(point - target)))
            valid.append(np.isfinite(point).all() and depth > 0)

        projected = self._project_points(object_points, camera_matrix, rotation, translation)
        reprojection_errors = np.linalg.norm(projected - image_points, axis=1)
        valid_mask = np.asarray(valid, dtype=bool) & np.isfinite(distances)
        raw_pixels = None
        rectified = self.state.rectified or {}
        matrix_m = rectified.get('H')
        matrix_m_inv = rectified.get('H_inv')
        if matrix_m_inv is not None:
            raw_pixels = self._apply_homography(matrix_m_inv, image_points).tolist()

        return {
            'annotation_space': self.state.annotation_space,
            'matrix_A': camera_matrix.tolist(),
            'matrix_T': np.hstack([rotation, translation.reshape(3, 1)]).tolist(),
            'matrix_M': (
                np.asarray(matrix_m, dtype=np.float64).reshape(3, 3).tolist()
                if matrix_m is not None else None
            ),
            'matrix_M_inv': (
                np.asarray(matrix_m_inv, dtype=np.float64).reshape(3, 3).tolist()
                if matrix_m_inv is not None else None
            ),
            'plane_point': plane_point.tolist(),
            'plane_normal': plane_normal.tolist(),
            'camera_center_world': camera_center.tolist(),
            'back_projected_points': back_points,
            'raw_pixels_from_M_inv': raw_pixels,
            'distance_errors_m': distances,
            'mean_distance_error_m': (
                float(np.mean(np.asarray(distances)[valid_mask])) if np.any(valid_mask) else None
            ),
            'max_distance_error_m': (
                float(np.max(np.asarray(distances)[valid_mask])) if np.any(valid_mask) else None
            ),
            'reprojection_errors_px': reprojection_errors.tolist(),
            'mean_reprojection_error_px': float(np.mean(reprojection_errors)),
            'max_reprojection_error_px': float(np.max(reprojection_errors)),
            'valid_count': int(np.sum(valid_mask)),
            'point_count': int(len(pairs)),
        }

    def apply_auto_match_result(self, result: dict) -> dict:
        """把自动匹配得到的 2D-3D 对应点与位姿写入当前状态。"""
        correspondences = list(result.get('correspondences') or [])
        inlier_indices = result.get('inlier_indices') or []
        if inlier_indices:
            selected = []
            for idx in inlier_indices:
                try:
                    selected.append(correspondences[int(idx)])
                except (IndexError, TypeError, ValueError):
                    continue
            if selected:
                correspondences = selected
        pairs = []
        for item in correspondences:
            image_point = item.get('image_point')
            object_point = item.get('object_point')
            if image_point is None or object_point is None:
                continue
            photo_xy = tuple(float(v) for v in np.asarray(image_point, dtype=float).reshape(2))
            cloud_xyz = tuple(float(v) for v in np.asarray(object_point, dtype=float).reshape(3))
            pairs.append(MatchPair(photo=photo_xy, cloud=cloud_xyz))
        if len(pairs) < 6:
            raise ValueError('自动匹配有效点对不足 6 对，无法写入结果')
        pose = {
            key: value
            for key, value in result.items()
            if key not in ('correspondences', 'match_visualization')
        }
        self.state.correspondences = pairs
        self.state.annotating = False
        self.state.next_is_photo = True
        self.state.pose = pose
        self.state.match_mode = 'auto'
        return pose


def bgr_to_qimage(bgr) -> QImage:
    """OpenCV BGR 图转为可安全持有的 QImage。"""
    import cv2

    rgb = cv2.cvtColor(np.asarray(bgr), cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    height, width = rgb.shape[:2]
    return QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()

