"""手动 2D-3D 匹配：点对状态与相机位姿估计。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtGui import QImage, QImageReader

from algorithms.View_aligned_photo_pointcloud_matching import (
    MIN_MATCH_PAIRS,
    validate_photo_path,
    rectify_photo_perspective,
    validate_scan_pose_json,
    build_scan_viewport_camera,
    estimate_match_matrix,
    match_photo_to_cloud_view,
    remap_cloud_points_to_photo,
    default_projection_params,
    render_projection,
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
    raw_image_width: int = 0
    raw_image_height: int = 0
    correspondences: list[MatchPair] = field(default_factory=list)
    annotating: bool = False
    next_is_photo: bool = True
    pose: Optional[dict] = None
    match_mode: str = 'manual'
    rectified: Optional[dict] = None
    scan_pose_path: Optional[str] = None
    scan_pose_meta: Optional[dict] = None
    annotation_space: str = 'raw'
    remapped_photo_points: list[Optional[tuple[float, float]]] = field(
        default_factory=list
    )
    projection_params: dict = field(default_factory=dict)


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
        self.state.raw_image_width = image.width()
        self.state.raw_image_height = image.height()
        self.state.correspondences = []
        self.state.annotating = False
        self.state.next_is_photo = True
        self.state.pose = None
        self.state.match_mode = 'manual'
        self.state.rectified = None
        self.state.annotation_space = 'raw'
        self.state.remapped_photo_points = []
        return image

    def rectify_perspective(self) -> dict:
        """按最小畸变竖直消失点算法摆正整栋建筑。"""
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
        self.state.remapped_photo_points = []
        return result

    def use_raw_annotation_space(self, image_width: int | None = None, image_height: int | None = None) -> bool:
        """切回原图坐标标注，返回是否清空了旧点对。"""
        if image_width is not None and image_height is not None:
            self.state.raw_image_width = int(image_width)
            self.state.raw_image_height = int(image_height)

        cleared = False
        if self.state.annotation_space != 'raw' and self.state.correspondences:
            self.state.correspondences = []
            cleared = True

        self.state.image_width = int(self.state.raw_image_width or self.state.image_width)
        self.state.image_height = int(self.state.raw_image_height or self.state.image_height)
        self.state.annotation_space = 'raw'
        self.state.pose = None
        self.state.match_mode = 'manual'
        self.state.next_is_photo = True
        self.state.remapped_photo_points = []
        return cleared

    def load_scan_pose(self, file_path: str) -> dict:
        """上传并校验扫描仪位姿 JSON。"""
        path = Path(file_path).expanduser().resolve()
        meta = validate_scan_pose_json(path)
        self.state.scan_pose_path = str(path)
        self.state.scan_pose_meta = meta
        self.state.projection_params = {}
        return meta

    def build_viewport_camera(self, lookat) -> dict:
        """按已上传的扫描位姿与观察目标计算视口相机。"""
        if not self.state.scan_pose_path:
            raise ValueError('请先上传扫描仪位姿文件')
        return build_scan_viewport_camera(self.state.scan_pose_path, lookat)

    def initialize_projection(self, points) -> dict:
        """根据扫描仪位姿和点云主体初始化针孔投影参数。"""
        meta = self.state.scan_pose_meta or {}
        transform = meta.get('transform_to_global')
        if transform is None:
            raise ValueError('扫描仪位姿缺少 transformToGlobal，无法生成针孔投影')
        params = default_projection_params(points, transform)
        self.state.projection_params = dict(params)
        return dict(params)

    def render_projection_view(
        self,
        points,
        colors,
        params=None,
        *,
        crop_subject=False,
        image_size=(1024, 576),
    ) -> dict:
        """渲染右侧针孔投影预览或用于匹配的主体裁剪图。"""
        meta = self.state.scan_pose_meta or {}
        transform = meta.get('transform_to_global')
        if transform is None:
            raise ValueError('请先上传含 transformToGlobal 的扫描仪位姿')
        values = dict(params or self.state.projection_params)
        if not values:
            values = self.initialize_projection(points)
        self.state.projection_params = dict(values)
        return render_projection(
            points,
            colors,
            transform,
            values,
            image_size=image_size,
            crop_subject=crop_subject,
        )

    def complete_pair_count(self) -> int:
        return sum(
            1
            for pair in self.state.correspondences
            if pair.photo is not None and pair.cloud is not None
        )

    def can_exit_annotation(self) -> bool:
        return self.state.annotating

    def can_estimate_match_matrix(self) -> bool:
        return (
            not self.state.annotating
            and self.complete_pair_count() >= MIN_MATCH_PAIRS
        )

    def enter_annotation(self) -> None:
        if not self.state.photo_path:
            raise ValueError('请先上传 2D 照片')
        last = self.state.correspondences[-1] if self.state.correspondences else None
        if last is not None and last.photo is not None and last.cloud is None:
            self.state.next_is_photo = False
        else:
            self.state.next_is_photo = True
        self.state.annotating = True
        self.state.pose = None
        self.state.match_mode = 'manual'
        self.state.remapped_photo_points = []

    def exit_annotation(self, *, force: bool = False) -> int:
        _ = force  # 保留旧调用兼容；现在任何点数均允许退出。
        if not self.state.annotating:
            return self.complete_pair_count()
        last = self.state.correspondences[-1] if self.state.correspondences else None
        if last is not None and (last.photo is None or last.cloud is None):
            self.state.correspondences.pop()
        self.state.annotating = False
        self.state.next_is_photo = True
        return self.complete_pair_count()

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
        self.state.remapped_photo_points = []

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
        self.state.remapped_photo_points = []

    def undo_last(self) -> None:
        """撤销上一个标注点（照片或点云），而不是整对。"""
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
        self.state.remapped_photo_points = []

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
        if self.state.annotating:
            raise ValueError('请先退出标注，再估算匹配矩阵')
        pairs = [
            pair
            for pair in self.state.correspondences
            if pair.photo is not None and pair.cloud is not None
        ]
        result = estimate_match_matrix(
            object_points=[pair.cloud for pair in pairs],
            image_points=[pair.photo for pair in pairs],
            image_width=self.state.image_width,
            image_height=self.state.image_height,
        )
        self.state.pose = result
        self.state.match_mode = 'manual'
        self.state.remapped_photo_points = []
        return result

    def remap_cloud_annotations_to_photo(self) -> dict:
        """用已估算的匹配矩阵，把 3D 标注点投影回照片。"""
        if self.state.annotating:
            raise ValueError('请先退出标注并估算匹配矩阵')
        if self.state.pose is None:
            raise ValueError('请先估算匹配矩阵')
        cloud_points = [
            pair.cloud
            for pair in self.state.correspondences
            if pair.photo is not None and pair.cloud is not None
        ]
        pose = self.state.pose
        result = remap_cloud_points_to_photo(
            cloud_points,
            match_matrix=pose.get('match_matrix') or pose.get('projection_matrix'),
            camera_matrix=pose.get('camera_matrix'),
            rotation_matrix=pose.get('rotation_matrix'),
            translation_vector=pose.get('translation_vector'),
            image_width=self.state.image_width,
            image_height=self.state.image_height,
        )
        photo_xy = []
        for pixel, valid in zip(result['image_points'], result['valid_mask']):
            if valid and np.isfinite(pixel).all():
                photo_xy.append((float(pixel[0]), float(pixel[1])))
            else:
                photo_xy.append(None)
        self.state.remapped_photo_points = photo_xy

        annotated = np.asarray(self.photo_points(), dtype=np.float64).reshape(-1, 2)
        remapped = np.asarray(result['image_points'], dtype=np.float64).reshape(-1, 2)
        count = min(len(annotated), len(remapped))
        if count:
            errors = np.linalg.norm(annotated[:count] - remapped[:count], axis=1)
            finite = np.isfinite(errors)
            result['reprojection_errors_px'] = errors.tolist()
            result['mean_reprojection_error_px'] = (
                float(np.mean(errors[finite])) if np.any(finite) else None
            )
            result['max_reprojection_error_px'] = (
                float(np.max(errors[finite])) if np.any(finite) else None
            )
        else:
            result['reprojection_errors_px'] = []
            result['mean_reprojection_error_px'] = None
            result['max_reprojection_error_px'] = None
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
        if self.state.annotation_space == 'rectified' and matrix_m_inv is not None:
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

    def auto_match_current_view(self, photo_bgr, captured_view: dict) -> dict:
        """匹配当前点云截图与照片，并写入自动生成的 2D-3D 点对和矩阵。"""
        if not self.state.photo_path:
            raise ValueError('请先上传 2D 照片')
        if self.state.annotating:
            raise ValueError('请先退出手动标注模式')
        result = match_photo_to_cloud_view(
            photo_bgr=photo_bgr,
            view_bgr=captured_view['view_bgr'],
            depth_image=captured_view['depth_image'],
            view_camera_matrix=captured_view['camera_matrix'],
            view_extrinsic=captured_view['extrinsic'],
            cloud_points=captured_view.get('cloud_points'),
            pixel_point_index=captured_view.get('pixel_point_index'),
        )
        result['annotation_space'] = self.state.annotation_space
        self.apply_auto_match_result(result)
        return result

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
        has_pose = bool(result.get('pose_estimated')) and bool(
            result.get('match_matrix') or result.get('projection_matrix')
        )
        pose = None
        if has_pose:
            pose = {
                key: value
                for key, value in result.items()
                if key not in ('correspondences', 'match_visualization')
            }
        self.state.correspondences = pairs
        self.state.annotating = False
        self.state.next_is_photo = True
        self.state.pose = pose
        self.state.match_mode = 'auto' if has_pose else 'auto_partial'
        photo_size = result.get('photo_image_size') or []
        if len(photo_size) == 2:
            self.state.image_width = int(photo_size[0])
            self.state.image_height = int(photo_size[1])
        self.state.annotation_space = str(
            result.get('annotation_space') or self.state.annotation_space
        )
        self.state.remapped_photo_points = []
        return pose


def bgr_to_qimage(bgr) -> QImage:
    """OpenCV BGR 图转为可安全持有的 QImage。"""
    import cv2

    rgb = cv2.cvtColor(np.asarray(bgr), cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    height, width = rgb.shape[:2]
    return QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()

