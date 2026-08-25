"""手动 2D-3D 匹配：点对状态与相机位姿估计。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtGui import QImage, QImageReader

from algorithms.photo_pointcloud_matching import solve_camera_pose


PHOTO_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


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


class PhotoMatchService:
    def __init__(self):
        self.state = PhotoMatchState()

    def reset(self):
        self.state = PhotoMatchState()

    def load_photo(self, file_path: str) -> QImage:
        path = Path(file_path).expanduser().resolve()
        if path.suffix.lower() not in PHOTO_SUFFIXES:
            raise ValueError(f'不支持的照片格式: {path.suffix}')
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
        return image

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
        return result
