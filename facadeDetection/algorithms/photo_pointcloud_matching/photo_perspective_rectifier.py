"""兼容层：已迁移至 algorithms.View_aligned_photo_pointcloud_matching。"""

from algorithms.View_aligned_photo_pointcloud_matching.photo_perspective_rectifier import (
    PhotoPerspectiveRectifier,
    rectify_photo_perspective,
)
from algorithms.View_aligned_photo_pointcloud_matching.photo_loader import (
    read_bgr_image as _read_bgr_image,
)

__all__ = [
    'PhotoPerspectiveRectifier',
    'rectify_photo_perspective',
    '_read_bgr_image',
]
