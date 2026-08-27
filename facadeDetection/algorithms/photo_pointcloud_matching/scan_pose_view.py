"""兼容层：已迁移至 algorithms.View_aligned_photo_pointcloud_matching。"""

from algorithms.View_aligned_photo_pointcloud_matching.scan_pose_view import (
    build_scan_viewport_camera,
    validate_scan_pose_json,
)

__all__ = [
    'validate_scan_pose_json',
    'build_scan_viewport_camera',
]
