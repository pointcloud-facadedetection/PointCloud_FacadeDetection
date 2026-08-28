"""View-aligned photo point cloud matching: upload photo, rectify view, scan pose, cloud view."""

from .photo_loader import (
    PHOTO_SUFFIXES,
    load_photo_info,
    read_bgr_image,
    validate_photo_path,
)
from .photo_perspective_rectifier import (
    PhotoPerspectiveRectifier,
    rectify_photo_perspective,
)
from .scan_pose_view import (
    build_scan_viewport_camera,
    validate_scan_pose_json,
)
from .manual_match import (
    MIN_MATCH_PAIRS,
    estimate_match_matrix,
    remap_cloud_points_to_photo,
)
from .auto_view_match import match_photo_to_cloud_view
from .perspective_camera import (
    default_projection_params,
    projection_camera,
    render_projection,
)

__all__ = [
    'PHOTO_SUFFIXES',
    'validate_photo_path',
    'read_bgr_image',
    'load_photo_info',
    'PhotoPerspectiveRectifier',
    'rectify_photo_perspective',
    'validate_scan_pose_json',
    'build_scan_viewport_camera',
    'MIN_MATCH_PAIRS',
    'estimate_match_matrix',
    'remap_cloud_points_to_photo',
    'match_photo_to_cloud_view',
    'default_projection_params',
    'projection_camera',
    'render_projection',
]
