"""2D 照片与 3D 点云匹配算法（PnP、热力图等）。"""

from .pnp_solver import (
    estimate_camera_matrix,
    estimate_camera_matrix_from_correspondences,
    estimate_camera_matrix_with_radial_distortion,
    solve_camera_pose,
    refine_camera_pose_fixed_intrinsics,
)
from .facade_heatmap import (
    create_photo_facade_heatmap,
    select_visible_largest_facade,
    select_facade_in_photo_quadrilateral,
    build_region_highlight_colors,
    resolve_photo_detection_region,
    refine_quadrilateral_corners_on_facade,
    project_world_points_to_image,
    snap_point_to_facade_plane,
)

__all__ = [
    "estimate_camera_matrix",
    "estimate_camera_matrix_from_correspondences",
    "estimate_camera_matrix_with_radial_distortion",
    "solve_camera_pose",
    "refine_camera_pose_fixed_intrinsics",
    "create_photo_facade_heatmap",
    "select_visible_largest_facade",
    "select_facade_in_photo_quadrilateral",
    "build_region_highlight_colors",
    "resolve_photo_detection_region",
    "refine_quadrilateral_corners_on_facade",
    "project_world_points_to_image",
    "snap_point_to_facade_plane",
]
