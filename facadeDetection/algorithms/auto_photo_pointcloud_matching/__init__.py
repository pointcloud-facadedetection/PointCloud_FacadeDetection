"""自动 2D 照片与 3D 点云立面匹配（无需人工标点对）。"""

from .matcher import (
    AutoMatchError,
    match_photo_to_facade,
    match_photo_to_facade_2d,
    ortho_image_for_matching,
)
from .photo_rectifier import (
    PhotoPerspectiveRectifier,
    load_perspective_rectify_meta,
    perspective_rectify_paths,
    rectify_photo_for_matching,
)
from .backProjector import (
    Facade3DBackprojector,
    backproject_matches_to_3d,
    mapping_to_render_assets,
)
from .facade_2D_matcher import (
    FacadeFeatureMatcher,
    sample_verify_matches,
    scale_photo_keypoints_to_original,
)
from .facade_glurestick_2D_matcher import (
    GLUESTICK_AVAILABLE,
    FacadeGlueStickMatcher,
    match_photo_to_ortho,
)
from .facade_render import (
    FacadeOrthorenderer,
    normalize_plane_params,
    pixel_to_uv,
    pixel_to_world,
    uv_to_pixel,
    world_to_pixel,
)
from .ortho_renderer import render_facade_orthographic, prepare_ortho_api_payload
from .pose_estimator import FacadePoseEstimator, estimate_camera_pose_ransac
from .heatmap_overplay import FacadeHeatmapOverlay, create_facade_heatmap_overlay
from .view_rectifier import create_ortho_rectified_view, rectify_image_to_ortho_view
from .main_pipeline import run_auto_matching_pipeline

__all__ = [
    "AutoMatchError",
    "match_photo_to_facade",
    "match_photo_to_facade_2d",
    "ortho_image_for_matching",
    "PhotoPerspectiveRectifier",
    "rectify_photo_for_matching",
    "load_perspective_rectify_meta",
    "perspective_rectify_paths",
    "run_auto_matching_pipeline",
    "FacadeFeatureMatcher",
    "FacadeGlueStickMatcher",
    "GLUESTICK_AVAILABLE",
    "FacadePoseEstimator",
    "estimate_camera_pose_ransac",
    "FacadeHeatmapOverlay",
    "create_facade_heatmap_overlay",
    "create_ortho_rectified_view",
    "rectify_image_to_ortho_view",
    "match_photo_to_ortho",
    "scale_photo_keypoints_to_original",
    "sample_verify_matches",
    "Facade3DBackprojector",
    "backproject_matches_to_3d",
    "mapping_to_render_assets",
    "FacadeOrthorenderer",
    "normalize_plane_params",
    "render_facade_orthographic",
    "prepare_ortho_api_payload",
    "pixel_to_uv",
    "pixel_to_world",
    "uv_to_pixel",
    "world_to_pixel",
]
