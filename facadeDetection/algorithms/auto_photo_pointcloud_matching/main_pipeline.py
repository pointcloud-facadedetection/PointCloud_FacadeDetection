"""全自动 2D-3D 对齐与热力图贴合流水线（与 matcher.match_photo_to_facade 对齐）。"""

from __future__ import annotations

from .matcher import match_photo_to_facade


def run_auto_matching_pipeline(
    photo_path,
    facade,
    points,
    image_width,
    image_height,
    colors=None,
    horizontal_fov_deg=60.0,
    resolution_m=0.02,
    resize_max=1024,
    min_match_confidence=0.1,
    min_correspondences=6,
    reprojection_error_px=5.0,
    pnp_confidence=0.999,
):
    """
    参考 main_pipeline 五步流程的统一入口：

    1. 正射渲染 + index_map
    2. SuperPoint + LightGlue 2D 匹配
    3. 正射像素 → 3D 回溯
    4. PnP RANSAC
    5. （热力图叠加由 heatmap_overlay 模块单独调用）
    """
    return match_photo_to_facade(
        photo_path=photo_path,
        facade=facade,
        points=points,
        image_width=image_width,
        image_height=image_height,
        colors=colors,
        horizontal_fov_deg=horizontal_fov_deg,
        resolution_m=resolution_m,
        resize_max=resize_max,
        min_match_confidence=min_match_confidence,
        min_correspondences=min_correspondences,
        reprojection_error_px=reprojection_error_px,
        pnp_confidence=pnp_confidence,
    )


__all__ = ['run_auto_matching_pipeline']
