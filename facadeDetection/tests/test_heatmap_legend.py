import inspect

import numpy as np
import pytest

from algorithms.facade.interactive_ruler import transparent_png_uv_rect
from services.heatmap_renderer import (
    FacadeHeatmapTripletRenderer,
    _HEATMAP_MAX_SIZE,
)
from services.heatmap_spec import heatmap_spec
from ui.widgets.heatmap_legend import HeatmapLegendWidget


def test_legend_metadata_and_export_payload(tmp_path):
    from services.result_export_service import ResultExportService
    metadata = FacadeHeatmapTripletRenderer._legend_metadata(
        {'vmin': 0.004, 'vmax': 0.012},
        heatmap_spec('global_plane_flatness_area'))
    assert metadata['limit_mm'] == 4.0
    assert metadata['vmax_mm'] == 12.0

    class Renderer:
        def render(self, *_args, **_kwargs):
            image = np.zeros((2, 2, 4), dtype=np.uint8)
            return {'overlay': image, 'heatmap_grid': image, 'photo': image,
                    'transparent': image, 'legend': metadata}

        def fit_report_image(self, image):
            return image

    service = ResultExportService()
    service._renderer = Renderer()
    service._safe_imwrite = lambda _path, _image: True
    exported = service.export_all_heatmaps(
        tmp_path, 1, np.zeros((1, 3)), None, {
            'plane_model': [1, 0, 0, 0],
            'quality_comparison': {'methods': {
                method: {
                    metric: {'windows': [{}]}
                    for metric in ('flatness', 'verticality')
                }
                for method in ('global_plane', 'ruler')
            }},
        })
    assert exported['global_plane_flatness_area']['legend'] == metadata


def test_review_legend_uses_exported_values_and_original_clears(qapp):
    from ui.pages.inspection_review_page import InspectionReviewPageMixin
    harness = InspectionReviewPageMixin()
    harness.heatmap_legend = HeatmapLegendWidget()
    harness._review_heatmap_mode = 'global_plane_flatness_area'
    quality = {'__export_context': {'heatmaps': {
        'global_plane_flatness_area': {'legend': {
            'limit_mm': 4.0, 'vmax_mm': 12.0,
            'method': 'global_plane', 'metric': 'flatness',
        }},
    }}}
    harness._review_update_heatmap_legend(quality)
    assert harness.heatmap_legend.metadata['limit_mm'] == 4.0
    assert harness.heatmap_legend.metadata['vmax_mm'] == 12.0
    assert not harness.heatmap_legend.grab().isNull()

    harness._review_heatmap_mode = harness.REVIEW_ORIGINAL_MODE
    harness._review_update_heatmap_legend(quality)
    assert harness.heatmap_legend.metadata is None


def test_legacy_quality_shows_threshold_without_invented_max(qapp):
    from ui.pages.inspection_review_page import InspectionReviewPageMixin
    harness = InspectionReviewPageMixin()
    harness.heatmap_legend = HeatmapLegendWidget()
    harness._review_heatmap_mode = 'global_plane_flatness_area'
    harness._review_update_heatmap_legend({'parameters': {'flatness_limit_mm': 4.0}})
    assert harness.heatmap_legend.metadata['limit_mm'] == 4.0
    assert 'vmax_mm' not in harness.heatmap_legend.metadata


def test_alpha_edge_is_antialiased_without_changing_defect_colors():
    image = np.zeros((6, 6, 4), dtype=np.uint8)
    image[2:4, 2:4] = [255, 0, 0, 255]
    result = FacadeHeatmapTripletRenderer._smooth_rgba_edges(image)
    assert result.shape == image.shape
    assert result[2, 2, 3] == 255
    assert np.any((result[:, :, 3] > 0) & (result[:, :, 3] < 255))
    assert np.all(result[:, :, 1] == 0)
    assert np.all(result[:, :, 2] == 0)
    assert result[2, 2, 0] == 255


def test_raster_limit_matches_review_uv_reconstruction():
    assert _HEATMAP_MAX_SIZE == 8192
    signature = inspect.signature(transparent_png_uv_rect)
    assert signature.parameters['max_size'].default == 8192


def test_global_samples_include_signed_pass_values_from_quality_domain():
    points = np.array([
        [-0.002, 0.0, 0.0],
        [0.000, 1.0, 0.0],
        [0.006, 2.0, 0.0],
    ], dtype=np.float32)
    quality = {
        '__global_indices': np.arange(3),
        'facade_geometry': {'plane_model': [1.0, 0.0, 0.0, 0.0]},
        'thresholds': {'flatness_limit_mm': 4.0},
    }
    centers, values, _colors = FacadeHeatmapTripletRenderer._prepare_global_plane_point_samples(
        points, [], heatmap_spec('global_plane_flatness_area'), quality)
    assert np.allclose(centers, points)
    assert np.allclose(values, [-2.0, 0.0, 6.0])
    assert np.any(np.abs(values) <= 4.0), '连续场必须包含阈值内样本'


def test_continuous_signed_raster_has_smooth_values_and_preserves_holes():
    from algorithms.facade.projection import rasterize_facade
    from services.heatmap_spec import bipolar_colormap

    y, z = np.meshgrid(np.arange(9, dtype=float), np.arange(9, dtype=float))
    points = np.column_stack([
        np.linspace(-0.006, 0.006, y.size), y.ravel(), z.ravel()])
    values = points[:, 0].copy()
    raster = rasterize_facade(
        points, None, [1.0, 0.0, 0.0, 0.0], values, 0.004,
        pixel_size=0.001, base_points=points, physical_cell_m=2.0,
        continuous_signed=True, continuous_colormap=bipolar_colormap)
    overlay = raster['overlay_rgba']
    occupied = overlay[:, :, 3] > 0
    signed = raster['pixel_defect_values']
    pass_pixels = occupied & (np.abs(signed) <= 0.004)
    defect_pixels = occupied & (np.abs(signed) > 0.004)
    assert np.any(pass_pixels), '阈值内区域必须可见'
    assert np.any(defect_pixels), '超限区域必须可见'
    assert np.all(overlay[:, :, 3][pass_pixels] >= 80)
    assert np.all(overlay[:, :, 3][defect_pixels] >= 140)
    assert len(np.unique(overlay[:, :, :3][occupied].reshape(-1, 3), axis=0)) >= 4

    corners = np.array([
        [-0.006, 0.0, 0.0], [0.006, 8.0, 0.0],
        [0.000, 0.0, 8.0], [0.000, 8.0, 8.0],
    ])
    sparse = rasterize_facade(
        corners, None, [1.0, 0.0, 0.0, 0.0], corners[:, 0], 0.004,
        pixel_size=1.0, base_points=corners, physical_cell_m=1.0,
        uv_range=(0.0, 0.0, 8.0, 8.0), continuous_signed=True,
        continuous_colormap=bipolar_colormap)
    center_alpha = sparse['overlay_rgba'][4, 4, 3]
    assert center_alpha == 0, '无采样的大空洞不得插值填色'


def test_global_vmax_uses_robust_range_instead_of_single_outlier():
    values = np.array([5.0] * 95 + [10.0, 12.0, 15.0, 20.0, 85.845])
    robust = FacadeHeatmapTripletRenderer._vmax_m(values, 8.0, robust=True)
    actual = FacadeHeatmapTripletRenderer._vmax_m(values, 8.0, robust=False)
    assert robust < actual
    assert robust < 0.085845
    assert robust >= 0.0096
    assert actual == pytest.approx(0.085845)


def test_bipolar_midrange_is_not_fully_saturated():
    from services.heatmap_spec import bipolar_colormap
    middle = bipolar_colormap([-12.0, 12.0], 8.0, scale_mm=16.0)
    warm_end = bipolar_colormap([-24.0], 8.0, scale_mm=16.0)[0]
    cold_end = bipolar_colormap([24.0], 8.0, scale_mm=16.0)[0]
    assert not np.allclose(middle[0], warm_end)
    assert not np.allclose(middle[1], cold_end)
    assert middle[0, 1] > warm_end[1]
    assert middle[1, 0] > cold_end[0]


def test_pass_band_has_signed_gradient_and_explicit_threshold_colors():
    from services.heatmap_spec import bipolar_colormap
    colors = bipolar_colormap([-8.0, -4.0, 0.0, 4.0, 8.0], 8.0, scale_mm=16.0)
    assert not np.allclose(colors[1], colors[3]), '合格区不能整片同色'
    assert np.allclose(colors[2], [158 / 255, 178 / 255, 158 / 255])
    assert np.allclose(colors[0], [0.95, 0.65, 0.12])
    assert np.allclose(colors[4], [0.10, 0.65, 0.85])


def test_sparse_samples_keep_visibility_and_threshold_contour():
    from algorithms.facade.projection import _continuous_signed_overlay
    from services.heatmap_spec import bipolar_colormap

    values = np.linspace(-0.006, 0.006, 81)
    overlay, _smooth = _continuous_signed_overlay(
        values.astype(np.float32), np.arange(81, dtype=np.int64),
        np.ones((9, 9), dtype=bool), 0.004, 0.02, 0.01,
        bipolar_colormap)
    visible = overlay[:, :, 3] > 0
    assert visible.any()
    assert np.all(overlay[:, :, 3][visible] >= 100), '稀疏样本不得产生接近透明的条纹'
    boundary = np.all(overlay[:, :, :3] == [40, 45, 50], axis=2) & visible
    assert int(boundary.sum()) > 0, '±阈值位置必须绘制独立分界线'
    assert np.all(overlay[:, :, 3][boundary] == 255)
