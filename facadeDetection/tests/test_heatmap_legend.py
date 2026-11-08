import inspect

import numpy as np

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
    assert _HEATMAP_MAX_SIZE == 4096
    signature = inspect.signature(transparent_png_uv_rect)
    assert signature.parameters['max_size'].default == 4096
