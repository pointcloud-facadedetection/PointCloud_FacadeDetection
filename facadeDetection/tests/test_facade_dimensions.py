from types import SimpleNamespace

import numpy as np
import pytest

from algorithms.facade.facade_geometry import facade_dimension_spec


CORNERS = [[0, 0, 0], [4, 0, 0], [4, 0, 3], [0, 0, 3]]


def test_dimension_spec_uses_actual_corner_distances():
    spec = facade_dimension_spec(CORNERS)
    assert spec['width_m'] == pytest.approx(4.0)
    assert spec['height_m'] == pytest.approx(3.0)
    assert spec['labels'][0]['text'] == 'W 4.000 m'
    assert spec['labels'][1]['text'] == 'H 3.000 m'
    for start, end in (spec['width_line'], spec['height_line']):
        assert np.linalg.norm(np.asarray(end) - start) > 0
    assert spec['corners'] == CORNERS
    assert min(point[2] for point in spec['width_line']) < 0
    assert max(point[0] for point in spec['height_line']) < 0


def test_dimension_offset_is_bounded_for_tall_narrow_facades():
    spec = facade_dimension_spec(
        [[0, 0, 0], [3, 0, 0], [3, 0, 86], [0, 0, 86]])
    assert 0 < abs(spec['width_line'][0][2]) <= 0.5
    assert spec['height_m'] == pytest.approx(86)


def test_dimension_spec_rejects_degenerate_geometry():
    with pytest.raises(ValueError):
        facade_dimension_spec([[0, 0, 0]] * 4)


def test_dimension_actors_cache_labels_and_clear(qapp):
    import vtk
    from view3d.pyvista_viewport import PyVistaViewport

    viewport = PyVistaViewport()
    try:
        result = viewport.show_facade_dimensions(CORNERS, cache_key=('facade', 1))
        assert result == pytest.approx((4.0, 3.0))
        actors = list(viewport._dimension_actors)
        labels = [actor for actor in actors if isinstance(actor, vtk.vtkTextActor)]
        assert len(actors) == 12
        assert [actor.GetInput() for actor in labels] == ['W 4.000 m', 'H 3.000 m']
        assert all(actor.GetTextProperty().GetBackgroundOpacity() > 0.9
                   for actor in labels)
        viewport._on_render_start()
        assert labels[0].GetPositionCoordinate().GetCoordinateSystemAsString() == 'Display'
        original_project = viewport.project_points
        viewport.project_points = lambda _points: (
            np.array([[-10000, 10000], [10000, -10000]]), np.array([True, True]))
        viewport._update_dimension_label_positions()
        width = max(viewport._plotter.renderer.GetSize()[0],
                    viewport._widget.width())
        height = max(viewport._plotter.renderer.GetSize()[1],
                     viewport._widget.height())
        for actor in labels:
            x, y = actor.GetPosition()
            assert 0 <= x <= width and 0 <= y <= height
        viewport.project_points = original_project

        viewport.clear_facade_dimensions()
        assert viewport._dimension_actors == []
        assert all(not actor.GetVisibility() for actor in actors)
        viewport.show_facade_dimensions(CORNERS, cache_key=('facade', 1))
        assert viewport._dimension_actors == actors

        enlarged = [[0, 0, 0], [5, 0, 0], [5, 0, 3], [0, 0, 3]]
        viewport.show_facade_dimensions(enlarged, cache_key=('facade', 2))
        assert any(isinstance(actor, vtk.vtkTextActor) and
                   actor.GetInput() == 'W 5.000 m'
                   for actor in viewport._dimension_actors)
        assert all(not actor.GetVisibility() for actor in actors)

        viewport.clear()
        assert viewport._dimension_actors == []
        assert viewport._dimension_groups == {}
    finally:
        viewport.destroy()


def test_dimensions_are_cleared_with_review_selection(qapp):
    from PySide6.QtWidgets import QLabel
    from ui.pages.inspection_review_page import InspectionReviewPageMixin

    class Viewport:
        def __init__(self):
            self.clear_calls = 0

        def show_facade_dimensions(self, corners, cache_key=None):
            assert len(corners) == 4
            assert cache_key[0] == 'dimensions'
            return 4.0, 3.0

        def clear_facade_dimensions(self):
            self.clear_calls += 1

        def clear_heatmap_quad(self):
            pass

    harness = InspectionReviewPageMixin()
    harness.viewport = Viewport()
    harness.current_project = SimpleNamespace(project_id='project')
    harness.lbl_review_dimensions = QLabel()
    harness._review_camera_snapshot = None
    facade = {'id': 1}
    geometry = {'corners_3d': CORNERS}
    harness._review_present_dimensions(facade, geometry)
    assert '横向 4.000 m' in harness.lbl_review_dimensions.text()
    assert '纵向 3.000 m' in harness.lbl_review_dimensions.text()
    harness._review_clear_presentation()
    assert harness.viewport.clear_calls == 1
    assert harness.lbl_review_dimensions.text() == ''
