import numpy as np

from algorithms.facade import interactive_ruler
from fakes import pump_until
from ui.controllers.viewport_tools import ViewportToolsController
from view3d.pyvista_viewport import PyVistaViewport


PLANE = np.array([1.0, 0.0, 0.0, 0.0])
GEOMETRY = {
    'plane_model': PLANE.tolist(),
    'origin': [0.0, 0.0, 0.0],
    'u_axis': [0.0, 1.0, 0.0],
    'v_axis': [0.0, 0.0, 1.0],
    'uv_bounds': [0.0, 0.0, 4.0, 3.0],
}


class FakeViewport:
    def __init__(self):
        self.rulers = []
        self.markers = []

    def clear_ruler_overlay(self):
        self.rulers.clear()

    def update_pick_markers(self, src_points=None, **_kwargs):
        self.markers = list(src_points or [])

    def clear_pick_markers(self):
        self.markers = []

    def show_ruler_overlay(self, render, exceeded=False):
        self.rulers.append((render, exceeded))

    def clear_tool_visuals(self):
        self.rulers.clear()
        self.markers = []

    def exit_pick_mode(self):
        pass


def _wall():
    y, z = np.meshgrid(np.arange(0, 4, 0.05), np.arange(0, 3, 0.05))
    return np.column_stack((np.zeros(y.size), y.ravel(), z.ravel()))


def test_two_click_ruler_uses_start_and_places_in_background(qapp):
    viewport = FakeViewport()
    controller = ViewportToolsController(
        viewport, object(), lambda: 'cloud', lambda: {'id': 1},
        lambda: {'ruler_length': 2.0, 'threshold_mm': 4.0})
    controller._tool = controller.TOOL_RULER
    controller._plane_model = PLANE
    controller._plane_geometry = GEOMETRY
    controller._ruler_base = {
        'points': _wall(), 'plane_origin': np.zeros(3), 'revision': '1'}

    start = np.array([0.0, 1.0, 1.5])
    controller._on_ruler_pick({'point': start})
    np.testing.assert_array_equal(controller._ruler_start, start)
    controller._on_ruler_pick({'point': [0.0, 3.0, 1.5]})
    assert controller._placement_worker is not None
    assert pump_until(qapp, lambda: bool(viewport.rulers))

    body = np.asarray(viewport.rulers[0][0]['body_line'])
    direction = np.array([0.0, 1.0, 0.0])
    assert float((body[0] - start) @ direction) == 0.0
    assert float((body[1] - start) @ direction) == 2.0
    controller.cancel()


def test_tool_renderer_is_above_heatmap_renderer():
    import vtk

    viewport = PyVistaViewport.__new__(PyVistaViewport)
    render_window = vtk.vtkRenderWindow()
    main = vtk.vtkRenderer()

    class Plotter:
        pass

    viewport._plotter = Plotter()
    viewport._plotter.render_window = render_window
    viewport._plotter.renderer = main
    viewport._overlay_renderer = None
    viewport._tool_overlay_renderer = None

    heatmap = viewport._get_overlay_renderer()
    tools = viewport._get_tool_overlay_renderer()
    assert heatmap is not tools
    assert heatmap.GetLayer() == 1
    assert tools.GetLayer() == 2
    assert render_window.GetNumberOfLayers() == 3
    assert tools.GetActiveCamera() is main.GetActiveCamera()

    actor = vtk.vtkActor()
    viewport._to_overlay(actor, tool=True)
    assert tools.GetActors().GetNumberOfItems() == 1
    assert heatmap.GetActors().GetNumberOfItems() == 0


def test_color_update_preserves_display_proxy_lookup():
    import pyvista as pv

    viewport = PyVistaViewport.__new__(PyVistaViewport)
    poly = pv.PolyData(np.zeros((2, 3)))
    poly.point_data['colors'] = np.zeros((2, 3), dtype=np.uint8)
    lookup = {10: 0, 20: 1}
    viewport.point_data = {
        'cloud': {
            'pos': np.zeros((2, 3), dtype=np.float32),
            'color': np.zeros((2, 3), dtype=np.float32),
            'proxy_ids': np.array([10, 20]),
            'display_proxy_lookup': lookup,
        }
    }
    viewport._polys = {'cloud': poly}
    viewport._actors = {}
    viewport._hidden_rows = {}
    viewport._render = lambda: None
    viewport.update_cloud_color('cloud', np.ones((2, 3), dtype=np.float32))

    assert viewport.point_data['cloud']['display_proxy_lookup'] is lookup
    np.testing.assert_array_equal(poly.point_data['colors'], 255)
