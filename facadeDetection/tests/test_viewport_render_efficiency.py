import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from time import monotonic

import numpy as np
import pytest
from PySide6.QtTest import QTest


@pytest.fixture
def viewport(qapp):
    from view3d.pyvista_viewport import PyVistaViewport
    vp = PyVistaViewport()
    yield vp
    try:
        vp.destroy()
    except Exception:
        pass


def test_multisample_rendering_is_disabled(viewport):
    assert viewport._widget.render_window.GetMultiSamples() == 0


def test_display_lod_preserves_full_business_data_and_color_mapping(viewport):
    viewport._MAX_DISPLAY_POINTS = 1_000
    count = 5_000
    positions = np.arange(count * 3, dtype=np.float32).reshape(-1, 3)
    colors = np.linspace(0, 1, count * 3, dtype=np.float32).reshape(-1, 3)
    viewport.add_cloud('large', positions, colors)

    expected = np.arange(1_000, dtype=np.int64) * (count - 1) // 999
    assert len(viewport.point_data['large']['pos']) == count
    assert len(viewport.point_data['large']['color']) == count
    assert viewport._polys['large'].n_points == 1_000
    assert np.allclose(viewport._polys['large'].points, positions[expected])
    assert np.array_equal(
        viewport._polys['large'].point_data['colors'],
        (colors[expected] * 255).astype(np.uint8))

    updated = colors[::-1].copy()
    viewport.update_cloud_color('large', updated)
    assert viewport._polys['large'].n_points == 1_000
    assert np.array_equal(
        viewport._polys['large'].point_data['colors'],
        (updated[expected] * 255).astype(np.uint8))
    assert len(viewport.point_data['large']['color']) == count


def test_default_cloud_uses_efficient_point_size(viewport):
    viewport.add_cloud('cloud', np.zeros((10, 3), dtype=np.float32))
    assert viewport._actors['cloud'].prop.point_size == 2


def test_navigation_render_requests_are_coalesced_and_rate_limited(
        viewport, qapp):
    updates = []
    viewport._plotter.update = lambda: updates.append(monotonic())
    viewport._last_render_time = monotonic()
    viewport._touch_active = True

    for _ in range(10):
        viewport._render()
    assert viewport._render_pending is True
    qapp.processEvents()
    assert updates == []

    QTest.qWait(120)
    assert len(updates) == 1
    assert viewport._render_pending is False


def test_non_navigation_render_is_not_delayed(viewport, qapp):
    updates = []
    viewport._plotter.update = lambda: updates.append(1)
    viewport._last_render_time = monotonic()
    viewport._touch_active = False
    viewport._drag_left = None
    viewport._drag_right = None

    viewport._render()
    qapp.processEvents()
    assert updates == [1]
