"""原生触摸导航、双指缩放及触控滑块回归测试。"""
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtWidgets import QApplication, QGestureEvent, QPanGesture, QStyle, QStyleOptionSlider


@pytest.fixture
def viewport(qapp):
    from view3d.pyvista_viewport import PyVistaViewport
    vp = PyVistaViewport()
    vp.add_cloud('t', np.random.rand(100, 3).astype(np.float32))
    yield vp
    try:
        vp.destroy()
    except Exception:
        pass


def _point(point_id, x, y, state=Qt.TouchPointState.TouchPointStationary):
    return SimpleNamespace(
        id=lambda: point_id,
        position=lambda: QPointF(x, y),
        state=lambda: state,
    )


def _touch(event_type, points):
    return SimpleNamespace(
        type=lambda: event_type,
        touchPoints=lambda: points,
        accept=lambda: None,
    )


def _send(viewport, event_type, specs):
    points = [_point(*spec) for spec in specs]
    return viewport._gesture_filter.eventFilter(
        viewport._widget, _touch(event_type, points))


def _parallel_scale(vp):
    return float(vp._plotter.camera.parallel_scale)


def test_native_pinch_zoom_in_halves_parallel_scale_without_rotation(viewport):
    before = _parallel_scale(viewport)
    calls = []
    viewport._turntable_rotate = lambda dx, dy: calls.append(('rotate', dx, dy))
    viewport._camera_pan = lambda dx, dy: calls.append(('pan', dx, dy))

    _send(viewport, QEvent.Type.TouchBegin, [(1, 10, 10), (2, 110, 10)])
    _send(viewport, QEvent.Type.TouchUpdate, [(1, 10, 10), (2, 210, 10)])

    assert _parallel_scale(viewport) == pytest.approx(before / 2.0)
    assert calls == []


def test_native_pinch_zoom_out_doubles_parallel_scale(viewport):
    before = _parallel_scale(viewport)
    _send(viewport, QEvent.Type.TouchBegin, [(1, 10, 10), (2, 210, 10)])
    _send(viewport, QEvent.Type.TouchUpdate, [(1, 10, 10), (2, 110, 10)])
    assert _parallel_scale(viewport) == pytest.approx(before * 2.0)


def test_native_pinch_with_unchanged_distance_changes_nothing(viewport):
    before = _parallel_scale(viewport)
    _send(viewport, QEvent.Type.TouchBegin, [(1, 10, 10), (2, 110, 10)])
    _send(viewport, QEvent.Type.TouchUpdate, [(1, 20, 20), (2, 120, 20)])
    assert _parallel_scale(viewport) == pytest.approx(before)


def test_non_touch_gesture_is_ignored(viewport):
    before = _parallel_scale(viewport)
    consumed = viewport._gesture_filter.eventFilter(
        viewport._widget, QGestureEvent([QPanGesture()]))
    assert consumed is False
    assert _parallel_scale(viewport) == pytest.approx(before)


def test_single_touch_uses_selected_navigation_mode(viewport):
    calls = []
    viewport._turntable_rotate = lambda dx, dy: calls.append(('rotate', dx, dy))
    viewport._camera_pan = lambda dx, dy: calls.append(('pan', dx, dy))

    _send(viewport, QEvent.Type.TouchBegin, [(1, 10, 10)])
    _send(viewport, QEvent.Type.TouchUpdate, [(1, 20, 25)])
    assert calls == [('rotate', 10.0, -15.0)]

    viewport.set_touch_navigation_mode('pan')
    _send(viewport, QEvent.Type.TouchEnd, [])
    _send(viewport, QEvent.Type.TouchBegin, [(1, 20, 25)])
    _send(viewport, QEvent.Type.TouchUpdate, [(1, 35, 30)])
    assert calls[-1] == ('pan', 15.0, -5.0)


def test_single_touch_pan_really_translates_camera_without_rotation(viewport):
    viewport.set_touch_navigation_mode('pan')
    camera = viewport._plotter.camera
    before_position = np.asarray(camera.position, dtype=float)
    before_focal = np.asarray(camera.focal_point, dtype=float)
    before_offset = before_position - before_focal

    _send(viewport, QEvent.Type.TouchBegin, [(1, 20, 25)])
    _send(viewport, QEvent.Type.TouchUpdate, [(1, 65, 55)])

    after_position = np.asarray(camera.position, dtype=float)
    after_focal = np.asarray(camera.focal_point, dtype=float)
    assert np.linalg.norm(after_focal - before_focal) > 1e-6
    assert np.allclose(after_position - before_position,
                       after_focal - before_focal)
    assert np.allclose(after_position - after_focal, before_offset)


def test_zoom_percent_uses_one_reference_scale(viewport):
    viewport._plotter.enable_parallel_projection()
    before = _parallel_scale(viewport)
    viewport.set_zoom_percent(200)
    assert _parallel_scale(viewport) == pytest.approx(before / 2)
    viewport.set_zoom_percent(50)
    assert _parallel_scale(viewport) == pytest.approx(before * 2)
    viewport.set_zoom_percent(25)
    assert _parallel_scale(viewport) == pytest.approx(before * 4)


def test_touch_controls_sync_and_ruler_restores_mode(qapp):
    from ui.main_window import MainWindow

    window = MainWindow()
    try:
        assert not hasattr(window, 'btn_review_touch_mode')
        assert not hasattr(window, 'review_zoom_slider')
        assert not hasattr(window, 'btn_review_reset_view')
        assert window.btn_operation_touch_mode.text() == '旋转'
        assert window.lbl_review_ruler_example.text()
        window.btn_operation_touch_mode.click()
        assert window.viewport._touch_navigation_mode == 'pan'
        assert window.btn_operation_touch_mode.text() == '平移'

        window.btn_operation_touch_mode.click()
        assert window.viewport._touch_navigation_mode == 'rotate'
        window._on_review_tool_activated('ruler')
        assert window.viewport._touch_navigation_mode == 'pan'
        window._on_review_tool_deactivated()
        assert window.viewport._touch_navigation_mode == 'rotate'

        window.viewport._plotter.enable_parallel_projection()
        before = _parallel_scale(window.viewport)
        window.operation_zoom_slider.setValue(200)
        assert _parallel_scale(window.viewport) == pytest.approx(before / 2)
        window.btn_operation_reset_view.click()
        assert window.operation_zoom_slider.value() == 100
    finally:
        window.close()


def test_zoom_slider_has_large_touch_target(qapp):
    from ui.main_window import MainWindow

    window = MainWindow()
    try:
        slider = window.operation_zoom_slider
        slider.ensurePolished()
        option = QStyleOptionSlider()
        slider.initStyleOption(option)
        handle = slider.style().subControlRect(
            QStyle.ComplexControl.CC_Slider, option,
            QStyle.SubControl.SC_SliderHandle, slider)
        assert slider.height() >= 48
        assert handle.width() >= 40
        assert handle.height() >= 40
    finally:
        window.close()


def test_viewport_uses_application_native_touch_filter(viewport):
    assert viewport._widget.testAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents)
    assert viewport._gesture_app is QApplication.instance()


def test_pinch_locks_single_touch_until_all_fingers_lift(viewport):
    calls = []
    viewport._turntable_rotate = lambda dx, dy: calls.append(('rotate', dx, dy))
    viewport._camera_pan = lambda dx, dy: calls.append(('pan', dx, dy))

    _send(viewport, QEvent.Type.TouchBegin, [(1, 10, 10), (2, 110, 10)])
    assert viewport._touch_pinching is True
    _send(viewport, QEvent.Type.TouchUpdate, [(1, 30, 30)])
    _send(viewport, QEvent.Type.TouchUpdate, [(1, 50, 50)])
    assert calls == [], '双指后剩余单指也不得叠加导航'

    _send(viewport, QEvent.Type.TouchEnd, [
        (1, 50, 50, Qt.TouchPointState.TouchPointReleased)])
    assert viewport._touch_pinching is False
    _send(viewport, QEvent.Type.TouchBegin, [(1, 10, 10)])
    _send(viewport, QEvent.Type.TouchUpdate, [(1, 20, 20)])
    assert calls, '全部手指抬起后应恢复单指导航'


def test_synthesized_mouse_cannot_reenter_vtk_navigation(viewport):
    viewport._drag_left = (10, 10)
    viewport._touch_active = True
    event = SimpleNamespace(
        type=lambda: QEvent.Type.MouseButtonPress,
        source=lambda: Qt.MouseEventSource.MouseEventSynthesizedBySystem,
        accept=lambda: None,
    )
    consumed = viewport._gesture_filter.eventFilter(viewport._widget, event)
    assert consumed is True
    assert viewport._drag_left is None
    assert viewport._drag_right is None


def test_real_mouse_still_uses_vtk_navigation(viewport):
    event = SimpleNamespace(
        type=lambda: QEvent.Type.MouseButtonPress,
        source=lambda: Qt.MouseEventSource.MouseEventNotSynthesized,
        accept=lambda: None,
    )
    consumed = viewport._gesture_filter.eventFilter(viewport._widget, event)
    assert consumed is False
