"""触控捏合缩放的实效测试（含反测，防止无条件绿灯）。

链路：Qt PinchGesture → _GestureFilter → viewport._zoom_by_factor。
正交模式下应改 parallel_scale，透视模式应 Dolly。

断言（正反两侧）：
1. scaleFactor=2.0 → parallel_scale 减半（放大）；
2. scaleFactor=0.5 → parallel_scale 加倍（缩小）——方向错误必然挂；
3. scaleFactor=1.0 → 不变（无操作不得改相机）；
4. 非捏合手势（PanGesture）→ 不变且不抛异常——过滤逻辑必须挑食。
"""
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest
from PySide6.QtWidgets import QGestureEvent, QPanGesture, QPinchGesture


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


def _pinch_event(scale_factor):
    """构造捏合手势事件。手势对象必须随事件一并返回保活：
    PySide6 不接管手势所有权，局部对象被 GC 后事件里只剩悬垂指针。"""
    gesture = QPinchGesture()
    gesture.setScaleFactor(scale_factor)
    return QGestureEvent([gesture]), gesture


def _parallel_scale(vp):
    return float(vp._plotter.camera.parallel_scale)


def test_pinch_zoom_in_halves_parallel_scale(viewport):
    before = _parallel_scale(viewport)
    event, _keep = _pinch_event(2.0)
    consumed = viewport._gesture_filter.eventFilter(viewport._widget, event)
    assert consumed is True
    assert _parallel_scale(viewport) == pytest.approx(before / 2.0)


def test_pinch_zoom_out_doubles_parallel_scale(viewport):
    """反测：方向必须正确——缩小手势必须放大 scale，而不是减半。"""
    before = _parallel_scale(viewport)
    event, _keep = _pinch_event(0.5)
    viewport._gesture_filter.eventFilter(viewport._widget, event)
    assert _parallel_scale(viewport) == pytest.approx(before * 2.0)


def test_neutral_factor_changes_nothing(viewport):
    """反测：scaleFactor=1.0 不得动相机。"""
    before = _parallel_scale(viewport)
    event, _keep = _pinch_event(1.0)
    viewport._gesture_filter.eventFilter(viewport._widget, event)
    assert _parallel_scale(viewport) == pytest.approx(before)


def test_non_pinch_gesture_ignored(viewport):
    """反测：Pan 手势不得触发缩放，也不得抛异常。"""
    before = _parallel_scale(viewport)
    pan = QGestureEvent([QPanGesture()])
    consumed = viewport._gesture_filter.eventFilter(viewport._widget, pan)
    assert _parallel_scale(viewport) == pytest.approx(before)


def test_single_touch_uses_selected_navigation_mode(viewport):
    from PySide6.QtCore import QEvent, QPointF
    from types import SimpleNamespace

    calls = []
    viewport._turntable_rotate = lambda dx, dy: calls.append(('rotate', dx, dy))
    viewport._camera_pan = lambda dx, dy: calls.append(('pan', dx, dy))

    def touch(event_type, x, y):
        point = SimpleNamespace(position=lambda: QPointF(x, y))
        return SimpleNamespace(
            type=lambda: event_type,
            touchPoints=lambda: [point])

    viewport._gesture_filter.eventFilter(
        viewport._widget, touch(QEvent.Type.TouchBegin, 10, 10))
    viewport._gesture_filter.eventFilter(
        viewport._widget, touch(QEvent.Type.TouchUpdate, 20, 25))
    assert calls == [('rotate', 10.0, 15.0)]

    viewport.set_touch_navigation_mode('pan')
    viewport._gesture_filter.eventFilter(
        viewport._widget, touch(QEvent.Type.TouchBegin, 20, 25))
    viewport._gesture_filter.eventFilter(
        viewport._widget, touch(QEvent.Type.TouchUpdate, 35, 30))
    assert calls[-1] == ('pan', 15.0, 5.0)


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
        assert window.btn_operation_touch_mode.text() == '旋转'
        assert window.btn_review_touch_mode.text() == '旋转'
        assert window.lbl_review_ruler_example.text()
        window.btn_operation_touch_mode.click()
        assert window.viewport._touch_navigation_mode == 'pan'
        assert window.btn_review_touch_mode.text() == '平移'

        window.btn_review_touch_mode.click()
        assert window.viewport._touch_navigation_mode == 'rotate'
        window._on_review_tool_activated('ruler')
        assert window.viewport._touch_navigation_mode == 'pan'
        window._on_review_tool_deactivated()
        assert window.viewport._touch_navigation_mode == 'rotate'

        window.viewport._plotter.enable_parallel_projection()
        before = _parallel_scale(window.viewport)
        window.review_zoom_slider.setValue(200)
        assert window.operation_zoom_slider.value() == 200
        assert _parallel_scale(window.viewport) == pytest.approx(before / 2)
        window.btn_review_reset_view.click()
        assert window.review_zoom_slider.value() == 100
        assert window.operation_zoom_slider.value() == 100
    finally:
        window.close()


def test_viewport_accepts_native_touch_events(viewport):
    from PySide6.QtCore import Qt
    assert viewport._widget.testAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents)
