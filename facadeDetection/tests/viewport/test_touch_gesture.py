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
