"""立面选中聚焦的实效测试。

断言：
1. 选中后非选中区域（含其他立面）降回默认灰，选中立面保留原色，
   AABB 线框挂出；
2. 清除选中后恢复正常配色管线（签名缓存失效）且线框摘除；
3. 切换选中对象时旧线框先摘后挂，场景里最多一个聚焦框。
"""
import numpy as np

from services.viewport_render_service import ViewportRenderService

DEFAULT_GRAY = np.array([0.75, 0.75, 0.75], dtype=np.float32)


class _FakeScene:
    def __init__(self):
        self.bbox_visible = {}


class _FakeViewport:
    def __init__(self, pos):
        self._data = {'pos': pos, 'proxy_ids': np.arange(len(pos))}
        self._scene = _FakeScene()
        self.colors = None
        self.bbox_events = []

    def get_cloud_data(self, name):
        return self._data

    def update_cloud_color(self, name, colors):
        self.colors = np.asarray(colors).copy()

    def toggle_bbox(self, name, min_bound, max_bound):
        visible = self._scene.bbox_visible.get(name, False)
        self._scene.bbox_visible[name] = not visible
        self.bbox_events.append((name, not visible))
        return not visible


def _make_service(n=10):
    pos = np.random.rand(n, 3).astype(np.float32)
    viewport = _FakeViewport(pos)
    service = ViewportRenderService(viewport, db=None)
    return service, viewport, pos


def _facades():
    return [
        {'id': 1, 'display_no': 1, 'proxy_indices': [0, 1, 2],
         'color': [0.9, 0.2, 0.2]},
        {'id': 2, 'display_no': 2, 'proxy_indices': [5, 6],
         'color': [0.2, 0.9, 0.2]},
    ]


def test_select_facade_focuses_color_and_bbox():
    service, viewport, pos = _make_service()
    facades = _facades()
    service.highlight_facades('cloud', facades)   # 先建立正常配色与缓存

    service.select_facade('cloud', 1)

    colors = viewport.colors
    # 选中立面 1 的点保留原色
    assert np.allclose(colors[[0, 1, 2]], np.array([0.9, 0.2, 0.2], dtype=np.float32))
    # 其余点（含立面 2）隐入背景色
    others = [i for i in range(len(pos)) if i not in (0, 1, 2)]
    assert np.allclose(colors[others], DEFAULT_GRAY)
    # AABB 线框已挂出
    assert viewport._scene.bbox_visible.get('facade_focus') is True


def test_clear_selection_restores_normal_colors_and_removes_bbox():
    service, viewport, _ = _make_service()
    facades = _facades()
    service.highlight_facades('cloud', facades)
    service.select_facade('cloud', 2)
    assert viewport._scene.bbox_visible.get('facade_focus') is True

    service.clear_selected_facade('cloud')

    assert viewport._scene.bbox_visible.get('facade_focus') is False
    # 正常配色恢复：两个立面各自着色，非立面点为基础灰
    colors = viewport.colors
    assert np.allclose(colors[[0, 1, 2]], np.array([0.9, 0.2, 0.2], dtype=np.float32))
    assert np.allclose(colors[[5, 6]], np.array([0.2, 0.9, 0.2], dtype=np.float32))
    assert np.allclose(colors[3], np.array([0.75, 0.75, 0.75], dtype=np.float32))


def test_switching_selection_replaces_bbox():
    service, viewport, _ = _make_service()
    service.highlight_facades('cloud', _facades())
    service.select_facade('cloud', 1)
    service.select_facade('cloud', 2)

    # 线框摘下再挂出，最终可见；颜色聚焦到立面 2
    assert viewport._scene.bbox_visible.get('facade_focus') is True
    colors = viewport.colors
    assert np.allclose(colors[[5, 6]], np.array([0.2, 0.9, 0.2], dtype=np.float32))
    assert np.allclose(colors[[0, 1, 2]], DEFAULT_GRAY)
