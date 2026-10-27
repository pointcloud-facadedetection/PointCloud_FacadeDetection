"""复核页呈现链路：选中立面零重算（不刷色/不藏点/不画 AABB，高亮由
热力贴图边框承担）、恒灰着色跳过、操作页行为不受 review 分支影响。
"""
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest

from services.viewport_render_service import ViewportRenderService, _BASE_GRAY


class _FakeViewport:
    def __init__(self, n=10):
        self.data = {'c': {
            'pos': np.random.default_rng(1).random((n, 3)).astype(np.float32),
            'color': np.random.default_rng(2).random((n, 3)).astype(np.float32),
            'proxy_ids': np.arange(n, dtype=np.int32),
        }}
        self.hidden = {}
        self.color_updates = 0
        self.bbox_calls = 0

    def get_cloud_data(self, name):
        return self.data.get(name)

    def get_cloud_names(self):
        return list(self.data)

    def update_cloud_color(self, name, colors):
        self.data[name]['color'] = np.asarray(colors, dtype=np.float32).copy()
        self.color_updates += 1

    def set_hidden_rows(self, name, rows=None):
        if rows is None or len(rows) == 0:
            self.hidden.pop(name, None)
        else:
            self.hidden[name] = np.asarray(rows)

    def toggle_bbox(self, *args, **kwargs):
        self.bbox_calls += 1


def _service():
    viewport = _FakeViewport()
    service = ViewportRenderService(viewport, db=None)
    facades = [{'id': 1, 'proxy_indices': [0, 1, 2]},
               {'id': 2, 'proxy_indices': [5, 6]}]
    return service, viewport, facades


class TestReviewModeSelection:
    def test_select_is_pure_noop_in_review_mode(self):
        service, viewport, facades = _service()
        service.set_review_mode(True)
        service.highlight_facades('c', facades)
        assert viewport.color_updates == 1
        assert np.allclose(viewport.data['c']['color'], _BASE_GRAY)

        service.select_facade('c', 1)
        assert viewport.hidden == {}, '复核页选中不得隐藏点（透过贴图可见灰云）'
        assert viewport.color_updates == 1, '选中不得重算点云颜色'
        assert viewport.bbox_calls == 0, '复核页选中不画 AABB，高亮由贴图边框承担'

        service.select_facade('c', 2)
        assert viewport.hidden == {}
        assert viewport.color_updates == 1
        assert viewport.bbox_calls == 0

        service.clear_selected_facade('c')
        assert viewport.hidden == {}
        assert viewport.color_updates == 1, '取消选中不得重算点云颜色'

    def test_gray_repaint_skipped_when_already_gray(self):
        service, viewport, facades = _service()
        service.set_review_mode(True)
        service.highlight_facades('c', facades)
        service.highlight_facades('c', facades)
        assert viewport.color_updates == 1, '已是恒灰时重复着色应跳过上传'

    def test_operation_mode_untouched(self):
        service, viewport, facades = _service()
        service.highlight_facades('c', facades)
        baseline = viewport.color_updates
        service.select_facade('c', 1)
        assert viewport.hidden == {}, '操作页选中不得隐藏点'
        assert viewport.color_updates > baseline, '操作页选中聚焦应刷色'
        after_select = viewport.color_updates
        service.clear_selected_facade('c')
        assert viewport.color_updates > after_select, '操作页取消选中应恢复配色'


@pytest.fixture
def viewport(qapp):
    from view3d.pyvista_viewport import PyVistaViewport
    vp = PyVistaViewport()
    vp.add_cloud('t', np.random.default_rng(3).random((100, 3)).astype(np.float32))
    yield vp
    try:
        vp.destroy()
    except Exception:
        pass


class TestHiddenRowsViewport:
    def test_mask_reduces_poly_and_restores(self, viewport):
        assert viewport._polys['t'].n_points == 100
        viewport.set_hidden_rows('t', np.arange(10))
        assert viewport._polys['t'].n_points == 90
        assert len(viewport.point_data['t']['pos']) == 100, '全量数组不变'

        # 隐藏态下整云配色仍按全量长度下发，上传时自动套用掩码
        viewport.update_cloud_color('t', np.full((100, 3), 0.5, np.float32))
        assert viewport._polys['t'].n_points == 90
        assert viewport._polys['t'].point_data['colors'].shape[0] == 90

        viewport.set_hidden_rows('t', None)
        assert viewport._polys['t'].n_points == 100

    def test_same_count_rehide_updates_content(self, viewport):
        viewport.set_hidden_rows('t', np.arange(10))
        hidden_pos = viewport._polys['t'].points.copy()
        viewport.set_hidden_rows('t', np.arange(10, 20))
        assert viewport._polys['t'].n_points == 90
        assert not np.allclose(viewport._polys['t'].points, hidden_pos)

    def test_update_cloud_points_resets_stale_mask(self, viewport):
        viewport.set_hidden_rows('t', np.arange(10))
        viewport.update_cloud_points(
            't', np.random.default_rng(4).random((50, 3)).astype(np.float32))
        assert 't' not in viewport._hidden_rows
        assert viewport._polys['t'].n_points == 50


class TestHeatmapQuadCache:
    """热力贴图 actor 组缓存：同 key 重贴只切可见性，不重建 actor。"""

    @staticmethod
    def _quad_args():
        image = np.zeros((8, 8, 4), dtype=np.uint8)
        image[..., 3] = 255
        return dict(image=image, origin=[0, 0, 0], u_axis=[0, 1, 0],
                    v_axis=[0, 0, 1], uv_bounds=(0, 0, 4, 3),
                    normal=[1, 0, 0])

    def test_same_key_reuses_actors(self, viewport):
        viewport.show_heatmap_quad(**self._quad_args(), cache_key=('f1', 'm1'))
        actors_first = list(viewport._heatmap_quad_actors)
        assert actors_first
        assert all(a.GetVisibility() for a in actors_first)

        viewport.clear_heatmap_quad()
        assert viewport._heatmap_quad_actors == []
        assert all(not a.GetVisibility() for a in actors_first), \
            'clear 只隐藏不销毁'

        viewport.show_heatmap_quad(**self._quad_args(), cache_key=('f1', 'm1'))
        assert viewport._heatmap_quad_actors == actors_first, \
            '同 key 重贴必须复用同一批 actor（不重传纹理）'
        assert all(a.GetVisibility() for a in actors_first)

    def test_different_key_rebuilds(self, viewport):
        viewport.show_heatmap_quad(**self._quad_args(), cache_key=('f1', 'm1'))
        actors_a = list(viewport._heatmap_quad_actors)
        viewport.show_heatmap_quad(**self._quad_args(), cache_key=('f1', 'm2'))
        actors_b = list(viewport._heatmap_quad_actors)
        assert actors_b and actors_b != actors_a
        assert all(not a.GetVisibility() for a in actors_a)
        assert all(a.GetVisibility() for a in actors_b)

    def test_lru_eviction(self, viewport):
        for i in range(viewport._HEATMAP_CACHE_MAX + 2):
            viewport.show_heatmap_quad(**self._quad_args(),
                                       cache_key=('f', i))
        assert len(viewport._heatmap_quad_groups) == viewport._HEATMAP_CACHE_MAX

    def test_frame_only_cached_too(self, viewport):
        corners = [[0, 0, 0], [0, 4, 0], [0, 4, 3], [0, 0, 3]]
        viewport.show_heatmap_frame(corners, cache_key=('frame', 1))
        actors_first = list(viewport._heatmap_quad_actors)
        assert actors_first
        viewport.clear_heatmap_quad()
        viewport.show_heatmap_frame(corners, cache_key=('frame', 1))
        assert viewport._heatmap_quad_actors == actors_first


def test_clear_tool_visuals_keeps_heatmap_visible(viewport):
    viewport.show_heatmap_quad(
        **TestHeatmapQuadCache._quad_args(), cache_key=('f1', 'm1'))
    actors = list(viewport._heatmap_quad_actors)
    viewport.clear_tool_visuals()
    assert viewport._heatmap_quad_actors == actors
    assert all(actor.GetVisibility() for actor in actors)


def test_pyvistaqt_auto_render_timer_is_disabled(viewport):
    assert not viewport._widget.render_timer.isActive()


def test_review_gray_actor_preserves_color_buffer(viewport, qapp):
    viewport.point_data['t']['proxy_ids'] = np.arange(100, dtype=np.int32)
    service = ViewportRenderService(viewport, db=None)
    facades = [{'id': 1, 'proxy_indices': [0, 1, 2]}]
    service.highlight_facades('t', facades)
    qapp.processEvents()
    colored = viewport.point_data['t']['color'].copy()

    service.set_review_mode(True)
    service.highlight_facades('t', facades)
    actor = viewport._actors['t']
    assert actor.GetMapper().GetScalarVisibility() == 0
    np.testing.assert_array_equal(viewport.point_data['t']['color'], colored)

    service.set_review_mode(False)
    service.highlight_facades('t', facades)
    qapp.processEvents()
    assert actor.GetMapper().GetScalarVisibility() == 1
    np.testing.assert_array_equal(viewport.point_data['t']['color'], colored)
