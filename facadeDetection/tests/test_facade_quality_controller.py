"""B2：FacadeQualityController offscreen 编排测试（mock service + 真 QThreadPool）。

Storage 在 fixture 中打桩，测试不会创建真实项目目录。
"""
import pytest
from PySide6.QtCore import QThreadPool, SignalInstance

import ui.controllers.facade_quality as fq
from fakes import SignalRecorder, pump_until
from ui.controllers.facade_quality import FacadeQualityController


class FakeDataset:
    dataset_id = 'ds-1'
    revision = 7


class FakeFacadeService:
    def __init__(self):
        self.index_service = object()
        self.commits = []
        self.compute_calls = 0
        self.raise_error = False

    def get_dataset(self, cloud):
        return FakeDataset()

    def compute_quality(self, cloud_name, facade, **kwargs):
        self.compute_calls += 1
        if self.raise_error:
            raise RuntimeError('模拟计算异常')
        return {'ok': True,
                'overall': {'candidate_window_count': 5,
                            'quality_valid_window_count': 3},
                'intervals': []}

    def commit_quality_success(self, project_uuid, facade_id, quality, **kwargs):
        self.commits.append((project_uuid, facade_id, quality, kwargs))


class FakeRenderFacade:
    def __init__(self):
        self.highlights = []
        self.quality_report_calls = []

    def highlight_facades(self, cloud, results):
        self.highlights.append((cloud, list(results)))

    def compatible_quality_reports(self, cloud, facades, index_service=None):
        return [('compat',)] if facades else []

    def quality_reports(self, cloud, facades, index_service=None, heatmap_mode=None):
        self.quality_report_calls.append(heatmap_mode)


class FakeRenderService:
    def get_cloud_names(self):
        return ['cloud1']

    def facade_color_for(self, f, no=None):
        return (1.0, 0.0, 0.0)


class FakePointCloudService:
    def resolve_processing_cloud(self):
        return 'cloud1'


class FakeOperationService:
    def __init__(self):
        self.last_facade_results = []


class FakeStationService:
    _active_station_id = None


@pytest.fixture
def env(qapp, monkeypatch, tmp_path):
    # Storage 打桩：结果目录指向 tmp_path，不触碰真实项目目录。
    monkeypatch.setattr(fq, 'Storage', type('FakeStorage', (), {
        'ensure_project_dirs': staticmethod(lambda uuid: {'results': str(tmp_path)}),
    }))
    facade_service = FakeFacadeService()
    render_facade = FakeRenderFacade()
    operation = FakeOperationService()
    rec = SignalRecorder()
    context = {'uuid': 'uuid-1', 'generation': 0}
    controller = FacadeQualityController(
        facade_service=facade_service,
        render_facade=render_facade,
        render_service=FakeRenderService(),
        pointcloud_service=FakePointCloudService(),
        project_operation_service=operation,
        station_service=FakeStationService(),
        pool=QThreadPool(),
        context_provider=lambda: (context['uuid'], context['generation']),
        profile_provider=lambda: None,
        grid_size_provider=lambda: 0.05,
    )
    controller.status_message.connect(lambda m, t: rec.events.append(('status', m, t)))
    controller.status_cleared.connect(lambda: rec.events.append(('cleared',)))
    controller.warning_requested.connect(lambda m: rec.events.append(('warn', m)))
    controller.info_requested.connect(lambda m: rec.events.append(('info', m)))
    controller.report_preview_refresh_requested.connect(
        lambda: rec.events.append(('report',)))
    controller.heatmap_button_refresh_requested.connect(
        lambda: rec.events.append(('heatmap',)))
    controller.show_dialog_requested.connect(
        lambda cl, f, q: rec.events.append(('dialog', cl)))
    return controller, facade_service, render_facade, operation, rec, qapp, context


def test_signals_exist(env):
    controller = env[0]
    for name in ('status_message', 'status_cleared', 'warning_requested',
                 'info_requested', 'report_preview_refresh_requested',
                 'heatmap_button_refresh_requested', 'show_dialog_requested'):
        assert isinstance(getattr(controller, name), SignalInstance), name


def test_process_facade_results_clears_cache_highlights_dedupes(env):
    controller, _, render_facade, _, _, _, _ = env
    controller.quality_result_cache[('k',)] = {'old': 1}
    f1 = {'id': 1, 'display_no': 1, 'preview_status': 'x', 'point_count': 10}
    f1b = {'id': 1, 'preview_status': 'y', 'extra': 5}
    f2 = {'id': 2, 'display_no': 2}
    out = controller.process_facade_results([f1, f1b, f2])
    assert controller.quality_result_cache == {}
    assert render_facade.highlights == [('cloud1', [f1, f1b, f2])]
    # 按 id 去重合并，preview_status 不覆盖旧值
    assert len(out) == 2
    assert out[0]['extra'] == 5 and out[0]['preview_status'] == 'x'
    controller.set_latest_results(out)
    assert env[3].last_facade_results is out


def test_evaluate_positive_roundtrip(env):
    controller, facade_service, _, operation, rec, qapp, _ = env
    operation.last_facade_results = [{'id': 1, 'display_no': 1}]
    controller.evaluate_facade({'id': 1, 'display_no': 1})
    assert controller.active_quality_worker is not None
    assert pump_until(qapp, lambda: bool(facade_service.commits))
    uuid, fid, quality, kwargs = facade_service.commits[0]
    assert (uuid, fid) == ('uuid-1', 1)
    assert kwargs['display_no'] == 1 and kwargs['dataset_revision'] == 7
    assert operation.last_facade_results[0]['quality_status'] == 'complete'
    assert len(controller.quality_reports) == 1
    assert len(controller.quality_result_cache) == 1
    assert pump_until(qapp, lambda: 'dialog' in rec.kinds())
    kinds = rec.kinds()
    # 成功链路：clearMessage -> commit -> heatmap 刷新 -> report 刷新 -> 延迟弹窗
    assert kinds.index('cleared') < kinds.index('heatmap') < kinds.index('report')
    assert not any(k in ('warn', 'info') for k in kinds)


def test_cache_hit_skips_compute(env):
    controller, facade_service, _, operation, rec, qapp, _ = env
    operation.last_facade_results = [{'id': 1, 'display_no': 1}]
    controller.evaluate_facade({'id': 1, 'display_no': 1})
    assert pump_until(qapp, lambda: bool(facade_service.commits))
    rec.clear()
    # 无 quality_status 的同 id 立面 -> 命中缓存，不再计算
    controller.evaluate_facade({'id': 1, 'display_no': 1})
    assert ('status', '已命中质量结果缓存', 3000) in rec.events
    assert 'dialog' in rec.kinds()
    assert facade_service.compute_calls == 1


def test_compute_exception_emits_info_flow(env):
    controller, facade_service, _, operation, rec, qapp, _ = env
    facade_service.raise_error = True
    operation.last_facade_results = [{'id': 2, 'display_no': 2}]
    controller.evaluate_facade({'id': 2, 'display_no': 2})
    assert pump_until(qapp, lambda: 'info' in rec.kinds())
    infos = [e for e in rec.events if e[0] == 'info']
    assert '立面 #2 质量评估结果' in infos[0][1]
    assert pump_until(qapp, lambda: 'dialog' in rec.kinds())
    assert not any(e[0] == 'warn' for e in rec.events)


def test_stale_token_result_discarded(env):
    controller, facade_service, _, operation, rec, _, _ = env
    operation.last_facade_results = [{'id': 1, 'display_no': 1}]
    controller._on_quality_finished(999, 'cloud1',
                                    {'id': 1, 'display_no': 1},
                                    {'ok': True})
    assert rec.events == []
    assert facade_service.commits == []
    assert controller.quality_result_cache == {}
    assert controller.quality_reports == []


def test_stale_project_context_not_saved(env):
    controller, facade_service, _, operation, rec, qapp, context = env
    operation.last_facade_results = [{'id': 1, 'display_no': 1}]
    controller.evaluate_facade({'id': 1, 'display_no': 1})
    # worker 完成后项目已切换 -> 过期身份，不持久化
    context['uuid'] = 'uuid-other'
    assert pump_until(qapp, lambda: facade_service.compute_calls == 1)
    assert pump_until(
        qapp,
        lambda: ('status', '质量结果已过期，当前站点已变化，未保存。', 5000)
        in rec.events)
    assert facade_service.commits == []
    assert controller.active_quality_worker is None


def test_no_project_warns_and_skips(env):
    controller, facade_service, _, _, rec, _, context = env
    context['uuid'] = None
    controller.evaluate_facade({'id': 1, 'display_no': 1})
    assert ('warn', '请先选择项目。') in rec.events
    assert controller.active_quality_worker is None
    assert facade_service.compute_calls == 0


def test_historical_complete_reopens_dialog(env):
    controller, facade_service, _, _, rec, _, _ = env
    controller.evaluate_facade({'id': 9, 'display_no': 9,
                                'quality_status': 'complete',
                                'quality_report': {'ok': True}})
    assert 'dialog' in rec.kinds()
    assert facade_service.compute_calls == 0


def test_on_quality_failed_token_semantics(env):
    controller, _, _, _, rec, _, _ = env
    controller.on_quality_failed(999, '过期错误')
    assert rec.events == []
    controller.on_quality_failed(controller._quality_request_token, '模拟失败')
    assert ('warn', '质量计算失败：模拟失败') in rec.events
    assert ('status', '质量计算失败', 0) in rec.events
    assert ('report',) in rec.events
    assert controller.active_quality_worker is None


def test_toggle_heatmap_flips_mode_and_renders(env):
    controller, _, render_facade, operation, rec, _, _ = env
    operation.last_facade_results = [{'id': 1}]
    controller.toggle_heatmap_display()
    assert controller.heatmap_mode == 'verticality'
    assert render_facade.quality_report_calls == ['verticality']
    assert ('heatmap',) in rec.events


def test_toggle_heatmap_without_compatible_results(env):
    controller, _, render_facade, operation, rec, _, _ = env
    operation.last_facade_results = []
    controller.toggle_heatmap_display()
    assert controller.heatmap_mode == 'flatness'  # 未翻转
    assert render_facade.quality_report_calls == []
    assert ('heatmap',) in rec.events
