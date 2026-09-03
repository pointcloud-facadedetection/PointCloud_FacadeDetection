"""B3：ProjectLifecycleController offscreen 编排测试（mock service）。"""
from types import SimpleNamespace

import pytest
from PySide6.QtCore import SignalInstance

from fakes import Row, SignalRecorder
from ui.controllers.project_lifecycle import ProjectLifecycleController
from ui.main_window_config import PAGE_DEFINITIONS

OPERATION_INDEX = next(i for i, (_t, k) in enumerate(PAGE_DEFINITIONS)
                       if k == 'project_operation')


class FakeOverviewService:
    def __init__(self, log):
        self.log = log
        self.upload_result = ['f1']
        self.fls_payload = {'success': True, 'uploaded': 2}
        self.on_upload = None

    def activate_project(self, uuid):
        self.log.append(('overview.activate', uuid))

    def upload_files(self, paths, pid):
        self.log.append(('overview.upload', pid))
        if self.on_upload is not None:
            self.on_upload()
        return self.upload_result

    def import_fls_directory(self, directory, pid):
        self.log.append(('overview.fls', pid))
        return self.fls_payload

    def load_historical_facades(self, uuid, station_id):
        self.log.append(('overview.historical', uuid, station_id))
        return [{'id': 1}]


class FakePointCloudService:
    def __init__(self, log):
        self.log = log
        self.fail_set_project = False

    def set_project(self, uuid):
        self.log.append(('pc.set_project', uuid))
        if self.fail_set_project:
            raise RuntimeError('模拟站点恢复失败')

    def close_project(self):
        self.log.append(('pc.close',))


class FakeStationService:
    def __init__(self, log):
        self.log = log
        self.rows = [Row(1)]
        self._active_station_id = None

    def set_project(self, uuid):
        self.log.append(('station.set_project', uuid))

    def list_stations(self):
        return self.rows

    def refresh(self):
        self.log.append(('station.refresh',))

    def show_single(self, row):
        self.log.append(('station.show_single', row.id))

    def restore_view(self):
        self.log.append(('station.restore_view',))


class FakeOperationService:
    def __init__(self, log):
        self.log = log
        self.last_facade_results = []

    def invalidate_async_jobs(self):
        self.log.append(('op.invalidate',))

    def clear_processing_state(self):
        self.log.append(('op.clear_state',))

    def set_active_project_uuid(self, uuid):
        self.log.append(('op.set_uuid', uuid))


class FakeRenderService:
    def __init__(self, log):
        self.log = log

    def clear_scene_display(self):
        self.log.append(('render.clear_scene',))

    def clear_runtime(self):
        self.log.append(('render.clear_runtime',))

    def clear_viewport(self):
        self.log.append(('render.clear_viewport',))


@pytest.fixture
def env(qapp):
    log = []
    quality = SimpleNamespace(active_quality_worker=object(),
                              quality_result_cache={'k': 1},
                              quality_reports=[{'r': 1}])
    context = {'closing': False, 'current_id': None}
    rec = SignalRecorder()
    controller = ProjectLifecycleController(
        project_overview_service=FakeOverviewService(log),
        pointcloud_service=FakePointCloudService(log),
        station_service=FakeStationService(log),
        project_operation_service=FakeOperationService(log),
        render_service=FakeRenderService(log),
        facade_quality_controller=quality,
        context_provider=lambda: (context['closing'], context['current_id']),
    )
    controller.info_requested.connect(lambda t, m: rec.events.append(('info', t, m)))
    controller.warning_requested.connect(lambda t, m: rec.events.append(('warn', t, m)))
    controller.status_message.connect(lambda m, t: rec.events.append(('status', m, t)))
    controller.status_cleared.connect(lambda: rec.events.append(('cleared',)))
    controller.station_panel_refresh_requested.connect(
        lambda x: rec.events.append(('station_panel', x)))
    controller.project_list_refresh_requested.connect(
        lambda: rec.events.append(('proj_list',)))
    controller.facade_list_reset_requested.connect(
        lambda: rec.events.append(('facade_reset',)))
    controller.station_list_reset_requested.connect(
        lambda: rec.events.append(('station_reset',)))
    controller.current_project_change_requested.connect(
        lambda p: rec.events.append(('set_project', p)))
    controller.report_preview_refresh_requested.connect(
        lambda: rec.events.append(('report',)))
    controller.facade_results_refresh_requested.connect(
        lambda r: rec.events.append(('facade_results', r)))
    controller.page_change_requested.connect(lambda i: rec.events.append(('page', i)))
    project = SimpleNamespace(project_id='uuid-1', name='演示', directory_path='.')
    return controller, quality, rec, log, project, context


def make_controller(log, closing=False, current_id=None):
    quality = SimpleNamespace(active_quality_worker=None,
                              quality_result_cache={}, quality_reports=[])
    return ProjectLifecycleController(
        project_overview_service=FakeOverviewService(log),
        pointcloud_service=FakePointCloudService(log),
        station_service=FakeStationService(log),
        project_operation_service=FakeOperationService(log),
        render_service=FakeRenderService(log),
        facade_quality_controller=quality,
        context_provider=lambda: (closing, current_id),
    )


def test_signals_exist(env):
    controller = env[0]
    for name in ('info_requested', 'warning_requested', 'status_message',
                 'status_cleared', 'station_panel_refresh_requested',
                 'project_list_refresh_requested', 'facade_list_reset_requested',
                 'station_list_reset_requested',
                 'current_project_change_requested',
                 'report_preview_refresh_requested',
                 'facade_results_refresh_requested', 'page_change_requested'):
        assert isinstance(getattr(controller, name), SignalInstance), name


def test_activate_full_sequence(env):
    controller, _, rec, log, project, _ = env
    controller.start_load('activate', 'uuid-1', project=project)
    assert log == [
        ('overview.activate', 'uuid-1'),
        ('pc.set_project', 'uuid-1'),
        ('station.set_project', 'uuid-1'),
        ('station.restore_view',),
        ('op.set_uuid', 'uuid-1'),
        ('overview.historical', 'uuid-1', None),
    ]
    assert ('set_project', project) in rec.events
    assert ('station_panel', None) in rec.events
    assert ('facade_reset',) in rec.events
    assert ('report',) in rec.events
    assert ('facade_results', [{'id': 1}]) in rec.events
    assert ('page', OPERATION_INDEX) in rec.events
    assert ('proj_list',) in rec.events and ('cleared',) in rec.events
    assert ('status', '正在加载点云，请稍候...', 0) in rec.events
    assert controller._load_in_progress is False
    assert controller.project_operation_service.last_facade_results == [{'id': 1}]


def test_activate_empty_stations_branch(env):
    controller, _, rec, log, project, _ = env
    controller.station_service.rows = []
    controller.start_load('activate', 'uuid-1', project=project)
    assert ('render.clear_scene',) in log
    assert ('station.restore_view',) not in log
    assert ('set_project', project) in rec.events
    assert ('status', '项目已打开，但未发现可用 PLY 站点。', 5000) in rec.events
    assert not any(e[0] == 'page' for e in rec.events)


def test_set_project_failure_wrapped_and_warned(env):
    controller, _, rec, log, project, _ = env
    controller.pointcloud_service.fail_set_project = True
    controller.start_load('activate', 'uuid-1', project=project)
    assert ('render.clear_scene',) in log
    assert ('status', '点云加载失败', 5000) in rec.events
    warns = [e for e in rec.events if e[0] == 'warn']
    assert warns and warns[0][1] == '点云加载'
    assert '站点恢复失败' in warns[0][2]
    assert controller._load_in_progress is False


def test_closing_guard_blocks_start(qapp):
    log = []
    controller = make_controller(log, closing=True)
    controller.start_load('activate', 'uuid-1',
                          project=SimpleNamespace(project_id='uuid-1'))
    assert log == []


def test_busy_guard_rejects_reentry(env):
    controller, _, rec, log, project, _ = env
    controller._load_in_progress = True
    controller.start_load('activate', 'uuid-1', project=project)
    assert ('info', '点云加载', '已有加载任务正在执行，请稍候。') in rec.events
    assert log == []


def test_upload_new_station_flow(env):
    controller, _, rec, log, _, _ = env
    controller.station_service.rows = [Row(1), Row(2)]
    controller.project_overview_service.on_upload = (
        lambda: controller.station_service.rows.append(Row(3)))
    controller.start_load('upload', 'uuid-1', file_paths=['a'])
    assert ('station.show_single', 3) in log
    assert ('station_panel', 3) in rec.events
    assert ('proj_list',) in rec.events


def test_upload_empty_result_warns(env):
    controller, _, rec, _, _, _ = env
    controller.project_overview_service.upload_result = []
    controller.start_load('upload', 'uuid-1', file_paths=[])
    assert ('warn', '直接上传文件', '未成功绑定任何点云文件。') in rec.events


def test_fls_failure_warns(env):
    controller, _, rec, _, _, _ = env
    controller.project_overview_service.fls_payload = {
        'success': False, 'message': '目录为空'}
    controller.start_load('fls', 'uuid-1', directory='/x')
    assert ('warn', 'FLS 导入', '目录为空') in rec.events


def test_prepare_activation_disposes_and_bumps_generation(env):
    controller, quality, rec, log, _, _ = env
    gen_before = controller.project_generation
    controller.prepare_project_activation('uuid-2')
    assert controller.project_generation == gen_before + 1
    assert quality.active_quality_worker is None
    assert quality.quality_result_cache == {}
    assert quality.quality_reports == []
    for item in (('op.invalidate',), ('op.clear_state',),
                 ('render.clear_runtime',), ('pc.close',),
                 ('render.clear_viewport',)):
        assert item in log
    assert ('facade_reset',) in rec.events
    assert ('station_reset',) in rec.events


def test_stale_generation_load_failed_ignored(env):
    controller, _, rec, _, _, _ = env
    controller.on_load_failed(999, '过期')
    assert rec.events == []


def test_on_load_finished_upload_branch(env):
    controller, _, rec, log, _, _ = env
    controller.on_load_finished(controller.project_generation, 'upload',
                                'uuid-1', None, {'uploaded': ['a', 'b']})
    assert ('status', '已增量添加 2 个文件，已有站点资源未重新加载。', 5000) \
        in rec.events
    assert ('station.refresh',) in log
    assert ('station_panel', None) in rec.events


def test_on_load_finished_stale_generation_ignored(env):
    controller, _, rec, log, _, _ = env
    controller.on_load_finished(999, 'upload', 'uuid-1', None,
                                {'uploaded': ['a']})
    assert rec.events == []
    assert log == []
