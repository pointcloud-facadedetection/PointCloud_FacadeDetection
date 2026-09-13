"""B3：ProjectLifecycleController offscreen 编排测试（mock service）。"""
import threading
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QThreadPool, SignalInstance

from fakes import Row, SignalRecorder, pump_until
from ui.controllers.project_lifecycle import ProjectLifecycleController
from ui.main_window_config import PAGE_DEFINITIONS
from utils.workers import PointCloudLoadWorker

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

    def commit_prepared_uploads(self, prepared):
        self.log.append(('overview.commit', len(prepared or [])))

    def create_load_worker(self, operation, project_uuid, *,
                           file_paths=None, directory=None, directories=None):
        """与真实 ProjectOverviewService 相同的 worker 契约：计算段在
        worker.run() 内执行，结果携带 prepared/uploaded 供 GUI 回调提交。"""
        def run(worker):
            if operation == 'upload':
                return {'operation': operation,
                        'uploaded': self.upload_files(file_paths, project_uuid),
                        'prepared': []}
            if operation == 'fls':
                dirs = list(directories or [])
                if directory:
                    dirs.append(directory)
                all_results = [self.import_fls_directory(d, project_uuid) for d in dirs]
                merged = {
                    'success': any(r.get('success') for r in all_results),
                    'message': '; '.join(
                        r.get('message', '') for r in all_results
                        if not r.get('success') and r.get('message')) or '',
                    'ply_paths': [p for r in all_results for p in r.get('ply_paths', [])],
                    'uploaded': sum(r.get('uploaded', 1) for r in all_results if r.get('success')),
                }
                return {'operation': operation,
                        'result': merged, 'results': all_results}
            raise ValueError(f'未知加载操作: {operation}')
        return PointCloudLoadWorker(run)

    def load_historical_facades(self, uuid, station_id):
        self.log.append(('overview.historical', uuid, station_id))
        return [{'id': 1}]

    def load_all_historical_facades(self, uuid):
        self.log.append(('overview.historical', uuid, None))
        return {1: [{'id': 1}]}


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
        self.prepare_threads = []
        self.commit_threads = []
        self.commit_count = 0

    def set_project(self, uuid):
        self.log.append(('station.set_project', uuid))

    def list_stations(self):
        return self.rows

    def refresh(self):
        self.log.append(('station.refresh',))

    def show_single(self, row):
        self.log.append(('station.show_single', row.id))

    def prepare_restore_view(self):
        """与真实 StationService 相同的契约：计算段只准备数据，不渲染。"""
        import threading
        self.log.append(('station.prepare_restore',))
        self.prepare_threads.append(threading.get_ident())
        if not self.rows:
            return (None, None)
        return (self.rows[0], object())

    def commit_restore_view(self, prepared):
        import threading
        self.log.append(('station.restore_view',))
        self.commit_threads.append(threading.get_ident())
        self.commit_count += 1
        if prepared is not None and prepared[0] is not None:
            self._active_station_id = prepared[0].id


class FakeOperationService:
    def __init__(self, log):
        self.log = log
        self.last_facade_results = []

    def invalidate_async_jobs(self):
        self.log.append(('op.invalidate',))

    def clear_processing_state(self):
        self.log.append(('op.clear_state',))

    def set_facade_results_for_station(self, station_id, facades):
        self.log.append(('op.set_facade_results', station_id))

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
        ('station.prepare_restore',),
        ('station.restore_view',),
        ('op.set_uuid', 'uuid-1'),
        ('overview.historical', 'uuid-1', None),
        ('op.set_facade_results', 1),
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
    assert ('station.prepare_restore',) in log
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
                                'uuid-1',
                                {'uploaded': ['a', 'b'], 'prepared': ['p1', 'p2']},
                                before_ids=set())
    # 提交段真实执行（2 个 prepared），随后刷新站点并展示新站点
    assert ('overview.commit', 2) in log
    assert ('station.refresh',) in log
    assert ('station.show_single', 1) in log
    assert ('station_panel', 1) in rec.events
    assert ('proj_list',) in rec.events
    assert ('cleared',) in rec.events
    assert controller._load_in_progress is False


def test_on_load_finished_stale_generation_ignored(env):
    controller, _, rec, log, _, _ = env
    controller.on_load_finished(999, 'upload', 'uuid-1',
                                {'uploaded': ['a'], 'prepared': ['p1']},
                                before_ids=set())
    assert rec.events == []
    assert log == []


# ---------------------------------------------------------------------------
# activate 后台化：worker 计算段离 GUI 线程，提交段恰好一次回 GUI
# ---------------------------------------------------------------------------
def _load_signal_recorder(controller):
    rec = SignalRecorder()
    controller.load_started.connect(lambda: rec.events.append(('load_started',)))
    controller.load_finished.connect(lambda: rec.events.append(('load_finished',)))
    return rec


class TestActivateBackgroundSplit:
    def test_prepare_off_gui_commit_on_gui(self, qapp):
        gui_ident = threading.get_ident()
        log = []
        quality = SimpleNamespace(active_quality_worker=None,
                                  quality_result_cache={}, quality_reports=[])
        pool = QThreadPool()
        pool.setMaxThreadCount(1)
        controller = ProjectLifecycleController(
            project_overview_service=FakeOverviewService(log),
            pointcloud_service=FakePointCloudService(log),
            station_service=FakeStationService(log),
            project_operation_service=FakeOperationService(log),
            render_service=FakeRenderService(log),
            facade_quality_controller=quality,
            context_provider=lambda: (False, None),
            load_pool=pool,
        )
        rec = _load_signal_recorder(controller)
        project = SimpleNamespace(project_id='uuid-1', name='演示',
                                  directory_path='.')

        controller.start_load('activate', 'uuid-1', project=project)
        assert controller._load_in_progress is True  # 调用立即返回，未同步阻塞
        assert pump_until(qapp, lambda: not controller._load_in_progress)

        # 机制：计算段（DB 激活/set_project/站点域准备）真实离开 GUI 线程
        station = controller.station_service
        assert len(station.prepare_threads) == 1
        assert station.prepare_threads[0] != gui_ident
        # 机制：GUI 提交段恰好执行一次，且发生在 GUI 线程
        assert station.commit_count == 1
        assert station.commit_threads == [gui_ident]
        # 机制：加载窗口信号各恰好一次（首开弹窗契约）
        assert rec.kinds().count('load_started') == 1
        assert rec.kinds().count('load_finished') == 1
        # 数据：完整激活事务真实走完（准备 → 提交 → 项目级状态）
        assert ('station.prepare_restore',) in log
        assert log.index(('station.prepare_restore',)) < \
            log.index(('station.restore_view',))
        assert ('overview.historical', 'uuid-1', None) in log
        assert controller._active_load_worker is None

    def test_second_activation_reuses_same_worker_contract(self, qapp):
        # 二开（同一控制器再次 activate）同样走 worker 且弹窗信号成对
        log = []
        quality = SimpleNamespace(active_quality_worker=None,
                                  quality_result_cache={}, quality_reports=[])
        controller = ProjectLifecycleController(
            project_overview_service=FakeOverviewService(log),
            pointcloud_service=FakePointCloudService(log),
            station_service=FakeStationService(log),
            project_operation_service=FakeOperationService(log),
            render_service=FakeRenderService(log),
            facade_quality_controller=quality,
            context_provider=lambda: (False, None),
            load_pool=None,
        )
        rec = _load_signal_recorder(controller)
        project = SimpleNamespace(project_id='uuid-1')
        controller.start_load('activate', 'uuid-1', project=project)
        controller.start_load('activate', 'uuid-1', project=project)
        assert rec.kinds() == ['load_started', 'load_finished',
                               'load_started', 'load_finished']
        assert controller.station_service.commit_count == 2

    def test_db_activate_failure_leaves_ui_untouched(self, qapp):
        log = []
        quality = SimpleNamespace(active_quality_worker=None,
                                  quality_result_cache={}, quality_reports=[])
        overview = FakeOverviewService(log)
        def failing_activate(uuid):
            raise RuntimeError('模拟 DB 激活失败')
        overview.activate_project = failing_activate
        controller = ProjectLifecycleController(
            project_overview_service=overview,
            pointcloud_service=FakePointCloudService(log),
            station_service=FakeStationService(log),
            project_operation_service=FakeOperationService(log),
            render_service=FakeRenderService(log),
            facade_quality_controller=quality,
            context_provider=lambda: (False, None),
            load_pool=None,  # 内联：信号直连，语义与同步路径等价
        )
        rec = SignalRecorder()
        controller.warning_requested.connect(
            lambda t, m: rec.events.append(('warn', t, m)))
        controller.current_project_change_requested.connect(
            lambda p: rec.events.append(('set_project', p)))
        load_rec = _load_signal_recorder(controller)
        project = SimpleNamespace(project_id='uuid-1')

        controller.start_load('activate', 'uuid-1', project=project)

        # 机制：DB 激活失败时计算段在 overview.activate 处中断——
        # set_project/站点准备/提交段都没有发生
        assert ('pc.set_project', 'uuid-1') not in log
        assert ('station.set_project', 'uuid-1') not in log
        assert ('station.prepare_restore',) not in log
        # 渲染/注册表未被改动：没有 clear_scene，没有 current_project_change
        assert ('render.clear_scene',) not in log
        assert not any(e[0] == 'set_project' for e in rec.events)
        # 统一错误路径：警告 + 状态复位 + 加载窗口成对关闭
        warns = [e for e in rec.events if e[0] == 'warn']
        assert warns and '模拟 DB 激活失败' in warns[0][2]
        assert '站点恢复失败' not in warns[0][2]
        assert load_rec.kinds() == ['load_started', 'load_finished']
        assert controller._load_in_progress is False
        assert controller._active_load_worker is None
