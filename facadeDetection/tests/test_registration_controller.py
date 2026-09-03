"""B1：RegistrationController offscreen 编排测试（mock service + 真 QThreadPool）。"""
import pytest
from PySide6.QtCore import QThreadPool, SignalInstance

from fakes import Row, SignalRecorder, pump_until
from ui.controllers.registration import RegistrationController


class FakeStationService:
    def __init__(self):
        self.rows = [Row(1), Row(2)]
        self.committed = []
        self.register_calls = []
        self.refreshed = 0

    def refresh(self):
        self.refreshed += 1

    def list_stations(self):
        return self.rows

    def register_selected(self, update_viewport=True, manual_points=None,
                          proxy_clouds=None):
        self.register_calls.append({
            'update_viewport': update_viewport,
            'manual_points': manual_points,
            'proxy_clouds': proxy_clouds,
        })
        return {'result_path': '', 'station_ids': [1, 2]}

    def commit_registration(self, payload):
        self.committed.append(payload)

    def prepare_registration_view(self, rows):
        return rows, ['cloud_t', 'cloud_s']


class FakeRenderService:
    def __init__(self):
        self.exited_pick = 0

    def get_cloud_data(self, name):
        return {'pos': [[0, 0, 0], [1, 1, 1]]}

    def registration_pick_points(self):
        return ([[0, 0, 0]], [[1, 1, 1]])

    def exit_pick_mode(self):
        self.exited_pick += 1


@pytest.fixture
def env(qapp):
    station = FakeStationService()
    render = FakeRenderService()
    rec = SignalRecorder()
    controller = RegistrationController(
        station, render, QThreadPool(), lambda: (0, 'uuid-1', False))
    controller.busy_changed.connect(lambda b: rec.events.append(('busy', b)))
    controller.status_message.connect(lambda m, t: rec.events.append(('status', m, t)))
    controller.warning_requested.connect(lambda m: rec.events.append(('warn', m)))
    controller.info_requested.connect(lambda m: rec.events.append(('info', m)))
    controller.station_panel_refresh_requested.connect(
        lambda: rec.events.append(('refresh',)))
    controller.manual_pick_requested.connect(
        lambda s, t: rec.events.append(('pick', s, t)))
    controller.manual_icp_prompt_requested.connect(
        lambda n: rec.events.append(('prompt', n)))
    return controller, station, render, rec, qapp


def test_signals_exist(env):
    controller = env[0]
    for name in ('busy_changed', 'status_message', 'warning_requested',
                 'info_requested', 'station_panel_refresh_requested',
                 'manual_pick_requested', 'manual_icp_prompt_requested'):
        assert isinstance(getattr(controller, name), SignalInstance), name


def test_requires_exactly_two_stations(env):
    controller, station, _, rec, _ = env
    station.rows = [Row(1)]
    controller.run_station_registration()
    assert ('warn', '点云配准需要恰好选择两个 PLY 站点') in rec.events
    assert controller._worker is None
    assert station.register_calls == []


def test_busy_guard_rejects_reentry(env):
    controller, _, _, rec, _ = env
    controller._worker = object()
    controller.run_station_registration()
    assert ('info', '已有配准任务正在执行，请稍候。') in rec.events


def test_auto_registration_roundtrip(env):
    controller, station, _, rec, qapp = env
    controller.run_station_registration()
    assert controller._worker is not None
    assert station.refreshed == 1
    assert rec.events[0] == ('status', '正在执行 GPS 全局坐标残差 ICP，请稍候...', 0)
    assert rec.events[1] == ('busy', False)
    assert pump_until(qapp, lambda: bool(station.committed))
    assert station.register_calls[0]['update_viewport'] is False
    assert station.register_calls[0]['manual_points'] is None
    # 成功链路顺序：commit -> 刷新面板 -> 状态栏 -> finally busy 复位
    kinds = rec.kinds()
    assert kinds.index('refresh') < kinds.index('status', kinds.index('refresh'))
    assert ('status', '点云配准完成，已显示注册合并结果。', 5000) in rec.events
    assert rec.events[-1] == ('busy', True)
    assert controller._worker is None


def test_failure_enters_manual_fallback(env):
    controller, _, _, rec, _ = env
    controller._rows = (1, 2)
    controller._on_registration_failed('ICP 未收敛')
    assert rec.events[0] == ('busy', True)
    assert ('pick', 'cloud_s', 'cloud_t') in rec.events
    assert ('status', '自动配准失败：ICP 未收敛', 5000) in rec.events


def test_pick_progress_status(env):
    controller, _, _, rec, _ = env
    controller._rows = (1, 2)
    controller._on_registration_failed('x')
    controller.handle_pick_updated(True)
    assert any(e[0] == 'status' and '配准选点：已完成 1 对' in e[1]
               for e in rec.events)


def test_manual_registration_roundtrip(env):
    controller, station, render, rec, qapp = env
    controller._rows = (1, 2)
    controller._on_registration_failed('x')
    controller.start_manual_registration()
    assert render.exited_pick == 1
    assert ('status', '正在执行人工初值代理域 ICP，请稍候...', 0) in rec.events
    assert pump_until(qapp, lambda: bool(station.committed))
    call = station.register_calls[0]
    assert call['manual_points'] is not None
    assert call['proxy_clouds'] is not None
    assert ('refresh',) in rec.events


def test_stale_generation_result_discarded(env, tmp_path):
    controller, station, _, rec, _ = env
    stale_file = tmp_path / 'registration_stale.ply'
    stale_file.write_text('x')
    payload = {
        '_project_generation': 99,  # 与当前代际 0 不符 -> 过期
        '_project_uuid': 'uuid-1',
        '_station_ids_snapshot': (1, 2),
        'result_path': str(stale_file),
    }
    controller._on_registration_finished(payload)
    assert station.committed == []
    assert not stale_file.exists()  # 过期结果文件被清理
    assert rec.events == [('busy', True)]
    assert controller._worker is None
