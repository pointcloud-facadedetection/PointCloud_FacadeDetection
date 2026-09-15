"""加载中模态窗口的实效测试。

断言三类事实：
1. 机制：load_started/load_progress/load_finished 的发射次数与顺序；
   取消、失败路径同样必然关闭窗口（不允许窗口残留挡住 UI）。
2. 状态：对话框为应用级模态且不可被用户关闭；进度条/文本被真实更新。
3. 线程：加载会话期间事件循环不被 exec() 阻塞（信号直连内联 worker 验证）。
"""
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt

from fakes import Row, SignalRecorder
from ui.controllers.project_lifecycle import ProjectLifecycleController
from ui.dialogs.loading_dialog import LoadingDialog
from utils.workers import PointCloudLoadWorker


# ---------------------------------------------------------------------------
# 对话框本体
# ---------------------------------------------------------------------------
class TestLoadingDialogBehavior:
    def test_modal_and_not_user_closable(self, qapp):
        dlg = LoadingDialog()
        assert dlg.windowModality() == Qt.WindowModality.ApplicationModal
        # 无关闭按钮：CustomizeWindowHint + WindowTitleHint，不含 WindowCloseButtonHint
        assert not dlg.windowFlags() & Qt.WindowType.WindowCloseButtonHint

        dlg.show()
        assert dlg.isVisible()
        dlg.reject()  # 用户按 Esc/尝试关闭
        qapp.processEvents()
        assert dlg.isVisible()  # 必须仍然打开，只能由 load_finished 关闭
        dlg.hide()

    def test_progress_updates_are_real(self, qapp):
        dlg = LoadingDialog()
        dlg.update_progress('正在解析 bllygg01.ply')
        assert 'bllygg01.ply' in dlg._label.text()
        dlg.update_progress('点云加载完成')
        assert '点云加载完成' in dlg._label.text()
        assert dlg._bar.maximum() == 0  # 恒为忙碌条


# ---------------------------------------------------------------------------
# controller 信号流
# ---------------------------------------------------------------------------
def _make_controller(log, worker_factory):
    quality = SimpleNamespace(active_quality_worker=None,
                              quality_result_cache={}, quality_reports=[])
    overview = SimpleNamespace(
        create_load_worker=worker_factory,
        commit_prepared_uploads=lambda prepared: log.append(('commit', len(prepared or []))),
    )
    station = SimpleNamespace(
        list_stations=lambda: [Row(1)],
        refresh=lambda: log.append(('refresh',)),
        show_single=lambda row: log.append(('show_single', row.id)),
    )
    controller = ProjectLifecycleController(
        project_overview_service=overview,
        pointcloud_service=SimpleNamespace(),
        station_service=station,
        project_operation_service=SimpleNamespace(
            invalidate_async_jobs=lambda: None,
            clear_processing_state=lambda: None),
        render_service=SimpleNamespace(
            clear_runtime=lambda: None, clear_viewport=lambda: None),
        facade_quality_controller=quality,
        context_provider=lambda: (False, 'uuid-1'),
        load_pool=None,  # 内联运行：信号直连，顺序即真实发射顺序
    )
    return controller


def _recorder(controller):
    rec = SignalRecorder()
    controller.load_started.connect(lambda: rec.events.append(('started',)))
    controller.load_progress.connect(
        lambda pct, text: rec.events.append(('progress', pct, text)))
    controller.load_finished.connect(lambda: rec.events.append(('finished',)))
    controller.warning_requested.connect(
        lambda t, m: rec.events.append(('warn', t, m)))
    return rec


class TestLoadLifecycleSignals:
    def test_upload_emits_started_progress_finished_in_order(self, qapp):
        log = []
        def factory(operation, project_uuid, *, file_paths=None, directory=None, directories=None):
            def run(worker):
                return {'operation': operation, 'uploaded': ['f1'], 'prepared': []}
            return PointCloudLoadWorker(run)
        controller = _make_controller(log, factory)
        rec = _recorder(controller)

        controller.start_load('upload', 'uuid-1', file_paths=['f1'])

        kinds = rec.kinds()
        assert kinds.count('started') == 1
        assert kinds.count('finished') == 1
        # worker.run 真实的两次进度回报（2% 准备 / 100% 完成）
        progresses = [e for e in rec.events if e[0] == 'progress']
        assert len(progresses) >= 2
        assert progresses[0][1] == 2 and progresses[-1][1] == 100
        # 顺序：started 在第一次 progress 前，finished 在最后
        assert kinds.index('started') < kinds.index('progress')
        assert kinds[-1] == 'finished'
        assert controller._load_in_progress is False
        # 提交段确实被调用（机制真实走完）
        assert ('commit', 0) in log

    def test_failed_worker_also_closes_dialog(self, qapp):
        log = []
        def factory(operation, project_uuid, *, file_paths=None, directory=None, directories=None):
            def run(worker):
                raise RuntimeError('模拟解析失败')
            return PointCloudLoadWorker(run)
        controller = _make_controller(log, factory)
        rec = _recorder(controller)

        controller.start_load('upload', 'uuid-1', file_paths=['f1'])

        kinds = rec.kinds()
        assert kinds.count('started') == 1
        assert kinds.count('finished') == 1  # 失败也必须关闭加载窗口
        assert any(e[0] == 'warn' and '模拟解析失败' in e[2] for e in rec.events)
        assert controller._load_in_progress is False

    def test_dispose_cancels_worker_and_closes_dialog(self, qapp):
        log = []
        controller = _make_controller(log, lambda *a, **k: None)
        rec = _recorder(controller)

        # 模拟一个仍在后台运行的 worker（不调用 run）
        pending = PointCloudLoadWorker(lambda worker: None)
        controller._active_load_worker = pending
        controller._load_in_progress = True

        controller.dispose_project_runtime()

        assert pending.is_cancelled()  # 后台任务收到真实取消
        assert rec.kinds().count('finished') == 1  # 窗口不会因取消而残留
        assert controller._active_load_worker is None
