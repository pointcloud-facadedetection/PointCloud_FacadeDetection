"""进度反馈基础件的实效测试。

分三层断言，与既有 test_loading_dialog.py 的风格保持一致：

1. 机制：ThrottledReporter 的"高频写入 → 低频刷新"确实成立，
   且终态必刷、无新数据时不空刷。
2. 兼容：LoadingDialog 收敛为子类后，历史接口
   （``_bar`` / ``_label`` / ``update_progress`` / ``reject``）
   行为逐条不变，避免既有测试与调用方被破坏。
3. 扩展：TaskProgressDialog 的不定量模式、取消信号与
   finish() 三条收尾路径（成功 / 失败 / 取消）都真实生效。
"""
import pytest
from PySide6.QtCore import Qt

from ui.dialogs.loading_dialog import LoadingDialog
from ui.dialogs.task_progress_dialog import TaskProgressDialog
from utils.throttled_reporter import ThrottledReporter


# ---------------------------------------------------------------------------
# ThrottledReporter
# ---------------------------------------------------------------------------
class TestThrottledReporter:
    def test_high_frequency_writes_merge_into_one_flush(self, qapp):
        """1000 次写入只应产生 1 次刷新 —— 这是"不假死"的核心保证。"""
        reporter = ThrottledReporter(interval_ms=5000)
        reporter.set_auto_start(False)  # 手动控制，避免测试依赖真实 5 秒
        emitted = []
        reporter.flushed.connect(lambda p, t: emitted.append((p, t)))

        for i in range(1000):
            reporter.report(i % 101, f'步骤 {i}')
        assert emitted == []          # 节流期内一次都没刷
        assert reporter.pending is True

        assert reporter.flush() is True
        assert len(emitted) == 1
        # 刷的是最后一次的值，中间态被合并丢弃
        assert emitted[0][1] == '步骤 999'
        assert reporter.pending is False

    def test_no_new_data_does_not_refresh(self, qapp):
        """无新写入时定时器不应重复刷新（避免无意义重绘）。"""
        reporter = ThrottledReporter(interval_ms=5000)
        reporter.set_auto_start(False)
        emitted = []
        reporter.flushed.connect(lambda p, t: emitted.append(p))

        assert reporter.flush() is False
        reporter.report(10, 'a')
        assert reporter.flush() is True
        assert reporter.flush() is False
        assert len(emitted) == 1

    def test_flush_now_bypasses_throttle_and_stops(self, qapp):
        """终态刷新必须立即生效并停表，否则进度条会停在中间值。"""
        reporter = ThrottledReporter(interval_ms=5000)
        reporter.set_auto_start(False)
        emitted = []
        reporter.flushed.connect(lambda p, t: emitted.append((p, t)))

        reporter.flush_now(100, '完成')
        assert emitted == [(100, '完成')]
        assert reporter.is_active() is False

    def test_percent_is_clamped_and_text_is_optional(self, qapp):
        reporter = ThrottledReporter(interval_ms=5000)
        reporter.set_auto_start(False)
        reporter.report(150, '超上限')
        assert reporter.snapshot[0] == 100
        reporter.report(-20)
        assert reporter.snapshot[0] == 0
        assert reporter.snapshot[1] == '超上限'  # 空文本不清空上一次文案
        reporter.report(None, '仅换文案')
        assert reporter.snapshot[0] == 0

    def test_reset_clears_buffered_state(self, qapp):
        reporter = ThrottledReporter(interval_ms=5000)
        reporter.report(42, '半途')
        reporter.reset()
        assert reporter.pending is False
        assert reporter.snapshot == (0, '')
        assert reporter.is_active() is False


# ---------------------------------------------------------------------------
# LoadingDialog 向后兼容（历史契约不可破坏）
# ---------------------------------------------------------------------------
class TestLoadingDialogCompatibility:
    def test_still_modal_and_not_user_closable(self, qapp):
        dlg = LoadingDialog()
        assert dlg.windowModality() == Qt.WindowModality.ApplicationModal
        assert not dlg.windowFlags() & Qt.WindowType.WindowCloseButtonHint
        dlg.show()
        dlg.reject()
        qapp.processEvents()
        assert dlg.isVisible()
        dlg.hide()

    def test_update_progress_semantics_unchanged(self, qapp):
        dlg = LoadingDialog()
        dlg.update_progress(45, '正在解析 bllygg01.ply')
        assert dlg._bar.value() == 45
        assert 'bllygg01.ply' in dlg._label.text()
        dlg.update_progress(100, '点云加载完成')
        assert dlg._bar.value() == 100
        dlg.update_progress(150, '超出上限应截断')
        assert dlg._bar.value() == 100

    def test_default_copy_is_preserved(self, qapp):
        dlg = LoadingDialog()
        assert dlg.windowTitle() == '点云加载'
        assert '正在加载点云' in dlg._label.text()


# ---------------------------------------------------------------------------
# TaskProgressDialog 新增能力
# ---------------------------------------------------------------------------
class TestTaskProgressDialog:
    def test_indeterminate_mode_survives_then_restores(self, qapp):
        """不定量阶段无百分比；一旦收到真实百分比自动恢复定量显示。"""
        dlg = TaskProgressDialog()
        dlg.start('点云去噪', determinate=False)
        assert dlg._bar.maximum() == 0
        assert '处理中' in dlg._status_label.text()

        dlg.update_progress(37, '去噪中')
        assert dlg._bar.maximum() == 100
        assert dlg._bar.value() == 37
        assert '37%' in dlg._status_label.text()

    def test_report_progress_is_throttled_until_flush(self, qapp):
        dlg = TaskProgressDialog(interval_ms=5000)
        dlg.report_progress(80, '立面提取中')
        assert dlg.progress_percent == 0        # 仍在缓存里，未上屏
        assert '立面提取中' not in dlg.message
        dlg.flush_progress()
        assert dlg.progress_percent == 80
        assert '立面提取中' in dlg.message

    def test_cancel_button_only_when_cancellable(self, qapp):
        plain = TaskProgressDialog(cancellable=False)
        assert plain._btn_cancel.isVisible() is False

        dlg = TaskProgressDialog(cancellable=True)
        dlg.show()
        assert dlg._btn_cancel.isVisible() is True
        seen = []
        dlg.cancel_requested.connect(lambda: seen.append(1))
        dlg._btn_cancel.click()
        assert seen == [1]
        assert dlg._btn_cancel.isEnabled() is False   # 防重复点击
        dlg.hide()

    def test_reject_becomes_cancel_when_cancellable(self, qapp):
        dlg = TaskProgressDialog(cancellable=True)
        seen = []
        dlg.cancel_requested.connect(lambda: seen.append(1))
        dlg.reject()                    # 模拟 Esc
        assert seen == [1]

        blocked = TaskProgressDialog(cancellable=False)
        blocked.show()
        blocked.reject()                # 不可取消：窗口必须留存
        qapp.processEvents()
        assert blocked.isVisible()
        blocked.hide()

    @pytest.mark.parametrize('success', [True, False])
    def test_finish_always_closes_and_flushes(self, qapp, success):
        """成功与失败都必须走完收尾：刷进度、禁按钮、关窗口。"""
        dlg = TaskProgressDialog()
        dlg.start('质量评估')
        dlg.report_progress(63, '评估中')
        dlg.show()

        dlg.finish(success=success, message='评估完成' if success else '评估失败')
        assert dlg.is_finished is True
        assert dlg.isVisible() is False                 # 窗口不残留
        assert dlg._btn_cancel.isEnabled() is False
        if success:
            assert dlg.progress_percent == 100
        else:
            assert '评估失败' in dlg.message

    def test_finish_is_idempotent(self, qapp):
        dlg = TaskProgressDialog()
        dlg.finish(success=True, message='第一次')
        dlg.finish(success=True, message='第二次')      # 不应覆盖或异常
        assert '第一次' in dlg.message

    def test_reuse_across_consecutive_tasks(self, qapp):
        """同一实例承载连续任务时状态必须干净，避免上一步的 100% 残留。"""
        dlg = TaskProgressDialog()
        dlg.start('去噪')
        dlg.report_progress(100, '去噪完成')
        dlg.flush_progress()
        dlg.finish(success=True)

        dlg.start('立面提取')
        assert dlg.is_finished is False
        assert dlg._reporter.snapshot == (0, '')
        dlg.update_progress(0)          # 归零显式可设
        assert dlg.progress_percent == 0
