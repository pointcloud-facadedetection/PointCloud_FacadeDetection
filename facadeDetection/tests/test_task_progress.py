"""进度反馈基础件的实效测试。

分两层断言：

1. 机制：ThrottledReporter 的"高频写入 → 低频刷新"确实成立，
   且终态必刷、无新数据时不空刷。
2. 忙碌条模式：进度条恒为不定量（无百分比、无取消按钮），
   finish() 成功 / 失败两条收尾路径都真实生效。
"""
import pytest
from PySide6.QtCore import Qt

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
        reporter.flushed.connect(lambda t: emitted.append(t))

        for i in range(1000):
            reporter.report(f'步骤 {i}')
        assert emitted == []          # 节流期内一次都没刷
        assert reporter.pending is True

        assert reporter.flush() is True
        assert len(emitted) == 1
        # 刷的是最后一次的值，中间态被合并丢弃
        assert emitted[0] == '步骤 999'
        assert reporter.pending is False

    def test_no_new_data_does_not_refresh(self, qapp):
        """无新写入时定时器不应重复刷新（避免无意义重绘）。"""
        reporter = ThrottledReporter(interval_ms=5000)
        reporter.set_auto_start(False)
        emitted = []
        reporter.flushed.connect(lambda t: emitted.append(t))

        assert reporter.flush() is False
        reporter.report('a')
        assert reporter.flush() is True
        assert reporter.flush() is False
        assert len(emitted) == 1

    def test_flush_now_bypasses_throttle_and_stops(self, qapp):
        """终态刷新必须立即生效并停表。"""
        reporter = ThrottledReporter(interval_ms=5000)
        reporter.set_auto_start(False)
        emitted = []
        reporter.flushed.connect(lambda t: emitted.append(t))

        reporter.flush_now('完成')
        assert emitted == ['完成']
        assert reporter.is_active() is False

    def test_empty_text_keeps_previous_copy(self, qapp):
        reporter = ThrottledReporter(interval_ms=5000)
        reporter.set_auto_start(False)
        reporter.report('超上限')
        reporter.report('')  # 空文本不清空上一次文案
        assert reporter.snapshot == '超上限'
        reporter.report('仅换文案')
        assert reporter.snapshot == '仅换文案'

    def test_reset_clears_buffered_state(self, qapp):
        reporter = ThrottledReporter(interval_ms=5000)
        reporter.report('半途')
        reporter.reset()
        assert reporter.pending is False
        assert reporter.snapshot == ''
        assert reporter.is_active() is False


# ---------------------------------------------------------------------------
# TaskProgressDialog 忙碌条模式
# ---------------------------------------------------------------------------
class TestTaskProgressDialog:
    def test_bar_is_always_indeterminate(self, qapp):
        """没有百分比模式：进度条恒为 setRange(0, 0) 的忙碌条。"""
        dlg = TaskProgressDialog()
        dlg.start('点云去噪')
        assert dlg._bar.maximum() == 0
        assert '处理中' in dlg._status_label.text()
        dlg.update_progress('去噪中')
        assert dlg._bar.maximum() == 0
        assert '去噪中' in dlg._label.text()

    def test_report_progress_is_throttled_until_flush(self, qapp):
        dlg = TaskProgressDialog(interval_ms=5000)
        dlg.report_progress(80, '立面提取中')   # 百分比被忽略
        assert '立面提取中' not in dlg.message   # 仍在缓存里，未上屏
        dlg.flush_progress()
        assert '立面提取中' in dlg.message

    def test_not_user_closable(self, qapp):
        """无取消按钮、无关闭按钮，Esc/关闭都被忽略。"""
        dlg = TaskProgressDialog()
        assert not dlg.windowFlags() & Qt.WindowType.WindowCloseButtonHint
        dlg.show()
        dlg.reject()                    # 模拟 Esc
        qapp.processEvents()
        assert dlg.isVisible()
        dlg.hide()

    @pytest.mark.parametrize('success', [True, False])
    def test_finish_always_closes(self, qapp, success):
        """成功与失败都必须走完收尾：停心跳、关窗口。"""
        dlg = TaskProgressDialog()
        dlg.start('质量评估')
        dlg.report_progress(text='评估中')
        dlg.show()

        dlg.finish(success=success, message='评估完成' if success else '评估失败')
        assert dlg.is_finished is True
        assert dlg.isVisible() is False                 # 窗口不残留
        if success:
            assert '评估完成' in dlg.message
        else:
            assert '评估失败' in dlg.message

    def test_finish_is_idempotent(self, qapp):
        dlg = TaskProgressDialog()
        dlg.finish(success=True, message='第一次')
        dlg.finish(success=True, message='第二次')      # 不应覆盖或异常
        assert '第一次' in dlg.message

    def test_reuse_across_consecutive_tasks(self, qapp):
        """同一实例承载连续任务时状态必须干净。"""
        dlg = TaskProgressDialog()
        dlg.start('去噪')
        dlg.report_progress(text='去噪完成')
        dlg.flush_progress()
        dlg.finish(success=True)

        dlg.start('立面提取')
        assert dlg.is_finished is False
        assert dlg._reporter.snapshot == ''
        assert '处理中' in dlg._status_label.text()
