"""统一耗时任务进度窗：默认忙碌条，导入任务可显示真实百分比。"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
)

from utils.throttled_reporter import DEFAULT_INTERVAL_MS, ThrottledReporter


class TaskProgressDialog(QDialog):
    """通用进度窗。子类只需覆写默认标题与文案。"""

    DEFAULT_TITLE = '任务进行中'
    DEFAULT_MESSAGE = '任务处理中'
    FINISH_MESSAGE = '处理完成'

    #: 心跳间隔：只刷新"已用时间"文案。
    HEARTBEAT_INTERVAL_MS = 1000

    def __init__(self, parent=None, *,
                 auto_close: bool = True,
                 interval_ms: int = DEFAULT_INTERVAL_MS,
                 title: str | None = None):
        super().__init__(parent)
        self._auto_close = bool(auto_close)
        self._finished = False
        self._elapsed_seconds = 0
        self._allow_percentage = False

        self.setWindowTitle(title or self.DEFAULT_TITLE)
        self.setMinimumWidth(360)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setModal(True)
        self.setWindowFlags(Qt.WindowType.Dialog
                            | Qt.WindowType.WindowTitleHint
                            | Qt.WindowType.CustomizeWindowHint)

        # 标题行：左侧任务名，右侧状态位（处理中 · 已用时间）。
        self._title_label = QLabel(self.DEFAULT_TITLE)
        self._title_label.setObjectName('taskProgressTitle')
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignLeft
                                       | Qt.AlignmentFlag.AlignVCenter)

        self._status_label = QLabel('处理中')
        self._status_label.setObjectName('taskProgressStatus')
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignRight
                                        | Qt.AlignmentFlag.AlignVCenter)

        # 主文案：固定提示"任务处理中"；名称保留供既有调用方与测试探测。
        self._label = QLabel(self.DEFAULT_MESSAGE)
        self._label.setObjectName('taskProgressMessage')
        self._label.setAlignment(Qt.AlignmentFlag.AlignLeft
                                 | Qt.AlignmentFlag.AlignVCenter)
        self._label.setWordWrap(True)

        # 默认忙碌条；只有允许百分比的任务才切换到 0–100 区间。
        self._bar = QProgressBar()
        self._bar.setRange(0, 0)
        self._bar.setTextVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 16)
        layout.setSpacing(10)
        layout.addLayout(self._build_header())
        layout.addWidget(self._label)
        layout.addWidget(self._bar)

        # 文案节流器：report_progress() 写入缓存，flushed 才真正刷 UI
        self._reporter = ThrottledReporter(interval_ms=interval_ms, parent=self)
        self._reporter.flushed.connect(self.update_progress)

        # 心跳只更新"已用时间"，保证界面持续活动而不干扰文案节拍。
        self._heartbeat = QTimer(self)
        self._heartbeat.setInterval(self.HEARTBEAT_INTERVAL_MS)
        self._heartbeat.timeout.connect(self._on_heartbeat)

    def _build_header(self) -> QHBoxLayout:
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        header.addWidget(self._title_label, 1)
        header.addWidget(self._status_label)
        return header

    # ------------------------------------------------------------------
    # 只读属性
    # ------------------------------------------------------------------
    @property
    def message(self) -> str:
        return self._label.text()

    @property
    def is_finished(self) -> bool:
        return self._finished

    # ------------------------------------------------------------------
    # 关闭策略：只能由 finish() 收尾，用户无法关闭/中止
    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:
        if self._finished:
            event.accept()
            return
        event.ignore()

    def reject(self) -> None:
        # Esc 不允许结束任务窗，只能由 finish() 驱动收尾。
        if self._finished:
            super().reject()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self, title: str | None = None, message: str | None = None,
              determinate: bool = False) -> None:
        """开始任务；只有允许百分比的任务才会显示真实进度数值。"""
        self._finished = False
        self._elapsed_seconds = 0
        self._allow_percentage = bool(determinate)
        self._set_progress_mode(self._allow_percentage)
        self._reporter.reset()
        if title:
            self.setWindowTitle(title)
            self._title_label.setText(title)
        if message is not None:
            self._label.setText(message)
        self._status_label.setText('处理中')
        self._heartbeat.start()

    def _set_progress_mode(self, determinate):
        self._bar.setRange(0, 100 if determinate else 0)
        self._bar.setTextVisible(bool(determinate))
        if determinate:
            self._bar.setValue(0)

    # ------------------------------------------------------------------
    # 心跳 / 文案
    # ------------------------------------------------------------------
    def _on_heartbeat(self) -> None:
        """每秒只刷新已用时间。"""
        if self._finished:
            return
        self._elapsed_seconds += 1
        minutes, seconds = divmod(int(self._elapsed_seconds), 60)
        self._status_label.setText(f'处理中 · {minutes:02d}:{seconds:02d}')

    # ------------------------------------------------------------------
    # 进度更新：只更新阶段文案，忙碌条无需数值
    # ------------------------------------------------------------------
    def update_progress(self, text: str = '') -> None:
        """直写阶段文案（立即生效，不做节流）。"""
        if text:
            self._label.setText(str(text))

    def report_progress(self, percent=None, text: str = '') -> None:
        """百分比来自任务的真实进度；未知阶段保留忙碌条。"""
        if self._allow_percentage:
            if percent is None or int(percent) < 0:
                self._set_progress_mode(False)
            else:
                if self._bar.maximum() != 100:
                    self._set_progress_mode(True)
                self._bar.setValue(min(100, int(percent)))
        self._reporter.report(text)

    def flush_progress(self) -> None:
        """强制把缓存的文案刷到 UI（阶段性里程碑可调用）。"""
        self._reporter.flush()

    def set_interval_ms(self, interval_ms: int) -> None:
        """调整合并间隔；主要供测试缩短等待时间。"""
        self._reporter.stop()
        self._reporter.deleteLater()
        self._reporter = ThrottledReporter(interval_ms=interval_ms, parent=self)
        self._reporter.flushed.connect(self.update_progress)

    # ------------------------------------------------------------------
    # 结束
    # ------------------------------------------------------------------
    def finish(self, success: bool = True, message: str = '') -> None:
        """终态收尾：停心跳并按需自动关闭（幂等）。"""
        if self._finished:
            return
        self._finished = True
        self._heartbeat.stop()
        self._reporter.stop()
        if message:
            self._label.setText(message)
        self._status_label.setText(
            self.FINISH_MESSAGE if success else '处理失败')

        if self._auto_close:
            self.accept()
