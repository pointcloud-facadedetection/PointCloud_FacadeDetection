"""统一耗时任务进度窗。"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from utils.throttled_reporter import DEFAULT_INTERVAL_MS, ThrottledReporter


class TaskProgressDialog(QDialog):
    """通用进度窗。子类只需覆写默认标题与文案。"""

    #: 用户请求中止任务。右上角关闭按钮、取消按钮、Esc、Alt+F4
    #: 全部汇聚到这里，调用方只需接一个信号即可完成"关闭即中止"。
    cancel_requested = Signal()

    DEFAULT_TITLE = '任务进行中'
    DEFAULT_MESSAGE = '任务处理中'
    FINISH_MESSAGE = '处理完成'

    #: 心跳间隔：只刷新"已用时间"文案，不动进度条数值。
    HEARTBEAT_INTERVAL_MS = 1000

    def __init__(self, parent=None, *, cancellable: bool = False,
                 auto_close: bool = True,
                 interval_ms: int = DEFAULT_INTERVAL_MS,
                 title: str | None = None):
        super().__init__(parent)
        self._auto_close = bool(auto_close)
        self._cancellable = bool(cancellable)
        self._finished = False
        #: 由控制器驱动收尾（finish/accept）时置位，避免把"自动关闭"
        #: 误判成"用户点 X 取消"，从而误触发一次取消回调。
        self._closing_by_controller = False
        self._elapsed_seconds = 0

        self.setWindowTitle(title or self.DEFAULT_TITLE)
        self.setMinimumWidth(360)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setModal(True)
        self._apply_window_flags()

        # 标题行：左侧任务名，右侧状态位（百分比 / 处理中 / 已用时间）。
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

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        # 百分比统一由标题右侧状态位展示，进度条内部不再重复绘制。
        self._bar.setTextVisible(False)

        self._btn_cancel = QPushButton('取消')
        self._btn_cancel.setObjectName('taskProgressCancel')
        self._btn_cancel.setProperty('buttonRole', 'secondary')
        self._btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_cancel.clicked.connect(self._on_cancel_clicked)
        self._btn_cancel.setVisible(self._cancellable)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        header.addWidget(self._title_label, 1)
        header.addWidget(self._status_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 16)
        layout.setSpacing(10)
        layout.addLayout(header)
        layout.addWidget(self._label)
        layout.addWidget(self._bar)
        if self._cancellable:
            footer = QHBoxLayout()
            footer.setContentsMargins(0, 0, 0, 0)
            footer.addStretch(1)
            footer.addWidget(self._btn_cancel)
            layout.addLayout(footer)

        # 进度节流器：report_progress() 写入缓存，flushed 才真正刷 UI
        self._reporter = ThrottledReporter(interval_ms=interval_ms, parent=self)
        self._reporter.flushed.connect(self.update_progress)

        # 心跳只更新"已用时间"，保证界面持续活动而不干扰真实进度节拍。
        self._heartbeat = QTimer(self)
        self._heartbeat.setInterval(self.HEARTBEAT_INTERVAL_MS)
        self._heartbeat.timeout.connect(self._on_heartbeat)

    # ------------------------------------------------------------------
    # 只读属性
    # ------------------------------------------------------------------
    @property
    def progress_percent(self) -> int:
        return int(self._bar.value())

    @property
    def message(self) -> str:
        return self._label.text()

    @property
    def is_finished(self) -> bool:
        return self._finished

    @property
    def cancellable(self) -> bool:
        return self._cancellable

    # ------------------------------------------------------------------
    # 窗口标志 / 关闭策略
    # ------------------------------------------------------------------
    def _apply_window_flags(self) -> None:
        """保留系统标题栏；仅可取消任务才装配右上角关闭按钮。"""
        flags = (Qt.WindowType.Dialog
                 | Qt.WindowType.WindowTitleHint
                 | Qt.WindowType.WindowSystemMenuHint)
        if self._cancellable:
            flags |= Qt.WindowType.WindowCloseButtonHint
        self.setWindowFlags(flags)
        # setWindowFlags 会重置模态相关状态，这里显式重新断言，
        # 保证运行期切换可取消性后弹窗仍是应用级模态。
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setModal(True)

    def set_cancellable(self, cancellable: bool) -> None:
        """运行期切换取消能力（批量任务出队结束后应置 False）。"""
        self._cancellable = bool(cancellable)
        self._apply_window_flags()
        self._btn_cancel.setVisible(self._cancellable)
        self._btn_cancel.setEnabled(self._cancellable)
        if self.isVisible():
            # setWindowFlags 会隐藏窗口，必须重新显示以保持模态。
            self.show()
            self.raise_()

    def closeEvent(self, event) -> None:
        """右上角关闭 / Alt+F4 的统一落点。

        * 终态或控制器收尾中 → 正常关闭，绝不触发取消；
        * 用户主动关闭且任务可取消 → 关闭窗口并请求中止全部后台进程；
        * 不可取消 → 忽略关闭请求，窗口只能由 finish() 收尾。
        """
        if self._finished or self._closing_by_controller:
            self._closing_by_controller = False
            event.accept()
            return
        if self._cancellable:
            event.accept()
            self._heartbeat.stop()
            self._on_cancel_clicked()
            return
        event.ignore()

    def reject(self) -> None:
        """Esc 走这里：语义与右上角关闭完全一致。"""
        if self._finished or self._closing_by_controller:
            self._closing_by_controller = False
            super().reject()
            return
        if self._cancellable:
            self._heartbeat.stop()
            self._on_cancel_clicked()
            return
        # 不可取消：不允许用 Esc 结束任务窗，只能由 finish() 驱动收尾。

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self, title: str | None = None, message: str | None = None,
              *, determinate: bool = True) -> None:
        """开始一个新任务；同一实例可复用承载连续任务。"""
        self._finished = False
        self._closing_by_controller = False
        self._elapsed_seconds = 0
        self._reporter.reset()
        if title:
            self.setWindowTitle(title)
            self._title_label.setText(title)
        if message is not None:
            self._label.setText(message)
        # 复用同一实例承载下一次任务时进度条必须归零：finish() 会把成功任务
        # 刷到 100% 且不重置数值，若不归零，下一次任务刚弹出就显示 100%。
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self.set_determinate(determinate)
        self._status_label.setText(self._status_text())
        self._btn_cancel.setEnabled(True)
        self._btn_cancel.setVisible(self._cancellable)
        self._heartbeat.start()

    def set_determinate(self, determinate: bool) -> None:
        """不定量模式（``setRange(0, 0)``）用于算法暂时无法回报百分比的阶段。"""
        if determinate:
            self._bar.setRange(0, 100)
        else:
            self._bar.setRange(0, 0)
        self._status_label.setText(self._status_text())

    # ------------------------------------------------------------------
    # 心跳 / 文案
    # ------------------------------------------------------------------
    def _status_text(self) -> str:
        """状态位文案：定量显示百分比，另附已用时间；不定量显示"处理中"。"""
        if self._bar.maximum() == 0:
            base = '处理中'
        else:
            base = f'{self._bar.value()}%'
        if self._elapsed_seconds <= 0:
            return base
        minutes, seconds = divmod(int(self._elapsed_seconds), 60)
        return f'{base} · {minutes:02d}:{seconds:02d}'

    def _on_heartbeat(self) -> None:
        """每秒只刷新已用时间，进度条数值仍由后台真实回报驱动。"""
        if self._finished:
            return
        self._elapsed_seconds += 1
        self._status_label.setText(self._status_text())

    # ------------------------------------------------------------------
    # 进度更新
    # ------------------------------------------------------------------
    def update_progress(self, percent, text: str = '') -> None:
        """直写进度条与文案（立即生效，不做节流）。

        保持与历史 ``LoadingDialog.update_progress`` 完全一致的行为，
        包括超过 100 截断、``percent=None`` 时只更新文案。
        """
        if text:
            self._label.setText(str(text))
        if percent is not None:
            try:
                value = int(percent)
            except (TypeError, ValueError):
                value = -1
            if value >= 0:
                if self._bar.maximum() == 0:  # 不定量模式：先恢复定量再设值
                    self._bar.setRange(0, 100)
                self._bar.setValue(min(100, value))
                self._status_label.setText(self._status_text())

    def report_progress(self, percent, text: str = '') -> None:
        """高频进度入口：写入缓存，最多每 ``interval_ms`` 刷新一次 UI。"""
        self._reporter.report(percent, text)

    def flush_progress(self) -> None:
        """强制把缓存的进度刷到 UI（阶段性里程碑可调用）。"""
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
        """终态收尾：必刷进度、停心跳、按需自动关闭。

        完成 / 失败 / 取消三条路径都必须经过这里，避免窗口残留。
        终态关闭不会触发 :attr:`cancel_requested`。
        """
        if self._finished:
            return
        self._finished = True
        self._heartbeat.stop()

        if success:
            self._reporter.flush_now(100, message or self.FINISH_MESSAGE)
            self._bar.setRange(0, 100)
            self._bar.setValue(100)
        else:
            # 失败时保留当前进度值，仅更新文案，便于用户判断停在哪一步
            self._reporter.flush_now(-1, message or '处理失败')
        self._status_label.setText(self._status_text())

        self._btn_cancel.setEnabled(False)
        self._btn_cancel.setVisible(False)

        if self._auto_close:
            self._closing_by_controller = True
            self.accept()

    def _on_cancel_clicked(self) -> None:
        self._btn_cancel.setEnabled(False)
        self._label.setText('正在取消...')
        self.cancel_requested.emit()
