from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QLabel, QProgressBar, QVBoxLayout


class LoadingDialog(QDialog):
    """点云加载中的模态等待窗。

    后台解析期间以应用级模态阻挡用户操作，但事件循环保持运行
    （worker 进度/完成信号照常投递）。无关闭按钮：加载只能由完成或
    失败回调结束，避免用户误以为点 X 可以取消。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('点云加载')
        self.setWindowFlags(Qt.WindowType.Dialog
                            | Qt.WindowType.CustomizeWindowHint
                            | Qt.WindowType.WindowTitleHint)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setMinimumWidth(380)

        self._label = QLabel('正在加载点云，请稍候...')
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setWordWrap(True)
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        layout.addWidget(self._label)
        layout.addWidget(self._bar)

    def update_progress(self, percent, text=''):
        if text:
            self._label.setText(text)
        if percent is not None and int(percent) >= 0:
            self._bar.setValue(min(100, int(percent)))

    def reject(self):
        # 不允许用户关闭加载窗口；只能由 load_finished 信号结束。
        pass
