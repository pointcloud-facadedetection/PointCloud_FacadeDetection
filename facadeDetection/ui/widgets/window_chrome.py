from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget

class ElidedLabel(QLabel):
    """在空间不足时省略中间文本，同时保留完整内容供 Tooltip 查看。"""

    def __init__(self, text='', parent=None, maximum_hint_width=480):
        super().__init__('', parent)
        self._full_text = str(text)
        self._maximum_hint_width = maximum_hint_width
        self.setMinimumWidth(0)
        self.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self._update_elided_text()

    def setText(self, text):
        self._full_text = str(text)
        self.updateGeometry()
        self._update_elided_text()

    def fullText(self):
        return self._full_text

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_elided_text()

    def sizeHint(self):
        hint = super().sizeHint()
        # 使用完整文本计算理想宽度；布局空间不足时再由 resizeEvent 省略。
        full_text_width = self.fontMetrics().horizontalAdvance(self._full_text) + 32
        if self._maximum_hint_width is not None:
            full_text_width = min(full_text_width, self._maximum_hint_width)
        hint.setWidth(max(hint.width(), full_text_width))
        return hint

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        hint.setWidth(0)
        return hint

    def _update_elided_text(self):
        available_width = max(0, self.contentsRect().width())
        visible_text = self.fontMetrics().elidedText(
            self._full_text,
            Qt.TextElideMode.ElideMiddle,
            available_width,
        )
        QLabel.setText(self, visible_text)

class ApplicationTitleBar(QWidget):
    """无边框窗口的可拖动标题栏，保留系统标题栏的常用交互。"""

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            window_handle = self.window().windowHandle()
            if window_handle is not None and window_handle.startSystemMove():
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            toggle_window_state = getattr(
                self.window(),
                '_toggle_maximize_restore',
                None,
            )
            if toggle_window_state is not None:
                toggle_window_state()
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

class WindowResizeHandle(QWidget):
    """透明窗口缩放热区，替代 Windows 会露出白边的原生粗边框。"""

    def __init__(self, edges, cursor, parent=None):
        super().__init__(parent)
        self._edges = edges
        self._last_start_result = None
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setCursor(cursor)

    def mousePressEvent(self, event):
        if (
            event.button() == Qt.MouseButton.LeftButton
            and not self.window().isMaximized()
        ):
            window_handle = self.window().windowHandle()
            if window_handle is not None:
                # 交给操作系统执行缩放，拖动过程比手算 geometry 更顺滑。
                self._last_start_result = window_handle.startSystemResize(
                    self._edges
                )
                if self._last_start_result:
                    event.accept()
                    return
        super().mousePressEvent(event)
