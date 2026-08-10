from PySide6.QtCore import Qt, QPoint, QObject, QEvent, QRect
from PySide6.QtGui import QPainter, QPen, QColor
from PySide6.QtWidgets import QWidget


class _ContainerResizeWatcher(QObject):
    def __init__(self, overlay, parent=None):
        super().__init__(parent)
        self._overlay = overlay

    def eventFilter(self, watched, event):
        et = event.type()
        if et in (QEvent.Resize, QEvent.Show):
            try:
                if self._overlay is not None and self._overlay.parent() is not None:
                    p = self._overlay.parent()
                    self._overlay.setGeometry(0, 0, p.width(), p.height())
                    self._overlay.raise_()
            except Exception:
                pass
        return False


class ROISelectionOverlay(QWidget):
    """Top-level overlay that captures all input during ROI selection and draws a 2D rectangle."""
    def __init__(self, controller, parent, flags: int | None = None):
        # 顶层工具窗，确保位于原生 Open3D 窗口之上
        if flags is None:
            flags = int(Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        super().__init__(None, Qt.WindowFlags(flags))
        self._controller = controller
        self._p1: QPoint | None = None
        self._p2: QPoint | None = None
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        # correct translucent attribute name
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._container = parent

    def activate(self):
        # become opaque-to-input to capture everything
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.grabMouse()
        self.grabKeyboard()
        self.setFocus()
        try:
            from PySide6.QtGui import QCursor
            self.setCursor(Qt.CrossCursor)
        except Exception:
            pass
        # 将自身对齐到 container 的全局几何
        try:
            if self._container is not None:
                top_left = self._container.mapToGlobal(QPoint(0, 0))
                self.setGeometry(top_left.x(), top_left.y(), self._container.width(), self._container.height())
        except Exception:
            pass
        self.show()
        self.raise_()

    def deactivate(self):
        try:
            self.releaseMouse()
        except Exception:
            pass
        try:
            self.releaseKeyboard()
        except Exception:
            pass
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        try:
            self.unsetCursor()
        except Exception:
            pass
        self._p1 = None
        self._p2 = None
        self.update()

    def paintEvent(self, event):
        if self._p1 is None or self._p2 is None:
            return
        try:
            from config.settings import Config
            x1, y1 = self._p1.x(), self._p1.y()
            x2, y2 = self._p2.x(), self._p2.y()
            x, y = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)

            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing, True)
            br, bg, bb, ba = getattr(Config, 'SELECT_BORDER_RGBA', (255, 0, 0, 240))
            bw = int(getattr(Config, 'SELECT_BORDER_WIDTH', 2))
            pen = QPen(QColor(int(br), int(bg), int(bb), int(ba)))
            pen.setWidth(max(1, bw))
            pen.setStyle(Qt.SolidLine)
            painter.setPen(pen)

            # ROI 模式只保留红色实线边框，避免辅助线/填充遮挡点云。
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(x, y, w, h)
            painter.end()
        except Exception:
            pass

    # ------------- Input capture -------------
    def mousePressEvent(self, event):
        try:
            if event.button() == Qt.LeftButton:
                self._p1 = event.pos()
                self._p2 = event.pos()
                self.update()
            elif event.button() == Qt.RightButton:
                self._controller.cancel()
        except Exception:
            pass
        event.accept()

    def mouseMoveEvent(self, event):
        try:
            if self._p1 is not None:
                old_p2 = self._p2
                self._p2 = event.pos()
                # 仅重绘变动区域（旧矩形与新矩形的并集）
                if old_p2 is not None:
                    x1, y1 = self._p1.x(), self._p1.y()
                    ox2, oy2 = old_p2.x(), old_p2.y()
                    nx2, ny2 = self._p2.x(), self._p2.y()
                    rx1, ry1 = min(x1, ox2), min(y1, oy2)
                    rw1, rh1 = abs(ox2 - x1), abs(oy2 - y1)
                    rx2, ry2 = min(x1, nx2), min(y1, ny2)
                    rw2, rh2 = abs(nx2 - x1), abs(ny2 - y1)
                    rect1 = QRect(rx1 - 24, ry1 - 24, rw1 + 48, rh1 + 48)
                    rect2 = QRect(rx2 - 24, ry2 - 24, rw2 + 48, rh2 + 48)
                    self.update(rect1.united(rect2))
                else:
                    self.update()
        except Exception:
            pass
        event.accept()

    def mouseReleaseEvent(self, event):
        try:
            if event.button() == Qt.LeftButton and self._p1 is not None:
                p1, p2 = self._p1, event.pos()
                self._p2 = p2
                self.update()
                self._controller.finish(p1, p2)
        except Exception:
            pass
        event.accept()

    def wheelEvent(self, event):
        # swallow zoom
        event.accept()

    def keyPressEvent(self, event):
        try:
            if event.key() == Qt.Key_Escape:
                self._controller.cancel()
        except Exception:
            pass
        event.accept()


class ROISelectionController:
    """Controller to run ROI selection on top of the Open3D viewport."""
    def __init__(self, viewport, container_widget=None):
        self._viewport = viewport
        self._container = container_widget
        self._overlay: ROISelectionOverlay | None = None
        self._on_complete = None
        self._cloud_name = None
        self._resize_filter = None

    def bind_container(self, container_widget):
        self._container = container_widget
        if self._overlay is not None:
            # 同步几何到新容器
            try:
                top_left = self._container.mapToGlobal(QPoint(0, 0))
                self._overlay.setGeometry(top_left.x(), top_left.y(), self._container.width(), self._container.height())
                self._overlay.raise_()
            except Exception:
                pass

    def start(self, cloud_name: str | None, on_complete):
        if self._container is None:
            raise RuntimeError('ROISelectionController: container not bound')
        self._on_complete = on_complete
        self._cloud_name = cloud_name
        if self._overlay is None:
            self._overlay = ROISelectionOverlay(self, self._container)
        # 先同步几何到 container 的全局位置
        try:
            top_left = self._container.mapToGlobal(QPoint(0, 0))
            self._overlay.setGeometry(top_left.x(), top_left.y(), self._container.width(), self._container.height())
        except Exception:
            pass
        self._overlay.show()
        self._overlay.activate()
        try:
            self._overlay.raise_()
        except Exception:
            pass
        # install resize watcher (keeps overlay synced with container size)
        if self._resize_filter is None:
            try:
                self._resize_filter = _ContainerResizeWatcher(self._overlay, parent=self._container)
                self._container.installEventFilter(self._resize_filter)
            except Exception:
                pass
        # 切换视口交互模式为 ROI（单一路径，无事件竞争）
        try:
            if hasattr(self._viewport, 'set_mode') and hasattr(self._viewport, 'InteractionMode'):
                self._viewport.set_mode(self._viewport.InteractionMode.ROI)
        except Exception:
            pass

    def cancel(self):
        # release and hide overlay
        if self._overlay is not None:
            self._overlay.deactivate()
            try:
                self._overlay.hide()
            except Exception:
                pass
        # 恢复视口交互为导航模式
        try:
            if hasattr(self._viewport, 'set_mode') and hasattr(self._viewport, 'InteractionMode'):
                self._viewport.set_mode(self._viewport.InteractionMode.NAVIGATE)
        except Exception:
            pass

    def finish(self, p1: QPoint, p2: QPoint):
        # compute indices then call on_complete
        idx = self._compute_indices(p1, p2, self._cloud_name)
        cb = self._on_complete
        # keep overlay visible until service clears? We hide to avoid confusion.
        self.cancel()
        if callable(cb):
            try:
                cb(None, None, idx)
            except Exception:
                pass

    # expose overlay for viewport window bridge
    def overlay(self):
        return self._overlay

    # ---------- core: compute indices in rect ----------
    def _compute_indices(self, p1: QPoint, p2: QPoint, cloud_name: str | None):
        # ROI 控制器只负责窗口生命周期与输入；投影和索引计算归口交互层。
        interactor = getattr(self._viewport, '_interactor', None)
        if interactor is None:
            return []
        return interactor.select_indices_in_rect(cloud_name, p1, p2)
