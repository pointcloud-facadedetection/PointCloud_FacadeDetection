from __future__ import annotations

from PySide6.QtCore import QPoint, Qt, QEvent, QObject
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget


class _ResizeWatcher(QObject):
    def __init__(self, overlay, parent=None):
        super().__init__(parent)
        self.overlay = overlay

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Resize, QEvent.Move, QEvent.Show):
            self.overlay.sync_geometry()
        return False


class PointPickOverlay(QWidget):
    """顶层点选输入层；不包含业务或配准算法。"""

    def __init__(self, viewport, container, parent=None):
        super().__init__(parent, Qt.Tool | Qt.FramelessWindowHint |
                         Qt.WindowStaysOnTopHint)
        self.viewport = viewport
        self.container = container
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setFocusPolicy(Qt.StrongFocus)
        self._watcher = _ResizeWatcher(self, container)
        container.installEventFilter(self._watcher)
        self._press_pos = None

    def sync_geometry(self):
        if self.container is None:
            return
        try:
            p = self.container.mapToGlobal(QPoint(0, 0))
            self.setGeometry(p.x(), p.y(), self.container.width(), self.container.height())
            self.raise_()
        except RuntimeError:
            pass

    def activate(self):
        self.sync_geometry()
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.show()
        self.raise_()
        self.grabMouse()
        self.grabKeyboard()
        self.setFocus(Qt.OtherFocusReason)

    def deactivate(self):
        try:
            self.releaseMouse()
            self.releaseKeyboard()
        except RuntimeError:
            pass
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.hide()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        src, tgt = self.viewport.registration_pick_points()
        for points, color, prefix in ((src, QColor(255, 75, 75), 'S'),
                                      (tgt, QColor(70, 230, 120), 'T')):
            projected = self.viewport.project_points(points)
            if projected is None:
                continue
            screen, valid = projected
            painter.setFont(QFont('Segoe UI', 10, QFont.Bold))
            for i, (xy, ok) in enumerate(zip(screen, valid), 1):
                if not ok:
                    continue
                x, y = int(round(xy[0])), int(round(xy[1]))
                painter.setPen(QPen(color, 2))
                painter.setBrush(color)
                painter.drawEllipse(x - 5, y - 5, 10, 10)
                painter.setPen(Qt.white)
                painter.drawText(x + 7, y - 7, f'{prefix}{i}')
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.viewport.exit_pick_mode()
        elif event.button() == Qt.LeftButton:
            self._press_pos = event.pos()
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            start = self._press_pos
            self._press_pos = None
            if (start is not None and
                    (event.pos() - start).manhattanLength() <= 4):
                self.viewport._handle_pick_screen(event.pos())
        event.accept()

    def wheelEvent(self, event):
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.viewport.exit_pick_mode()
        event.accept()