from __future__ import annotations

from PySide6.QtCore import QPoint, Qt, QEvent, QObject, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget


class _ResizeWatcher(QObject):
    """监听容器尺寸变化，同步覆盖层几何。"""
    def __init__(self, overlay, parent=None):
        super().__init__(parent)
        self.overlay = overlay

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Resize, QEvent.Move, QEvent.Show):
            self.overlay.sync_geometry()
        return False


class PointPickOverlay(QWidget):
    """
    顶层点选输入层，用于捕获鼠标事件、渲染已选点标记、将点击坐标转发给视口回调。
    """

    def __init__(self, viewport, container, parent=None):
        # 不要使用 WindowStaysOnTopHint：独立置顶窗会盖住三维视口，
        # 或在切换其他软件时继续浮在桌面上。
        super().__init__(parent, Qt.FramelessWindowHint | Qt.WindowDoesNotAcceptFocus)
        self.viewport = viewport
        self.container = container
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setFocusPolicy(Qt.NoFocus)
        self._watcher = _ResizeWatcher(self, container)
        container.installEventFilter(self._watcher)
        self._press_pos = None
        self._active = False

    def sync_geometry(self):
        """将覆盖层对齐到容器全局几何。"""
        if self.container is None or not self._active:
            return
        try:
            # 使用 container 的顶级窗口作为父窗口参考，确保坐标正确
            top_level = self.container.window()
            if top_level is not None:
                # 获取 container 在屏幕上的位置
                container_global = self.container.mapToGlobal(QPoint(0, 0))
                self.setGeometry(
                    container_global.x(),
                    container_global.y(),
                    self.container.width(),
                    self.container.height()
                )
            self.raise_()
        except RuntimeError:
            pass

    def activate(self):
        """激活点选模式：显示覆盖层、捕获输入。"""
        self.sync_geometry()
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.show()
        self.raise_()
        self._active = True

    def deactivate(self):
        """退出点选模式：释放输入、隐藏覆盖层。"""
        self._active = False
        try:
            # No grab was taken: the embedded Open3D QWindow remains the
            # owner of navigation and receives events immediately on exit.
            self.clearFocus()
        except RuntimeError:
            pass
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.hide()
        self._press_pos = None

    def paintEvent(self, event):
        """绘制已选点标记和连线 —— 纯视觉反馈，无算法逻辑。"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 获取当前选点状态（由视口维护）
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
                # 绘制圆形标记
                painter.setPen(QPen(color, 2))
                painter.setBrush(color)
                painter.drawEllipse(x - 6, y - 6, 12, 12)
                # 绘制序号标签
                painter.setPen(Qt.white)
                painter.drawText(x + 8, y - 8, f'{prefix}{i}')

        # 绘制源点和目标点之间的连线
        if len(src) > 0 and len(tgt) > 0:
            src_proj = self.viewport.project_points(src)
            tgt_proj = self.viewport.project_points(tgt)
            if src_proj is not None and tgt_proj is not None:
                src_screen, src_valid = src_proj
                tgt_screen, tgt_valid = tgt_proj
                pair_count = min(len(src), len(tgt))
                painter.setPen(QPen(QColor(255, 255, 255, 180), 1, Qt.DashLine))
                for i in range(pair_count):
                    if src_valid[i] and tgt_valid[i]:
                        painter.drawLine(
                            int(round(src_screen[i][0])), int(round(src_screen[i][1])),
                            int(round(tgt_screen[i][0])), int(round(tgt_screen[i][1]))
                        )

        painter.end()

    # ---------- 事件处理：只负责捕获和转发，不做算法决策 ----------

    def mousePressEvent(self, event):
        """记录按下位置，用于区分点击和拖拽。"""
        if event.button() == Qt.RightButton:
            self.viewport.exit_pick_mode()
        elif event.button() == Qt.LeftButton:
            self._press_pos = event.pos()
        event.accept()

    def mouseReleaseEvent(self, event):
        """左键释放时，若位移小则触发选点。"""
        if event.button() == Qt.LeftButton:
            start = self._press_pos
            self._press_pos = None
            if (start is not None and
                    (event.pos() - start).manhattanLength() <= 6):
                # 转发给视口处理选点逻辑
                self.viewport.handle_pick_screen(event.pos())
                # 选点后重绘以更新标记
                self.update()
        event.accept()

    def mouseMoveEvent(self, event):
        """鼠标移动时更新光标样式。"""
        self.setCursor(Qt.CrossCursor)
        event.accept()

    def wheelEvent(self, event):
        """吞掉滚轮事件，防止缩放干扰选点。"""
        event.accept()

    def keyPressEvent(self, event):
        """ESC 退出选点模式。"""
        if event.key() == Qt.Key_Escape:
            self.viewport.exit_pick_mode()
        event.accept()

    def enterEvent(self, event):
        """鼠标进入时设置十字光标。"""
        self.setCursor(Qt.CrossCursor)
        super().enterEvent(event)