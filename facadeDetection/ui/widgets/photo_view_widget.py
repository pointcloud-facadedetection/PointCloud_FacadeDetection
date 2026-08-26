"""点云上方的 2D 照片视图，支持点击标注、滚轮缩放与拖动平移。"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy


class PhotoViewWidget(QLabel):
    point_clicked = Signal(float, float)

    _MIN_ZOOM = 1.0
    _MAX_ZOOM = 8.0

    def __init__(self, parent=None, placeholder='尚未上传 2D 照片'):
        super().__init__(parent)
        self.setObjectName('photoViewWidget')
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(120, 60)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)
        self.setToolTip('滚轮缩放，右键拖动平移')
        self._placeholder = placeholder
        self._source = QImage()
        self._scaled = QPixmap()
        self._draw_rect = QRectF()
        self._markers: list[tuple[float, float]] = []
        self._interactive = False
        self._zoom = 1.0
        self._offset = QPointF(0, 0)
        self._panning = False
        self._pan_last = QPointF()
        self.setText(self._placeholder)
        self.setStyleSheet(
            'QLabel#photoViewWidget {'
            ' background-color: #1f2430; color: #c5cad3;'
            ' border: 1px solid #d5d9df; border-radius: 4px;'
            '}'
        )

    def set_interactive(self, enabled: bool):
        self._interactive = enabled
        self.setCursor(
            Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor
        )

    def set_image(self, image: QImage):
        self._source = QImage(image)
        self._zoom = 1.0
        self._offset = QPointF(0, 0)
        self.setText('')
        self._rebuild_scaled()
        self.update()

    def clear_image(self):
        self._source = QImage()
        self._scaled = QPixmap()
        self._draw_rect = QRectF()
        self._markers = []
        self._zoom = 1.0
        self._offset = QPointF(0, 0)
        self.setText(self._placeholder)
        self.update()

    def set_markers(self, points):
        self._markers = [(float(x), float(y)) for x, y in points]
        self.update()

    def has_image(self) -> bool:
        return not self._source.isNull()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._source.isNull():
            self._rebuild_scaled()

    def wheelEvent(self, event):
        if self._source.isNull() or self._draw_rect.width() <= 0:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta == 0:
            event.accept()
            return
        factor = 1.12 if delta > 0 else 1.0 / 1.12
        new_zoom = max(self._MIN_ZOOM, min(self._MAX_ZOOM, self._zoom * factor))
        if abs(new_zoom - self._zoom) < 1e-6:
            event.accept()
            return
        cursor = event.position()
        pixel = self._display_to_pixel(cursor)
        self._zoom = new_zoom
        self._rebuild_scaled()
        if pixel is not None:
            mapped = self._pixel_to_display(pixel[0], pixel[1])
            if mapped is not None:
                self._offset += cursor - mapped
                self._rebuild_scaled()
        event.accept()
        self.update()

    def mousePressEvent(self, event):
        if self._source.isNull():
            super().mousePressEvent(event)
            return
        if event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._panning = True
            self._pan_last = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._interactive
            and self._draw_rect.contains(event.position())
        ):
            pixel = self._display_to_pixel(event.position())
            if pixel is not None:
                self.point_clicked.emit(pixel[0], pixel[1])
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.position() - self._pan_last
            self._pan_last = event.position()
            self._offset += delta
            self._rebuild_scaled()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._panning and event.button() in (
            Qt.MouseButton.RightButton,
            Qt.MouseButton.MiddleButton,
        ):
            self._panning = False
            self.set_interactive(self._interactive)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._source.isNull() or self._scaled.isNull():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setClipRect(self.rect())
        painter.drawPixmap(self._draw_rect.toRect(), self._scaled)
        font = QFont()
        font.setPixelSize(11)
        font.setBold(True)
        painter.setFont(font)
        for index, (px, py) in enumerate(self._markers, start=1):
            display = self._pixel_to_display(px, py)
            if display is None:
                continue
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.setBrush(QColor(220, 40, 40))
            painter.drawEllipse(display, 8, 8)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(
                QRectF(display.x() - 8, display.y() - 8, 16, 16),
                Qt.AlignmentFlag.AlignCenter,
                str(index),
            )

    def _rebuild_scaled(self):
        if self._source.isNull() or self.width() <= 2 or self.height() <= 2:
            return
        pixmap = QPixmap.fromImage(self._source)
        fitted = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        width = max(1, int(round(fitted.width() * self._zoom)))
        height = max(1, int(round(fitted.height() * self._zoom)))
        self._scaled = pixmap.scaled(
            width,
            height,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = (self.width() - self._scaled.width()) / 2 + self._offset.x()
        y = (self.height() - self._scaled.height()) / 2 + self._offset.y()
        self._draw_rect = QRectF(x, y, self._scaled.width(), self._scaled.height())

    def _display_to_pixel(self, pos: QPointF):
        if self._draw_rect.width() <= 0 or self._draw_rect.height() <= 0:
            return None
        nx = (pos.x() - self._draw_rect.x()) / self._draw_rect.width()
        ny = (pos.y() - self._draw_rect.y()) / self._draw_rect.height()
        if not (0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0):
            return None
        return nx * self._source.width(), ny * self._source.height()

    def _pixel_to_display(self, px: float, py: float) -> QPointF | None:
        if self._source.width() <= 0 or self._source.height() <= 0:
            return None
        x = self._draw_rect.x() + px / self._source.width() * self._draw_rect.width()
        y = self._draw_rect.y() + py / self._source.height() * self._draw_rect.height()
        return QPointF(x, y)
