"""点云上方的 2D 照片视图，支持点击标注像素点。"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy


class PhotoViewWidget(QLabel):
    point_clicked = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('photoViewWidget')
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(140)
        self.setMaximumHeight(260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._source = QImage()
        self._scaled = QPixmap()
        self._draw_rect = QRectF()
        self._markers: list[tuple[float, float]] = []
        self._interactive = False
        self.setText('尚未上传 2D 照片')
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
        self.setText('')
        self._rebuild_scaled()
        self.update()

    def clear_image(self):
        self._source = QImage()
        self._scaled = QPixmap()
        self._draw_rect = QRectF()
        self._markers = []
        self.setText('尚未上传 2D 照片')
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

    def mousePressEvent(self, event):
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._interactive
            and not self._source.isNull()
            and self._draw_rect.contains(event.position())
        ):
            pixel = self._display_to_pixel(event.position())
            if pixel is not None:
                self.point_clicked.emit(pixel[0], pixel[1])
                return
        super().mousePressEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._source.isNull() or self._scaled.isNull():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
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
        self._scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = (self.width() - self._scaled.width()) / 2
        y = (self.height() - self._scaled.height()) / 2
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
