from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

class TechnicalCanvas(QWidget):
    """绘制与点云/工程场景相关的轻量空状态，不依赖额外图片资源。"""

    def __init__(self, variant='facade', message='', parent=None):
        super().__init__(parent)
        self.variant = variant
        self.message = message
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setMinimumHeight(220)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor('#FBFCFE'))

        canvas = self.rect().adjusted(24, 24, -24, -24)
        self._draw_grid(painter, canvas)
        if self.variant == 'document':
            self._draw_document(painter, canvas)
        else:
            self._draw_facade(painter, canvas)
        painter.end()

    def _draw_grid(self, painter, rect):
        painter.setPen(QPen(QColor('#EDF2F7'), 1))
        step = 32
        for x in range(rect.left(), rect.right() + 1, step):
            painter.drawLine(x, rect.top(), x, rect.bottom())
        for y in range(rect.top(), rect.bottom() + 1, step):
            painter.drawLine(rect.left(), y, rect.right(), y)

    def _draw_facade(self, painter, rect):
        width = min(440, max(220, int(rect.width() * 0.42)))
        height = min(248, max(150, int(rect.height() * 0.48)))
        center = rect.center()
        x = center.x() - width / 2
        y = center.y() - height / 2 - (18 if self.message else 0)
        facade = QRectF(x, y, width, height)

        painter.setPen(QPen(QColor('#B9C7DA'), 1.4))
        painter.drawRect(facade)
        depth = min(28, width * 0.08)
        painter.drawLine(
            facade.topRight(),
            facade.topRight() + QPointF(depth, -16),
        )
        painter.drawLine(
            facade.bottomRight(),
            facade.bottomRight() + QPointF(depth, -16),
        )
        painter.drawLine(
            facade.topRight() + QPointF(depth, -16),
            facade.bottomRight() + QPointF(depth, -16),
        )

        columns, rows = 7, 5
        painter.setPen(QPen(QColor('#D6DFEB'), 1))
        for column in range(1, columns):
            px = facade.left() + facade.width() * column / columns
            painter.drawLine(int(px), int(facade.top()), int(px), int(facade.bottom()))
        for row in range(1, rows):
            py = facade.top() + facade.height() * row / rows
            painter.drawLine(int(facade.left()), int(py), int(facade.right()), int(py))

        # 在立面网格交点绘制确定性的点云采样，避免伪造业务检测结果。
        point_color = QColor('#3B82F6')
        point_color.setAlpha(145 if self.variant == 'heatmap' else 105)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(point_color)
        for row in range(rows + 1):
            for column in range(columns + 1):
                px = facade.left() + facade.width() * column / columns
                py = facade.top() + facade.height() * row / rows
                jitter_x = ((row * 7 + column * 3) % 5) - 2
                jitter_y = ((row * 5 + column * 11) % 5) - 2
                radius = 2 if (row + column) % 3 else 2.6
                painter.drawEllipse(
                    QRectF(
                        px + jitter_x - radius,
                        py + jitter_y - radius,
                        radius * 2,
                        radius * 2,
                    )
                )

        if self.message:
            text_font = QFont(self.font())
            text_font.setPixelSize(16)
            text_font.setWeight(QFont.Weight.DemiBold)
            painter.setFont(text_font)
            painter.setPen(QColor('#334155'))
            text_rect = QRectF(
                facade.left(),
                facade.bottom() + 22,
                facade.width(),
                28,
            )
            painter.drawText(
                text_rect,
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                self.message,
            )

    def _draw_document(self, painter, rect):
        width = min(240, max(150, int(rect.width() * 0.22)))
        height = min(300, max(190, int(rect.height() * 0.54)))
        x = rect.center().x() - width / 2
        y = rect.center().y() - height / 2
        page = QRectF(x, y, width, height)

        painter.setBrush(QColor('#FFFFFF'))
        painter.setPen(QPen(QColor('#B9C7DA'), 1.4))
        painter.drawRoundedRect(page, 8, 8)
        painter.setPen(QPen(QColor('#D6DFEB'), 2))
        line_left = int(page.left() + 28)
        line_right = int(page.right() - 28)
        for offset, ratio in ((48, 1.0), (78, 0.72), (108, 0.88), (154, 1.0)):
            painter.drawLine(
                line_left,
                int(page.top() + offset),
                int(line_left + (line_right - line_left) * ratio),
                int(page.top() + offset),
            )
        painter.setPen(QPen(QColor('#3B82F6'), 3))
        painter.drawLine(
            line_left,
            int(page.top() + 24),
            int(page.left() + width * 0.48),
            int(page.top() + 24),
        )
