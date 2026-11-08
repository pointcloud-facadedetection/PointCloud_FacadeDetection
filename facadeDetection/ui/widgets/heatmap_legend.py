from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from services.heatmap_spec import bipolar_colormap, defect_colormap


class HeatmapLegendWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.metadata = None
        self.setMinimumSize(190, 78)
        self.setToolTip('颜色与导出热力图一致；数值为当前立面实际毫米范围')

    def set_metadata(self, metadata):
        self.metadata = dict(metadata) if metadata else None
        self.update()

    def clear_metadata(self):
        self.set_metadata(None)

    def sizeHint(self):
        return QSize(210, 74)

    def _gradient(self, width):
        method = self.metadata.get('method', 'ruler')
        limit_mm = float(self.metadata.get('limit_mm', 4.0))
        if method == 'global_plane':
            vmax_mm = self.metadata.get('vmax_mm')
            vmax_mm = float(vmax_mm) if vmax_mm is not None else limit_mm * 1.2
            limit_m = limit_mm / 1000.0
            vmax_m = vmax_mm / 1000.0
            scale_mm = max(vmax_m - limit_m, limit_m * 0.15, 1e-6) * 1000.0
            values = np.linspace(-vmax_mm, vmax_mm, width)
            return bipolar_colormap(values, limit_mm, scale_mm=scale_mm)
        return defect_colormap(np.linspace(0.0, 1.0, width))

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if not self.metadata:
            painter.setPen(QColor('#64748B'))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, '暂无比例尺')
            painter.end()
            return

        width = max(2, self.width() - 12)
        colors = self._gradient(width)
        for index, rgb in enumerate(colors):
            painter.setPen(QPen(QColor.fromRgbF(*[float(value) for value in rgb])))
            painter.drawLine(6 + index, 6, 6 + index, 20)

        method = self.metadata.get('method', 'ruler')
        metric = self.metadata.get('metric', 'flatness')
        limit_mm = float(self.metadata.get('limit_mm', 4.0))
        vmax = self.metadata.get('vmax_mm')
        painter.setPen(QColor('#334155'))
        if method == 'global_plane':
            maximum = float(vmax) if vmax is not None else limit_mm
            for rect, text, alignment in (
                (QRectF(0, 24, self.width() / 3, 14),
                 f'-{maximum:.4g}', Qt.AlignmentFlag.AlignLeft),
                (QRectF(self.width() / 3, 24, self.width() / 3, 14),
                 '0', Qt.AlignmentFlag.AlignCenter),
                (QRectF(self.width() * 2 / 3, 24, self.width() / 3, 14),
                 f'+{maximum:.4g}', Qt.AlignmentFlag.AlignRight),
                (QRectF(0, 38, self.width() / 3, 14),
                 f'-{limit_mm:.4g}', Qt.AlignmentFlag.AlignLeft),
                (QRectF(self.width() / 3, 38, self.width() / 3, 14),
                 '阈值 mm', Qt.AlignmentFlag.AlignCenter),
                (QRectF(self.width() * 2 / 3, 38, self.width() / 3, 14),
                 f'+{limit_mm:.4g}', Qt.AlignmentFlag.AlignRight),
            ):
                painter.drawText(rect, alignment, text)
        else:
            left = f'限值 {limit_mm:.4g}'
            right = f'最大 {float(vmax):.4g}' if vmax is not None else '超限'
            painter.drawText(QRectF(0, 26, self.width() / 3, 16),
                             Qt.AlignmentFlag.AlignLeft, left)
            painter.drawText(QRectF(self.width() / 3, 26, self.width() / 3, 16),
                             Qt.AlignmentFlag.AlignCenter, f'{limit_mm:.4g} mm')
            painter.drawText(QRectF(self.width() * 2 / 3, 26, self.width() / 3, 16),
                             Qt.AlignmentFlag.AlignRight, right)
        painter.setPen(QColor('#64748B'))
        if method == 'global_plane':
            positive_text = '外倾' if metric == 'verticality' else '凸起'
            negative_text = '内陷' if metric == 'verticality' else '凹陷'
            note = f'蓝色={positive_text}  红色={negative_text}'
        else:
            note = '颜色表示偏差量（mm）'
        painter.drawText(QRectF(0, 56, self.width(), 18),
                         Qt.AlignmentFlag.AlignCenter, note)
        painter.setPen(QPen(QColor('#CBD5E1'), 1))
        painter.drawRect(6, 6, max(1, self.width() - 12), 14)
        painter.end()
