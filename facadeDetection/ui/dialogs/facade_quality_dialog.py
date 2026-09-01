from __future__ import annotations

from typing import Callable, Optional
import math
import numpy as np

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QImageReader, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QHBoxLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
    QComboBox,
    QMessageBox,
    QGroupBox,
    QGridLayout,
    QFrame,
    QScrollArea,
    QHeaderView,
)


class FacadeQualityDialog(QDialog):
    """
    立面质量评估对话框 v6（修复版）：
    - 修复垂直度字段读取问题
    - 修复 interval 垂直度统计显示
    - 确保所有字段正确映射
    """

    def __init__(self, parent: Optional[QWidget], facade_label: str, quality_result: dict,
                 on_show_colors: Optional[Callable[[], None]] = None,
                 on_restore_colors: Optional[Callable[[], None]] = None,
                 project_name: str = ''):
        super().__init__(parent)

        title = f"{project_name} - {facade_label} 质量评估" if project_name else f"{facade_label} 质量评估"
        self.setWindowTitle(title)
        self.setMinimumSize(720, 620)
        self.resize(900, 780)
        self.setMaximumWidth(1200)

        self.setStyleSheet("""
            QDialog { background: #f8fafc; }
            QLabel#qualityHeader { 
                color: #1e293b; 
                font-size: 16px; 
                font-weight: 700; 
            }
            QLabel#summaryBanner { 
                background: #ffffff; 
                border: 1px solid #e2e8f0;
                border-radius: 8px; 
                padding: 12px 16px; 
                color: #334155; 
                font-weight: 600;
                font-size: 13px;
            }
            QTableWidget { 
                background: white; 
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                gridline-color: #f1f5f9; 
            }
            QHeaderView::section { 
                background: #f1f5f9; 
                padding: 8px 6px; 
                border: 0;
                color: #475569; 
                font-weight: 600; 
                font-size: 11px;
            }
            QTableWidget::item { 
                padding: 6px 4px; 
                font-size: 12px;
                color: #334155;
            }
            QPushButton { 
                min-height: 32px; 
                padding: 0 16px; 
                border-radius: 6px;
                border: 1px solid #cbd5e1; 
                background: #ffffff;
                color: #334155;
                font-size: 12px;
            }
            QPushButton:hover { 
                background: #f1f5f9; 
                border-color: #94a3b8; 
            }
            QPushButton:disabled { 
                background: #f1f5f9; 
                color: #94a3b8; 
                border-color: #e2e8f0; 
            }
            QPushButton#primaryBtn {
                background: #3b82f6;
                color: white;
                border-color: #3b82f6;
            }
            QPushButton#primaryBtn:hover {
                background: #2563eb;
                border-color: #2563eb;
            }
            QGroupBox { 
                border: 1px solid #e2e8f0; 
                border-radius: 8px; 
                margin-top: 8px; 
                padding-top: 8px;
                background: #ffffff;
            }
            QGroupBox::title { 
                subcontrol-origin: margin; 
                left: 10px; 
                padding: 0 6px;
                color: #64748b; 
                font-size: 11px; 
                font-weight: 600;
            }
            QLabel.metricLabel {
                color: #64748b;
                font-size: 11px;
            }
            QLabel.metricValue {
                color: #1e293b;
                font-size: 12px;
                font-weight: 600;
            }
            QComboBox {
                min-height: 28px;
                padding: 2px 8px;
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                background: white;
                font-size: 12px;
            }
        """)

        self._quality = quality_result or {}
        self._on_show_colors = on_show_colors
        self._on_restore_colors = on_restore_colors

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)

        # ── 标题 ──
        title = QLabel("质量评估结果")
        title.setObjectName('qualityHeader')
        layout.addWidget(title)

        # ── 状态横幅（错误时显示） ──
        ok = self._quality.get('ok', True)
        reason = self._quality.get('reason', '')
        message = self._quality.get('message', '')

        if not ok:
            status_box = QGroupBox("计算状态")
            status_layout = QVBoxLayout(status_box)
            status_layout.setContentsMargins(12, 8, 12, 8)
            error_label = QLabel(f"⚠️ {message or reason or '质量计算未完成'}")
            error_label.setStyleSheet('color: #dc2626; font-weight: 600; padding: 4px; font-size: 12px;')
            error_label.setWordWrap(True)
            status_layout.addWidget(error_label)
            layout.addWidget(status_box)

        # ── 核心摘要：评估标准与合格率 ──
        overall = self._quality.get('overall') or {}
        profile = self._quality.get('profile_snapshot') or {}

        # FIX: Safely get rates with proper defaults
        flat_rate = float(overall.get('flatness_pass_rate', 0.0) or 0.0) * 100.0
        vert_rate = float(overall.get('verticality_pass_rate', 0.0) or 0.0) * 100.0
        quality_rate = (flat_rate + vert_rate) / 2.0

        standard_name = profile.get('standard_name', '未指定')
        version = profile.get('version', '')
        standard_text = f"{standard_name} {version}".strip()

        summary = QLabel(
            f"当前评估标准：{standard_text}　|　"
            f"平整度合格率：{flat_rate:.1f}%　|　"
            f"垂直度合格率：{vert_rate:.1f}%　|　"
            f"综合合格率：{quality_rate:.1f}%"
        )
        summary.setObjectName('summaryBanner')
        summary.setWordWrap(True)
        layout.addWidget(summary)

        # ── 详细指标网格 ──
        metrics_frame = QFrame()
        metrics_frame.setStyleSheet("""
            QFrame { 
                background: #ffffff; 
                border: 1px solid #e2e8f0; 
                border-radius: 8px; 
            }
        """)
        metrics_layout = QGridLayout(metrics_frame)
        metrics_layout.setContentsMargins(14, 12, 14, 12)
        metrics_layout.setHorizontalSpacing(20)
        metrics_layout.setVerticalSpacing(10)

        # FIX: Safely extract all metrics with proper fallbacks
        gap = float(overall.get('flatness_max_gap_mm', 0.0) or 0.0)
        raw_gap = float(overall.get('flatness_raw_max_gap_mm', gap) or gap)

        # FIX: Try multiple field names for verticality angle
        vangle_raw = (overall.get('verticality_max_angle_deg') 
                      or overall.get('verticality_angle_deg'))
        vangle = float(vangle_raw) if vangle_raw is not None and np.isfinite(float(vangle_raw)) else None

        # FIX: Try multiple field names for verticality deviation
        vgap_raw = (overall.get('verticality_max_deviation_mm'))
        vgap = float(vgap_raw) if vgap_raw is not None and np.isfinite(float(vgap_raw)) else None

        pts = int(overall.get('point_count') or 0)

        n_candidates = int(overall.get('candidate_window_count', 0))
        n_geometry = int(overall.get('geometry_valid_window_count', 0))
        n_quality = int(overall.get('quality_valid_window_count', 0))

        metrics = [
            ('平整度有效最大间隙', f'{gap:.2f} mm'),
            ('平整度原始最大间隙', f'{raw_gap:.2f} mm'),
            ('垂直度最大偏差角', f'{vangle:.3f}°' if vangle is not None else '--'),
            ('垂直度最大偏差', f'{vgap:.2f} mm' if vgap is not None else '--'),
            ('检测窗口总数', f'{n_candidates}'),
            ('有效窗口', f'{n_geometry}'),
            ('合格窗口', f'{n_quality}'),
        ]

        for i, (label_text, value_text) in enumerate(metrics):
            row, col = divmod(i, 4)
            label = QLabel(label_text)
            label.setProperty('class', 'metricLabel')
            label.setStyleSheet('color: #64748b; font-size: 11px;')
            value = QLabel(value_text)
            value.setProperty('class', 'metricValue')
            value.setStyleSheet('color: #1e293b; font-size: 12px; font-weight: 600;')
            metrics_layout.addWidget(label, row, col * 2)
            metrics_layout.addWidget(value, row, col * 2 + 1)

        layout.addWidget(metrics_frame)

        # ── 区间表格 ──
        interval_size = float(self._quality.get('interval_size_m', 0.0))
        interval_count = int(self._quality.get('interval_count', 0) or 0)

        interval_header = QLabel(f"区间统计　|　区间尺寸：{interval_size:g}m　|　共 {interval_count} 个区间")
        interval_header.setStyleSheet('color: #475569; font-size: 12px; font-weight: 600; margin-top: 4px;')
        layout.addWidget(interval_header)

        self._table = table = QTableWidget(0, 8, self)
        table.setObjectName('tblQualityGrids')
        table.setHorizontalHeaderLabels([
            "区间高度", "点数", "窗口数", "合格数",
            "平整度最大间隙(mm)", "平整度合格率(%)",
            "垂直度最大偏差(mm)", "垂直度合格率(%)"
        ])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        table.horizontalHeader().setStretchLastSection(False)
        table.horizontalHeader().setDefaultSectionSize(90)
        table.horizontalHeader().setMinimumSectionSize(70)
        table.setColumnWidth(0, 100)
        table.setColumnWidth(1, 70)
        table.setColumnWidth(2, 70)
        table.setColumnWidth(3, 70)
        table.setColumnWidth(4, 130)
        table.setColumnWidth(5, 110)
        table.setColumnWidth(6, 130)
        table.setColumnWidth(7, 110)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        self._refresh_intervals()
        layout.addWidget(table, 1)

        self._heatmap_caption = QLabel('点击「显示检测效果」后，热力图将显示在此处，并同步到着色三维点云。')
        self._heatmap_caption.setStyleSheet('color: #64748b; font-size: 12px;')
        layout.addWidget(self._heatmap_caption)

        self._heatmap_scroll = QScrollArea()
        self._heatmap_scroll.setWidgetResizable(False)
        self._heatmap_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._heatmap_scroll.setMinimumHeight(220)
        self._heatmap_scroll.setMaximumHeight(380)
        self._heatmap_scroll.setStyleSheet(
            'QScrollArea { background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 6px; }')
        self._heatmap_image = QLabel()
        self._heatmap_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._heatmap_scroll.setWidget(self._heatmap_image)
        self._heatmap_scroll.setVisible(False)
        layout.addWidget(self._heatmap_scroll)

        # ── 底部按钮栏 ──
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch(1)

        self._mode_combo = QComboBox(self)
        self._mode_combo.addItem('平整度效果', 'flatness')
        self._mode_combo.addItem('垂直度效果', 'verticality')
        self._mode_combo.setMinimumWidth(130)

        btn_show = QPushButton("显示检测效果")
        btn_show.setObjectName('primaryBtn')
        btn_restore = QPushButton("恢复原始颜色")
        btn_close = QPushButton("关闭")

        btn_row.addWidget(self._mode_combo)
        btn_row.addWidget(btn_show)
        btn_row.addWidget(btn_restore)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        btn_close.clicked.connect(self.close)

        has_valid_windows = bool(ok and n_quality > 0)

        if callable(self._on_show_colors) and has_valid_windows:
            def _on_show_clicked():
                mode = self._mode_combo.currentData() or 'flatness'
                path = self._on_show_colors(mode)
                if isinstance(path, str) and path:
                    self.set_heatmap_preview(path)
            btn_show.clicked.connect(_on_show_clicked)
        else:
            btn_show.setEnabled(False)
            if not ok:
                btn_show.setToolTip('质量计算未成功完成')
            elif n_geometry == 0:
                btn_show.setToolTip('没有几何有效的检测窗口')
            else:
                btn_show.setToolTip('无可用的颜色渲染回调')

        if callable(self._on_restore_colors):
            btn_restore.clicked.connect(self._on_restore_colors)
        else:
            btn_restore.setEnabled(False)

    def set_heatmap_preview(self, image_path: str) -> None:
        reader = QImageReader(image_path)
        source = reader.size()
        box_w = max(self._heatmap_scroll.viewport().width(), 640)
        box_h = 360
        if source.isValid() and source.width() > 0 and source.height() > 0:
            if source.height() > source.width() * 1.25:
                # 高层立面通常是狭长图。按高度塞进预览框会导致图例和立面
                # 缩成一条细线；此处保持可读宽度，使用滚动条查看完整高度。
                target_w = min(source.width(), box_w)
                target_h = max(
                    1,
                    int(round(source.height() * target_w / source.width())),
                )
                reader.setScaledSize(QSize(target_w, target_h))
            else:
                reader.setScaledSize(
                    source.scaled(
                        QSize(box_w, box_h),
                        Qt.AspectRatioMode.KeepAspectRatio,
                    )
                )
        image = reader.read()
        if image.isNull():
            self._heatmap_caption.setText(
                '热力图已生成，但无法在此加载预览。请查看三维视口中的灰-黄-红着色。')
            return
        pixmap = QPixmap.fromImage(image)
        self._heatmap_image.setPixmap(pixmap)
        self._heatmap_image.adjustSize()
        self._heatmap_scroll.setVisible(True)
        self._heatmap_caption.setText(
            '检测效果热力图：灰色接近 0，黄色为标准限值，红色表示超限；'
            '狭长立面可上下滚动查看。')

    def _refresh_intervals(self):
        intervals = self._quality.get('intervals') or []
        self._table.setRowCount(len(intervals))

        def number(value, suffix=''):
            try:
                if value is None:
                    return '--'
                value = float(value)
                return f'{value:.2f}{suffix}' if math.isfinite(value) else '--'
            except (TypeError, ValueError):
                return '--'

        for r, item in enumerate(intervals):
            # FIX: Safely get verticality values with proper fallbacks
            vert_max = item.get('verticality_max_deviation_mm')

            vert_rate = item.get('verticality_pass_rate')

            vals = [
                str(item.get('label') or (
                    f"{float(item.get('v_min_m', 0.0)):.2f}–"
                    f"{float(item.get('v_max_m', 0.0)):.2f}m"
                )),
                str(item.get('point_count', 0)),
                str(item.get('window_count', 0)),
                str(item.get('valid_window_count', 0)),
                number(item.get('flatness_max_gap_mm')),
                number((float(item.get('flatness_pass_rate')) * 100)
                       if item.get('flatness_pass_rate') is not None else None),
                number(vert_max),
                number((float(vert_rate) * 100) if vert_rate is not None else None),
            ]
            for c, value in enumerate(vals):
                cell = QTableWidgetItem(value)
                cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._table.setItem(r, c, cell)

        self._table.resizeColumnsToContents()
        # Ensure columns don't exceed reasonable widths after resize
        for c in range(self._table.columnCount()):
            w = self._table.columnWidth(c)
            self._table.setColumnWidth(c, min(w, 160))