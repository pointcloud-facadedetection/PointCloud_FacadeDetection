from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from PySide6.QtCore import Qt
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
)
class FacadeQualityDialog(QDialog):
    """
    立面质量评估对话框：
    - 展示总体指标（垂直度、最大局部间隙、合格率、点数）
    - 展示按 20m 区间聚合的简表（每区间：gap、合格率、窗口统计）
    - 操作按钮：显示检测效果（应用质量颜色）、恢复原始颜色、关闭
    """

    def __init__(self, parent: Optional[QWidget], facade_label: str, quality_result: dict,
                 on_show_colors: Optional[Callable[[], None]] = None,
                 on_restore_colors: Optional[Callable[[], None]] = None):
        super().__init__(parent)
        self.setWindowTitle(f"立面质量评估 - {facade_label}")
        self.resize(900, 600)
        self.setStyleSheet("""
            QDialog { background: #f3f5f8; }
            QLabel#qualityHeader { color: #1f4f7a; font-size: 18px; font-weight: 700; }
            QLabel#qualityMeta { background: white; border: 1px solid #dce2e8;
                border-radius: 6px; padding: 12px; color: #46515d; }
            QTableWidget { background: white; border: 1px solid #dce2e8;
                gridline-color: #edf0f3; }
            QHeaderView::section { background: #eaf0f6; padding: 7px; border: 0;
                color: #29445d; font-weight: 600; }
            QPushButton { min-height: 32px; padding: 0 14px; border-radius: 4px;
                border: 1px solid #b8c7d6; background: white; }
            QPushButton:hover { background: #eaf3fb; border-color: #3975a8; }
        """)
        self._quality = quality_result or {}
        self._on_show_colors = on_show_colors
        self._on_restore_colors = on_restore_colors

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        overall = self._quality.get('overall') or {}
        title = QLabel("质量评估结果")
        title.setObjectName('qualityHeader')
        title.setStyleSheet('font-weight:600; color:#303641;')
        layout.addWidget(title)

        gap = float(overall.get('flatness_max_gap_mm', 0.0) or 0.0)
        vangle = float(overall.get('verticality_max_angle_deg', 0.0) or 0.0)
        vgap = float(overall.get('verticality_max_deviation_mm', 0.0) or 0.0)
        flat_rate = float(overall.get('flatness_pass_rate', 0.0) or 0.0) * 100.0
        vert_rate = float(overall.get('verticality_pass_rate', 0.0) or 0.0) * 100.0
        pts = int(overall.get('point_count') or 0)
        profile = self._quality.get('profile_snapshot') or {}
        standard = f"标准：{profile.get('standard_name', '未指定')} {profile.get('version', '')}"
        meta = QLabel(
            f"{standard}\n"
            f"平整度最大间隙: {gap:.2f} mm    垂直度最大偏差角: {vangle:.3f}°    "
            f"垂直度2m最大偏差: {vgap:.2f} mm    平整度合格率: {flat_rate:.1f}%    "
            f"垂直度合格率: {vert_rate:.1f}%    点数: {pts}"
        )
        meta.setObjectName('qualityMeta')
        layout.addWidget(meta)

        table_title = QLabel("区间汇总")
        table_title.setStyleSheet('font-weight:600; color:#303641;')
        layout.addWidget(table_title)

        interval_size = float(self._quality.get('interval_size_m', 0.0))
        z_min = float(self._quality.get('z_min_m', 0.0) or 0.0)
        z_max = float(self._quality.get('z_max_m', 0.0) or 0.0)
        interval_count = int(self._quality.get('interval_count', 0) or 0)
        layout.addWidget(QLabel(
            f"区间：{interval_size:g}m，共{interval_count}个）"
        ))
        self._table = table = QTableWidget(0, 8, self)
        table.setObjectName('tblQualityGrids')
        table.setHorizontalHeaderLabels([
            "区间", "点数", "窗口数", "平整度最大间隙(mm)",
            "平整度合格率(%)", "垂直度最大偏差角(°)",
            "垂直度最大偏差(mm)", "垂直度合格率(%)"
        ])
        self._refresh_intervals()
        table.resizeColumnsToContents()
        layout.addWidget(table, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._mode_combo = QComboBox(self)
        self._mode_combo.addItem('平整度效果', 'flatness')
        self._mode_combo.addItem('垂直度效果', 'verticality')
        btn_show = QPushButton("显示检测效果")
        btn_restore = QPushButton("恢复原始颜色")
        btn_close = QPushButton("关闭")
        btn_row.addWidget(self._mode_combo)
        btn_row.addWidget(btn_show)
        btn_row.addWidget(btn_restore)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        btn_close.clicked.connect(self.close)
        if callable(self._on_show_colors):
            btn_show.clicked.connect(lambda: self._on_show_colors(self._mode_combo.currentData() or 'flatness'))
        else:
            btn_show.setEnabled(False)
        if callable(self._on_restore_colors):
            btn_restore.clicked.connect(self._on_restore_colors)
        else:
            btn_restore.setEnabled(False)

    def _refresh_intervals(self):
        intervals = self._quality.get('intervals') or []
        self._table.setRowCount(len(intervals))
        for r, item in enumerate(intervals):
            vals = [
                str(item.get('label') or (
                    f"{float(item.get('z_min_m', 0.0)):.2f}–"
                    f"{float(item.get('z_max_m', 0.0)):.2f}m"
                )),
                str(item.get('point_count', 0)),
                str(item.get('window_count', 0)),
                f"{float(item.get('flatness_max_gap_mm', 0)):.2f}",
                f"{float(item.get('flatness_pass_rate', 0))*100:.1f}",
                f"{float(item.get('verticality_max_angle_deg', 0)):.3f}",
                f"{float(item.get('verticality_max_deviation_mm_2m', 0)):.2f}",
                f"{float(item.get('verticality_pass_rate', 0))*100:.1f}",
            ]
            for c, value in enumerate(vals):
                self._table.setItem(r, c, QTableWidgetItem(value))
        self._table.resizeColumnsToContents()
