from __future__ import annotations

from typing import Callable, Optional

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
        self.resize(760, 520)
        self._quality = quality_result or {}
        self._on_show_colors = on_show_colors
        self._on_restore_colors = on_restore_colors

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        overall = self._quality.get('overall') or {}
        title = QLabel("总体指标")
        title.setStyleSheet('font-weight:600; color:#303641;')
        layout.addWidget(title)

        vdeg = float(overall.get('verticality') or 0.0)
        gap = float(overall.get('gap') or 0.0) * 1000.0  # m -> mm (if provided as meters)
        rate = float(overall.get('compliance_rate') or 0.0) * 100.0
        pts = int(overall.get('point_count') or 0)
        meta = QLabel(f"垂直度: {vdeg:.3f}°    最差局部间隙: {gap:.2f} mm    合格率: {rate:.1f}%    点数: {pts}")
        meta.setStyleSheet('color:#525a66;')
        layout.addWidget(meta)

        table_title = QLabel("区间汇总（20m）")
        table_title.setStyleSheet('font-weight:600; color:#303641;')
        layout.addWidget(table_title)

        table = QTableWidget(0, 6, self)
        table.setObjectName('tblQualityGrids')
        table.setHorizontalHeaderLabels(["区间ID", "点数", "gap(mm)", "合格率(%)", "窗口总数", "合格窗口"])
        grids = self._quality.get('grids') or []
        table.setRowCount(len(grids))
        for r, g in enumerate(grids):
            def seti(c, val):
                item = QTableWidgetItem(val)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                table.setItem(r, c, item)
            gid = int(g.get('grid_id') or r)
            seti(0, str(gid))
            seti(1, str(int(g.get('point_count') or 0)))
            # gap provided in meters from algorithm; display in millimeters
            seti(2, f"{float(g.get('gap') or 0.0) * 1000.0:.2f}")
            seti(3, f"{float(g.get('compliance_rate') or 0.0) * 100.0:.1f}")
            seti(4, str(int(g.get('total_windows') or 0)))
            seti(5, str(int(g.get('compliant_windows') or 0)))
        table.resizeColumnsToContents()
        layout.addWidget(table, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_show = QPushButton("显示检测效果")
        btn_restore = QPushButton("恢复原始颜色")
        btn_close = QPushButton("关闭")
        btn_row.addWidget(btn_show)
        btn_row.addWidget(btn_restore)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        btn_close.clicked.connect(self.close)
        if callable(self._on_show_colors):
            btn_show.clicked.connect(self._on_show_colors)
        else:
            btn_show.setEnabled(False)
        if callable(self._on_restore_colors):
            btn_restore.clicked.connect(self._on_restore_colors)
        else:
            btn_restore.setEnabled(False)
