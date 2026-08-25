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
    QMessageBox,
    QGroupBox,
)

class FacadeQualityDialog(QDialog):
    """
    立面质量评估对话框 v4：
    - 展示总体指标（含geometry_valid/quality_valid区分，去重后统计）
    - 展示按区间聚合的简表（含status列）
    - 操作按钮：显示检测效果、恢复原始颜色、关闭
    - 兼容成功结果、错误结果、空结果
    - FIX: 使用 facade_no 替代 facade_id，统一标识
    """

    def __init__(self, parent: Optional[QWidget], facade_label: str, quality_result: dict,
                 on_show_colors: Optional[Callable[[], None]] = None,
                 on_restore_colors: Optional[Callable[[], None]] = None):
        super().__init__(parent)
        self.setWindowTitle(f"立面质量评估 - {facade_label}")
        self.resize(1000, 700)
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
            QPushButton:disabled { background: #e8e8e8; color: #999; border-color: #ccc; }
            QGroupBox { border: 1px solid #dce2e8; border-radius: 6px; 
                margin-top: 8px; padding-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px;
                color: #666; font-size: 11px; }
        """)
        self._quality = quality_result or {}
        self._on_show_colors = on_show_colors
        self._on_restore_colors = on_restore_colors

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # Title
        title = QLabel("质量评估结果")
        title.setObjectName('qualityHeader')
        title.setStyleSheet('font-weight:600; color:#303641;')
        layout.addWidget(title)

        # Status banner
        ok = self._quality.get('ok', True)
        reason = self._quality.get('reason', '')
        message = self._quality.get('message', '')

        if not ok:
            status_box = QGroupBox("计算状态")
            status_layout = QVBoxLayout(status_box)
            error_label = QLabel(f"⚠️ {message or reason or '质量计算未完成'}")
            error_label.setStyleSheet('color: #c0392b; font-weight: 600; padding: 4px;')
            status_layout.addWidget(error_label)

            proj = self._quality.get('projection', {})
            if proj:
                diag = QLabel(
                    f"投影范围：U={proj.get('u_range_m', 0):.3f}m, "
                    f"V={proj.get('v_range_m', 0):.3f}m | "
                    f"U轴=[{proj.get('u_min_m', 0):.2f}, {proj.get('u_max_m', 0):.2f}] | "
                    f"V轴=[{proj.get('v_min_m', 0):.2f}, {proj.get('v_max_m', 0):.2f}]"
                )
                diag.setStyleSheet('color: #666; font-size: 11px;')
                status_layout.addWidget(diag)

            dir_ranges = self._quality.get('direction_ranges', {})
            if dir_ranges:
                dir_text = ' | '.join(
                    f"{k}°: 沿={v.get('along_m', 0):.2f}m 横={v.get('across_m', 0):.2f}m"
                    for k, v in dir_ranges.items()
                )
                dir_label = QLabel(f"方向范围：{dir_text}")
                dir_label.setStyleSheet('color: #666; font-size: 11px;')
                dir_label.setWordWrap(True)
                status_layout.addWidget(dir_label)

            layout.addWidget(status_box)

        # Overall metrics
        overall = self._quality.get('overall') or {}
        gap = float(overall.get('flatness_max_gap_mm', 0.0) or 0.0)
        vangle_raw = overall.get('verticality_max_angle_deg')
        vgap_raw = overall.get('verticality_max_deviation_mm')
        vangle = float(vangle_raw) if vangle_raw is not None and np.isfinite(vangle_raw) else None
        vgap = float(vgap_raw) if vgap_raw is not None and np.isfinite(vgap_raw) else None
        flat_rate = float(overall.get('flatness_pass_rate', 0.0) or 0.0) * 100.0
        quality_rate = float(overall.get('quality_pass_rate', 0.0) or 0.0) * 100.0
        vert_rate = float(overall.get('verticality_pass_rate', 0.0) or 0.0) * 100.0
        pts = int(overall.get('point_count') or 0)

        # Window counts with new classification - support both old and new keys
        # FIX: Use deduplicated counts if available, fallback to raw counts
        n_candidates = int(overall.get('candidate_unique_count', 
                                       overall.get('candidate_window_count', 0)))
        n_geometry = int(overall.get('geometry_valid_unique_count',
                                     overall.get('geometry_valid_window_count', 0)))
        n_quality = int(overall.get('quality_valid_unique_count',
                                    overall.get('quality_valid_window_count', 0)))
        n_failed = int(overall.get('failed_window_count', 0))
        n_intervals = len(self._quality.get('intervals', []))

        profile = self._quality.get('profile_snapshot') or {}
        standard = f"标准：{profile.get('standard_name', '未指定')} {profile.get('version', '')}"

        meta_text = f"{standard}"
        if vangle is not None:
            meta_text += f"平整度最大间隙: {gap:.2f} mm    垂直度最大偏差角: {vangle:.3f}°    "
        else:
            meta_text += f"平整度最大间隙: {gap:.2f} mm    垂直度最大偏差角: --    "

        if vgap is not None:
            meta_text += f"垂直度2m最大偏差: {vgap:.2f} mm    "
        else:
            meta_text += "垂直度2m最大偏差: --    "

        meta_text += (
            f"平整度合格率: {flat_rate:.1f}%    "
            f"质量合格率: {quality_rate:.1f}%    垂直度合格率: {vert_rate:.1f}%    点数: {pts}"
            f"候选窗口: {n_candidates}    几何有效: {n_geometry}    "
            f"质量有效: {n_quality}    失败: {n_failed}    区间数: {n_intervals}"
        )

        meta = QLabel(meta_text)
        meta.setObjectName('qualityMeta')
        layout.addWidget(meta)

        # Only render pre-aggregated interval statistics. Never materialise or
        # scan the potentially hundreds of thousands of window records here.
        interval_widget = QWidget()
        interval_layout = QVBoxLayout(interval_widget)
        interval_layout.setContentsMargins(0, 0, 0, 0)

        interval_size = float(self._quality.get('interval_size_m', 0.0))
        interval_count = int(self._quality.get('interval_count', 0) or 0)
        interval_layout.addWidget(QLabel(
            f"区间：{interval_size:g}m，共{interval_count}个"
        ))

        self._table = table = QTableWidget(0, 10, self)
        table.setObjectName('tblQualityGrids')
        table.setHorizontalHeaderLabels([
            "区间", "状态", "点数", "窗口数", "有效窗口",
            "平整度最大间隙(mm)", "平整度合格率(%)", "质量合格率(%)",
            "垂直度最大偏差角(°)", "垂直度最大偏差(mm)"
        ])
        self._refresh_intervals()
        table.resizeColumnsToContents()
        interval_layout.addWidget(table, 1)
        layout.addWidget(interval_widget, 1)

        # Buttons
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

        has_valid_windows = bool(ok and n_quality > 0)

        if callable(self._on_show_colors) and has_valid_windows:
            btn_show.clicked.connect(
                lambda: self._on_show_colors(self._mode_combo.currentData() or 'flatness'))
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

    def _refresh_intervals(self):
        intervals = self._quality.get('intervals') or []
        self._table.setRowCount(len(intervals))
        for r, item in enumerate(intervals):
            status = item.get('status', 'ok')
            status_text = '✓' if status == 'ok' else ('⚠ ' + status if status != 'ok' else '✓')

            vals = [
                str(item.get('label') or (
                    f"{float(item.get('v_min_m', 0.0)):.2f}–"
                    f"{float(item.get('v_max_m', 0.0)):.2f}m"
                )),
                status_text,
                str(item.get('point_count', 0)),
                str(item.get('window_count', 0)),
                str(item.get('valid_window_count', 0)),
                f"{float(item.get('flatness_max_gap_mm', 0)):.2f}",
                f"{float(item.get('flatness_pass_rate', 0))*100:.1f}",
                f"{float(item.get('quality_pass_rate', 0))*100:.1f}",
                f"{float(item.get('verticality_max_angle_deg', 0)):.3f}",
                f"{float(item.get('verticality_max_deviation_mm_2m', 0)):.2f}",
            ]
            for c, value in enumerate(vals):
                cell = QTableWidgetItem(value)
                if status != 'ok':
                    cell.setForeground(Qt.GlobalColor.gray)
                self._table.setItem(r, c, cell)
        self._table.resizeColumnsToContents()
