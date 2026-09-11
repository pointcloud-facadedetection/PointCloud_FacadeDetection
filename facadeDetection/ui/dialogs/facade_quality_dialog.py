from __future__ import annotations

from typing import Callable, Optional
import math
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
    QGridLayout,
    QFrame,
    QHeaderView,
)


class FacadeQualityDialog(QDialog):
    """
    立面质量评估对话框 v7（重构版）：
      - 顶部：评估标准 + 阈值 + 立面面积
      - 中部：靠尺法检测结果 + 区间表格
      - 下部：全局平面基准法检测结果 + 区间表格
      - 区间合格率统一使用面积合格率
      - 渲染模式下拉框仅保留 4 组面积模式
    """

    def __init__(self, parent: Optional[QWidget], facade_label: str, quality_result: dict,
                 on_show_colors: Optional[Callable[[], None]] = None,
                 on_restore_colors: Optional[Callable[[], None]] = None,
                 project_name: str = ''):
        super().__init__(parent)

        title = f"{project_name} - {facade_label} 质量评估" if project_name else f"{facade_label} 质量评估"
        self.setWindowTitle(title)
        self.setMinimumSize(780, 580)
        self.resize(980, 720)
        self.setMaximumWidth(1400)

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

        # ── 数据源 ──
        overall = self._quality.get('overall') or {}
        profile = self._quality.get('profile_snapshot') or {}
        comparison = self._quality.get('quality_comparison') or {}
        methods = comparison.get('methods') or {}

        standard_name = profile.get('standard_name', '未指定')
        version = profile.get('version', '')
        standard_text = f"{standard_name} {version}".strip()

        flat_limit = float(profile.get('flatness_limit_mm', 4.0))
        vert_limit = float(profile.get('verticality_limit_mm', 4.0))
        facade_area = float(overall.get('area_m2', 0.0) or self._quality.get('area', 0.0) or 0.0)

        # ── 评估标准摘要栏 ──
        std_banner = QLabel(
            f"评估标准：{standard_text}　|　"
            f"平整度阈值：≤ {flat_limit:.1f} mm　|　"
            f"垂直度阈值：≤ {vert_limit:.1f} mm　|　"
            f"立面面积：{facade_area:.2f} m²" if facade_area > 0 else
            f"评估标准：{standard_text}　|　"
            f"平整度阈值：≤ {flat_limit:.1f} mm　|　"
            f"垂直度阈值：≤ {vert_limit:.1f} mm"
        )
        std_banner.setObjectName('summaryBanner')
        std_banner.setWordWrap(True)
        layout.addWidget(std_banner)

        def _get_rate(method_dict, metric, fallback=0.0):
            data = method_dict.get(metric, {}) if method_dict else {}
            rates = data.get('rates', {}) if isinstance(data, dict) else {}
            for key in ('area_rate', 'primary_area_rate'):
                val = rates.get(key)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
            return float(fallback)

        ruler = methods.get('ruler', {})
        global_plane = methods.get('global_plane', {})

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 靠尺法检测结果 + 区间表格
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ruler_group = QGroupBox("靠尺法（米字/I字靠尺）检测结果")
        ruler_group.setStyleSheet("""
            QGroupBox {
                font-weight: 700;
                font-size: 13px;
                color: #1e293b;
                border: 1px solid #3b82f6;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px;
                color: #2563eb;
            }
        """)
        ruler_layout = QVBoxLayout(ruler_group)
        ruler_layout.setContentsMargins(12, 8, 12, 8)
        ruler_layout.setSpacing(8)

        # 合格率 headline：仅展示各自指标，禁止合并成不具物理意义的平均率。
        ruler_flat_area = _get_rate(ruler, 'flatness') * 100.0
        ruler_vert_area = _get_rate(ruler, 'verticality') * 100.0

        ruler_head = QHBoxLayout()
        ruler_head.setSpacing(24)
        for lbl_text, rate, color in (
            ("平整度面积合格率", ruler_flat_area, "#1e293b"),
            ("垂直度面积合格率", ruler_vert_area, "#1e293b"),
        ):
            lbl = QLabel(lbl_text)
            lbl.setStyleSheet('color: #475569; font-size: 12px;')
            val = QLabel(f"{rate:.1f}%")
            val.setStyleSheet(f'color: {color}; font-size: 14px; font-weight: 700;')
            ruler_head.addWidget(lbl)
            ruler_head.addWidget(val)
        ruler_head.addStretch(1)
        ruler_layout.addLayout(ruler_head)

        # 区间表格
        self._ruler_table = self._build_interval_table()
        self._fill_interval_table(self._ruler_table, 'ruler')
        ruler_layout.addWidget(self._ruler_table, 1)
        layout.addWidget(ruler_group, 2)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 全局平面基准法检测结果 + 区间表格
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if global_plane:
            global_group = QGroupBox("全局平面基准法检测结果")
            global_group.setStyleSheet("""
                QGroupBox {
                    font-weight: 700;
                    font-size: 13px;
                    color: #1e293b;
                    border: 1px solid #10b981;
                    border-radius: 8px;
                    margin-top: 10px;
                    padding-top: 6px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 8px;
                    color: #059669;
                }
            """)
            global_layout = QVBoxLayout(global_group)
            global_layout.setContentsMargins(12, 8, 12, 8)
            global_layout.setSpacing(8)

            global_flat_area = _get_rate(global_plane, 'flatness') * 100.0
            global_vert_area = _get_rate(global_plane, 'verticality') * 100.0

            global_head = QHBoxLayout()
            global_head.setSpacing(24)
            for lbl_text, rate, color in (
                ("平整度面积合格率", global_flat_area, "#1e293b"),
                ("垂直度面积合格率", global_vert_area, "#1e293b"),
            ):
                lbl = QLabel(lbl_text)
                lbl.setStyleSheet('color: #475569; font-size: 12px;')
                val = QLabel(f"{rate:.1f}%")
                val.setStyleSheet(f'color: {color}; font-size: 14px; font-weight: 700;')
                global_head.addWidget(lbl)
                global_head.addWidget(val)
            global_head.addStretch(1)
            global_layout.addLayout(global_head)

            self._global_table = self._build_interval_table()
            self._fill_interval_table(self._global_table, 'global_plane')
            global_layout.addWidget(self._global_table, 1)
            layout.addWidget(global_group, 2)

        # ── 底部按钮栏 ──
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch(1)

        self._mode_combo = QComboBox(self)
        self._mode_combo.addItem('靠尺平整度（面积）', 'ruler_flatness_area')
        self._mode_combo.addItem('靠尺垂直度（面积）', 'ruler_verticality_area')
        self._mode_combo.addItem('全局平面平整度（面积）', 'global_plane_flatness_area')
        self._mode_combo.addItem('全局平面垂直度（面积）', 'global_plane_verticality_area')
        self._mode_combo.setMinimumWidth(180)

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

        n_quality = int(overall.get('quality_valid_window_count', 0) or 0)
        has_valid_windows = bool(ok and n_quality > 0)

        if callable(self._on_show_colors) and has_valid_windows:
            btn_show.clicked.connect(
                lambda: self._on_show_colors(self._mode_combo.currentData() or 'flatness'))
        else:
            btn_show.setEnabled(False)
            if not ok:
                btn_show.setToolTip('质量计算未成功完成')
            elif n_quality == 0:
                btn_show.setToolTip('没有有效检测窗口')
            else:
                btn_show.setToolTip('无可用的颜色渲染回调')

        if callable(self._on_restore_colors):
            btn_restore.clicked.connect(self._on_restore_colors)
        else:
            btn_restore.setEnabled(False)

    # ------------------------------------------------------------------
    # 区间表格构建与填充
    # ------------------------------------------------------------------
    def _build_interval_table(self) -> QTableWidget:
        table = QTableWidget(0, 8, self)
        table.setObjectName('tblQualityGrids')
        table.setHorizontalHeaderLabels([
            "区间高度", "点数", "窗口数", "合格数",
            "平整度最大间隙(mm)", "平整度面积合格率(%)",
            "垂直度最大偏差(mm)", "垂直度面积合格率(%)"
        ])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        table.horizontalHeader().setStretchLastSection(False)
        table.horizontalHeader().setDefaultSectionSize(90)
        table.horizontalHeader().setMinimumSectionSize(70)
        table.setColumnWidth(0, 110)
        table.setColumnWidth(1, 60)
        table.setColumnWidth(2, 60)
        table.setColumnWidth(3, 60)
        table.setColumnWidth(4, 130)
        table.setColumnWidth(5, 130)
        table.setColumnWidth(6, 130)
        table.setColumnWidth(7, 130)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        return table

    def _fill_interval_table(self, table: QTableWidget, method: str):
        comparison = self._quality.get('quality_comparison') or {}
        methods = comparison.get('methods') or {}
        method_data = methods.get(method) or {}
        # 新结果可直接提供方法专属区间；旧结果没有时才从该方法窗口
        # 生成简化区间，绝不再把公共 ruler intervals 同时绑定到两张表。
        intervals = method_data.get('intervals') or []
        if not intervals:
            intervals = self._quality.get('intervals') or [] if method == 'ruler' else []
        table.setRowCount(len(intervals))

        def number(value, suffix=''):
            try:
                if value is None:
                    return '--'
                value = float(value)
                return f'{value:.2f}{suffix}' if math.isfinite(value) else '--'
            except (TypeError, ValueError):
                return '--'

        for r, item in enumerate(intervals):
            # 优先读取服务层统一生成的物理面积率；旧结果才回退到窗口率。
            flat_rate = item.get('flatness_area_rate')
            if flat_rate is None:
                flat_rate = item.get('flatness_pass_rate')
            vert_rate = item.get('verticality_area_rate')
            if vert_rate is None:
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
                number((float(flat_rate) * 100) if flat_rate is not None else None),
                number(item.get('verticality_max_deviation_mm')),
                number((float(vert_rate) * 100) if vert_rate is not None else None),
            ]
            for c, value in enumerate(vals):
                cell = QTableWidgetItem(value)
                cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(r, c, cell)

        table.resizeColumnsToContents()
        for c in range(table.columnCount()):
            w = table.columnWidth(c)
            table.setColumnWidth(c, min(w, 160))