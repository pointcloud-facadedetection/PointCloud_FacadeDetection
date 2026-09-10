from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QDate, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QPushButton,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from services.inspection_profile import InspectionProfile, InspectionProfileService


class RegionSelector(QWidget):
    """
    省市区三级联动选择控件，从 utils/pca-code.json 加载离线数据。
    信号：selectionChanged(str, str, str) 当任意一级变化时发出（省、市、区名称）
    """
    selectionChanged = Signal(str, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = self._load_data()
        self.cb_province = QComboBox()
        self.cb_city = QComboBox()
        self.cb_district = QComboBox()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.cb_province)
        layout.addWidget(self.cb_city)
        layout.addWidget(self.cb_district)

        self.cb_province.addItem("请选择省份")
        for prov in self.data:
            self.cb_province.addItem(prov["name"])

        self.cb_province.currentIndexChanged.connect(self._on_province_changed)
        self.cb_city.currentIndexChanged.connect(self._on_city_changed)

        self._on_province_changed(0)

    def _load_data(self) -> list:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        package_dir = os.path.dirname(os.path.dirname(current_dir))
        json_path = os.path.join(package_dir, "utils", "pca-code.json")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            QMessageBox.critical(self, "错误", "未找到省市区数据")
            return []
        except json.JSONDecodeError:
            QMessageBox.critical(self, "错误", "省市区数据文件格式错误")
            return []

    def _on_province_changed(self, idx: int):
        self.cb_city.clear()
        self.cb_city.addItem("请选择城市")
        self.cb_district.clear()
        self.cb_district.addItem("请选择区县")

        if idx > 0 and self.data:
            prov = self.data[idx - 1]
            cities = prov.get("children", [])
            for city in cities:
                self.cb_city.addItem(city["name"])

        self._emit_selection()

    def _on_city_changed(self, idx: int):
        self.cb_district.clear()
        self.cb_district.addItem("请选择区县")

        prov_idx = self.cb_province.currentIndex() - 1
        if prov_idx >= 0 and idx > 0 and self.data:
            prov = self.data[prov_idx]
            cities = prov.get("children", [])
            if idx - 1 < len(cities):
                city = cities[idx - 1]
                districts = city.get("children", [])
                for dist in districts:
                    self.cb_district.addItem(dist["name"])

        self._emit_selection()

    def _emit_selection(self):
        prov = self.cb_province.currentText()
        city = self.cb_city.currentText()
        dist = self.cb_district.currentText()
        if prov == "请选择省份":
            prov = ""
        if city == "请选择城市":
            city = ""
        if dist == "请选择区县":
            dist = ""
        self.selectionChanged.emit(prov, city, dist)

    def get_selected(self) -> tuple[str, str, str]:
        prov = self.cb_province.currentText()
        city = self.cb_city.currentText()
        dist = self.cb_district.currentText()
        if prov == "请选择省份":
            prov = ""
        if city == "请选择城市":
            city = ""
        if dist == "请选择区县":
            dist = ""
        return prov, city, dist


class ProjectCreateDialog(QDialog):
    """
    创建/编辑项目四页签表单：
      - 基础信息
      - 报告信息
      - 检测参数
      - 文件导入
    """

    def __init__(self, parent=None, project=None):
        super().__init__(parent)
        self._project = project
        self._report_no_auto = project is None
        self.setWindowTitle("编辑项目" if project is not None else "创建项目")
        self.setModal(True)
        self._build_ui()
        if project is not None:
            self._load_project(project)

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------
    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setSpacing(12)

        title = QLabel("请输入项目信息")
        title.setStyleSheet("font-size:16px; font-weight:600; color:#333;")
        lay.addWidget(title)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_basic_tab(), "基础信息")
        self.tabs.addTab(self._build_report_tab(), "报告信息")
        self.tabs.addTab(self._build_inspection_tab(), "检测参数")
        self.tabs.addTab(self._build_import_tab(), "文件导入")
        lay.addWidget(self.tabs)

        self.btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self.btns.accepted.connect(self._on_accept)
        self.btns.rejected.connect(self.reject)
        lay.addWidget(self.btns)

        self.resize(600, 580)
        self.setMinimumSize(560, 480)

    # ---------- 基础信息 ----------
    def _build_basic_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setSpacing(10)

        self.edt_name = QLineEdit()
        self.edt_name.setPlaceholderText("必填")
        self.edt_name.textChanged.connect(self._update_report_no)
        form.addRow("项目名称：", self.edt_name)

        self.edt_org = QLineEdit()
        self.edt_org.setPlaceholderText("选填")
        form.addRow("所属单位：", self.edt_org)

        self.region_selector = RegionSelector()
        form.addRow("省市区：", self.region_selector)

        self.edt_address = QLineEdit()
        self.edt_address.setPlaceholderText("具体地址，如街道")
        form.addRow("详细地址：", self.edt_address)

        self.edt_building = QLineEdit()
        self.edt_building.setPlaceholderText("如：1号楼")
        form.addRow("楼栋号信息：", self.edt_building)

        self.edt_remarks = QTextEdit()
        self.edt_remarks.setPlaceholderText("选填")
        self.edt_remarks.setFixedHeight(80)
        form.addRow("备注：", self.edt_remarks)

        return page

    # ---------- 文件导入 ----------
    def _build_import_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)
        self._resource_lists = {}
        sections = (
            ('fls_directories', 'FLS 目录导入', '上传单站点原生 FLS 文件夹', 'directory'),
            ('pointcloud_files', '点云文件上传', '上传多站点拼接处理后点云文件 (E57/PLY)', 'pointcloud'),
            ('photo_files', '现场照片上传', '上传对应站点拍摄 2D 照片', 'photo'),
        )
        for key, title, hint, kind in sections:
            box = QGroupBox(title)
            layout = QVBoxLayout(box)
            hint_label = QLabel(hint)
            hint_label.setStyleSheet('color:#64748b;font-size:11px;')
            layout.addWidget(hint_label)
            listing = QListWidget()
            listing.setObjectName(f'{key}List')
            listing.setMinimumHeight(70)
            self._resource_lists[key] = listing
            layout.addWidget(listing, 1)
            actions = QHBoxLayout()
            actions.addStretch(1)
            add = QPushButton('新增')
            clear = QPushButton('清空')
            add.clicked.connect(lambda _=False, k=key, t=title, ty=kind: self._add_resource(k, t, ty))
            clear.clicked.connect(lambda _=False, k=key, t=title: self._clear_resources(k, t))
            actions.addWidget(add)
            actions.addWidget(clear)
            layout.addLayout(actions)
            outer.addWidget(box, 1)
        outer.addStretch(1)
        return page

    def _add_resource(self, key: str, title: str, kind: str):
        if kind == 'directory':
            selected = QFileDialog.getExistingDirectory(self, title)
            paths = [selected] if selected else []
        else:
            if kind == 'pointcloud':
                flt = '点云文件 (*.e57 *.ply *.pcd *.xyz *.xyzn *.xyzrgb *.pts);;所有文件 (*)'
            else:
                flt = '照片文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;所有文件 (*)'
            paths, _ = QFileDialog.getOpenFileNames(self, title, '', flt)
        for path in paths:
            candidate = str(Path(path).expanduser().resolve())
            if not Path(candidate).exists():
                QMessageBox.warning(self, title, f'路径不存在：\n{candidate}')
                continue
            existing = self._resource_values(key)
            if candidate in existing:
                continue
            self._append_resource_row(key, candidate)

    def _append_resource_row(self, key: str, path: str):
        listing = self._resource_lists[key]
        item = QListWidgetItem(listing)
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(6, 3, 4, 3)
        label = QLabel(path)
        label.setToolTip(path)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(label, 1)
        remove = QPushButton('删除')
        remove.setProperty('buttonRole', 'danger')
        remove.clicked.connect(
            lambda _=False, i=item, l=listing, title=key:
            self._remove_resource(l, i, title))
        layout.addWidget(remove)
        item.setSizeHint(row.sizeHint())
        listing.addItem(item)
        listing.setItemWidget(item, row)

    def _remove_resource(self, listing: QListWidget, item: QListWidgetItem,
                         title: str):
        if QMessageBox.question(
                self, '删除资源', f'确定删除该{title}资源条目吗？',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            listing.takeItem(listing.row(item))

    def _resource_values(self, key: str) -> list[str]:
        listing = self._resource_lists[key]
        values = []
        for index in range(listing.count()):
            row = listing.itemWidget(listing.item(index))
            if row is not None:
                label = row.findChild(QLabel)
                if label is not None:
                    values.append(label.text())
        return values

    def _clear_resources(self, key: str, title: str):
        if not self._resource_values(key):
            return
        if QMessageBox.question(self, title, '确定清空本板块全部资源吗？',
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            self._resource_lists[key].clear()

    def _load_resources(self, fls, pointclouds, photos):
        for key, values in (('fls_directories', fls), ('pointcloud_files', pointclouds), ('photo_files', photos)):
            for path in dict.fromkeys(str(value) for value in (values or []) if value):
                self._append_resource_row(key, path)

    # ---------- 报告信息 ----------
    def _build_report_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setSpacing(10)

        self.edt_construction_unit = QLineEdit()
        form.addRow("建设单位：", self.edt_construction_unit)

        self.edt_construction_unit_executor = QLineEdit()
        form.addRow("施工单位：", self.edt_construction_unit_executor)

        self.edt_inspection_unit = QLineEdit()
        form.addRow("检测单位：", self.edt_inspection_unit)

        self.edt_supervision_unit = QLineEdit()
        form.addRow("监理单位：", self.edt_supervision_unit)

        self.edt_client_unit = QLineEdit()
        form.addRow("委托单位：", self.edt_client_unit)

        self.edt_report_no = QLineEdit()
        self.edt_report_no.setPlaceholderText("RJ-项目名称拼音首字母，可人工修改")
        self.edt_report_no.textEdited.connect(lambda _text: setattr(self, '_report_no_auto', False))
        form.addRow("报告编号：", self.edt_report_no)

        self.date_inspection = QDateEdit()
        self.date_inspection.setCalendarPopup(True)
        self.date_inspection.setDisplayFormat("yyyy-MM-dd")
        self.date_inspection.setDate(QDateEdit().date().currentDate())
        form.addRow("检测日期：", self.date_inspection)

        self.date_report = QDateEdit()
        self.date_report.setCalendarPopup(True)
        self.date_report.setDisplayFormat("yyyy-MM-dd")
        self.date_report.setDate(QDateEdit().date().currentDate())
        form.addRow("报告日期：", self.date_report)

        return page

    # ---------- 检测参数 ----------
    def _build_inspection_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(10)
        layout.setContentsMargins(4, 4, 4, 4)

        # --- 标准选择 ---
        std_row = QHBoxLayout()
        std_row.addWidget(QLabel("检测标准："))
        self.cb_standard = QComboBox()
        for profile in InspectionProfileService.all():
            self.cb_standard.addItem(
                f"{profile.standard_name} · {profile.version}", profile.standard_id
            )
        self.cb_standard.currentIndexChanged.connect(self._on_standard_changed)
        std_row.addWidget(self.cb_standard, 1)
        layout.addLayout(std_row)

        self.lbl_standard_hint = QLabel()
        self.lbl_standard_hint.setStyleSheet("color:#64748b; font-size:11px; padding-left:4px;")
        layout.addWidget(self.lbl_standard_hint)

        # --- 主要参数表单（两列紧凑布局）---
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        def _add_row(row: int, label: str, widget: QWidget):
            lbl = QLabel(label)
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(lbl, row, 0)
            grid.addWidget(widget, row, 1)
            return row + 1

        r = 0
        self.cb_interval = QComboBox()
        for value in (3.0, 5.0, 10.0, 20.0):
            self.cb_interval.addItem(f"{value:g} m", value)
        self.cb_interval.setCurrentIndex(3)
        r = _add_row(r, "区间尺度：", self.cb_interval)

        self.cb_ruler_length = QComboBox()
        for value in (1.0, 2.0, 3.0):
            self.cb_ruler_length.addItem(f"{value:g} m", value)
        self.cb_ruler_length.setCurrentIndex(1)
        self.cb_ruler_length.currentIndexChanged.connect(self._on_ruler_length_changed)
        r = _add_row(r, "靠尺长度：", self.cb_ruler_length)

        self.spin_ruler_width = QDoubleSpinBox()
        self.spin_ruler_width.setRange(0.01, 0.5)
        self.spin_ruler_width.setDecimals(3)
        self.spin_ruler_width.setValue(0.055)
        self.spin_ruler_width.setSuffix(" m")
        r = _add_row(r, "靠尺宽度：", self.spin_ruler_width)

        self.spin_step_longitudinal = QDoubleSpinBox()
        self.spin_step_longitudinal.setRange(0.001, 10.0)
        self.spin_step_longitudinal.setDecimals(3)
        self.spin_step_longitudinal.setValue(2.0)
        self.spin_step_longitudinal.setSuffix(" m")
        r = _add_row(r, "纵向采样步长：", self.spin_step_longitudinal)

        self.spin_step_transverse = QDoubleSpinBox()
        self.spin_step_transverse.setRange(0.001, 1.0)
        self.spin_step_transverse.setDecimals(3)
        self.spin_step_transverse.setValue(0.055)
        self.spin_step_transverse.setSuffix(" m")
        r = _add_row(r, "横向采样步长：", self.spin_step_transverse)

        self.spin_flatness_limit = QDoubleSpinBox()
        self.spin_flatness_limit.setRange(0.1, 50.0)
        self.spin_flatness_limit.setDecimals(1)
        self.spin_flatness_limit.setValue(4.0)
        self.spin_flatness_limit.setSuffix(" mm")
        r = _add_row(r, "平整度阈值：", self.spin_flatness_limit)

        self.spin_verticality_limit = QDoubleSpinBox()
        self.spin_verticality_limit.setRange(0.1, 50.0)
        self.spin_verticality_limit.setDecimals(1)
        self.spin_verticality_limit.setValue(4.0)
        self.spin_verticality_limit.setSuffix(" mm")
        r = _add_row(r, "垂直度阈值：", self.spin_verticality_limit)

        self.spin_select_band = QDoubleSpinBox()
        self.spin_select_band.setRange(0.001, 0.5)
        self.spin_select_band.setDecimals(3)
        self.spin_select_band.setValue(0.01)
        self.spin_select_band.setSuffix(" m")
        r = _add_row(r, "表面带宽：", self.spin_select_band)

        self.spin_hole_band = QDoubleSpinBox()
        self.spin_hole_band.setRange(0.001, 0.5)
        self.spin_hole_band.setDecimals(3)
        self.spin_hole_band.setValue(0.02)
        self.spin_hole_band.setSuffix(" m")
        r = _add_row(r, "空洞带宽：", self.spin_hole_band)

        layout.addLayout(grid)

        # --- SOR 参数 ---
        sor_frame = QGroupBox("SOR 离群剔除")
        sor_layout = QGridLayout(sor_frame)
        sor_layout.setHorizontalSpacing(12)
        sor_layout.setVerticalSpacing(6)

        self.chk_sor = QCheckBox("启用")
        self.chk_sor.setChecked(True)
        sor_layout.addWidget(self.chk_sor, 0, 0)

        self.spin_sor_sigma = QDoubleSpinBox()
        self.spin_sor_sigma.setRange(0.1, 20.0)
        self.spin_sor_sigma.setDecimals(1)
        self.spin_sor_sigma.setValue(4.0)
        sor_layout.addWidget(QLabel("阈值 σ"), 0, 1)
        sor_layout.addWidget(self.spin_sor_sigma, 0, 2)

        self.spin_sor_k = QSpinBox()
        self.spin_sor_k.setRange(1, 128)
        self.spin_sor_k.setValue(8)
        sor_layout.addWidget(QLabel("邻居数 k"), 1, 1)
        sor_layout.addWidget(self.spin_sor_k, 1, 2)

        self.cb_sor_method = QComboBox()
        self.cb_sor_method.addItem("local", "local")
        self.cb_sor_method.addItem("grid", "grid")
        self.cb_sor_method.addItem("exact", "exact")
        sor_layout.addWidget(QLabel("方法"), 2, 1)
        sor_layout.addWidget(self.cb_sor_method, 2, 2)

        self.spin_sor_w_weight = QDoubleSpinBox()
        self.spin_sor_w_weight.setRange(0.0, 200.0)
        self.spin_sor_w_weight.setDecimals(1)
        self.spin_sor_w_weight.setValue(50.0)
        sor_layout.addWidget(QLabel("高度权重"), 3, 1)
        sor_layout.addWidget(self.spin_sor_w_weight, 3, 2)

        layout.addWidget(sor_frame)
        layout.addStretch(1)

        scroll.setWidget(content)
        outer.addWidget(scroll)

        # 初始化联动
        self._on_standard_changed(0)
        return page

    # ------------------------------------------------------------------
    # 联动逻辑
    # ------------------------------------------------------------------
    def _on_standard_changed(self, _index: int):
        profile = InspectionProfileService.get(self.cb_standard.currentData())
        if profile is None:
            return
        self.spin_flatness_limit.blockSignals(True)
        self.spin_verticality_limit.blockSignals(True)
        self.spin_flatness_limit.setValue(profile.flatness_limit_mm)
        self.spin_verticality_limit.setValue(profile.verticality_limit_mm)
        self.spin_flatness_limit.blockSignals(False)
        self.spin_verticality_limit.blockSignals(False)

        self.lbl_standard_hint.setText(
            f"平整度 ≤ {profile.flatness_limit_mm:g} mm  |  垂直度 ≤ {profile.verticality_limit_mm:g} mm"
        )

    def _on_ruler_length_changed(self, _index: int):
        length = self.cb_ruler_length.currentData()
        if length is not None:
            self.spin_step_longitudinal.blockSignals(True)
            self.spin_step_longitudinal.setValue(float(length))
            self.spin_step_longitudinal.blockSignals(False)

    def _update_report_no(self, project_name: str):
        """按项目名称实时生成报告编号；人工编辑后停止覆盖。"""
        if not getattr(self, '_report_no_auto', True):
            return
        name = (project_name or '').strip()
        try:
            from pypinyin import Style, lazy_pinyin
            initials = ''.join(
                item[:1] for item in lazy_pinyin(name, style=Style.FIRST_LETTER)
                if item
            ).upper()
        except Exception:
            initials = ''.join(char for char in name if char.isascii() and char.isalnum()).upper()
        self.edt_report_no.setText(f'RJ-{initials}' if initials else '')

    # ------------------------------------------------------------------
    # 数据回填
    # ------------------------------------------------------------------
    def _load_project(self, project):
        self.edt_name.setText(str(getattr(project, "name", "") or ""))
        self.edt_org.setText(str(getattr(project, "org_unit", "") or ""))
        self.edt_address.setText(str(getattr(project, "address", "") or ""))
        self.edt_building.setText(str(getattr(project, "building_floor", "") or ""))
        self.edt_remarks.setPlainText(str(getattr(project, "remarks", "") or ""))

        # address 在旧模型中保存为“省市区 详细地址”，仅用于回显选择器；
        # 无法拆分的历史地址仍完整保留在详细地址输入框中。
        self._restore_region(str(getattr(project, "address", "") or ""))

        self.edt_construction_unit.setText(str(getattr(project, "construction_unit", "") or ""))
        self.edt_construction_unit_executor.setText(str(getattr(project, "construction_unit_executor", "") or ""))
        self.edt_inspection_unit.setText(str(getattr(project, "inspection_unit", "") or ""))
        self.edt_supervision_unit.setText(str(getattr(project, "supervision_unit", "") or ""))
        self.edt_client_unit.setText(str(getattr(project, "client_unit", "") or ""))
        self.edt_report_no.setText(str(getattr(project, "report_no", "") or ""))
        self._report_no_auto = not bool(self.edt_report_no.text().strip())
        self._update_report_no(self.edt_name.text())

        d = getattr(project, "inspection_date", None)
        if d:
            self.date_inspection.setDate(
                QDate(d.year, d.month, d.day) if hasattr(d, 'year') else d)
        d = getattr(project, "report_date", None)
        if d:
            self.date_report.setDate(
                QDate(d.year, d.month, d.day) if hasattr(d, 'year') else d)

        params = getattr(project, "inspection_params_json", None)
        if params:
            try:
                p = json.loads(params) if isinstance(params, str) else params
                self._load_inspection_params(p)
            except Exception:
                pass

        self._load_resources(getattr(project, 'fls_directories', []),
                             getattr(project, 'pointcloud_files', []),
                             getattr(project, 'photo_files', []))

    def _restore_region(self, address: str):
        if not address or not self.region_selector.data:
            return
        for province_index, province in enumerate(self.region_selector.data, 1):
            pname = province.get('name', '')
            if not address.startswith(pname):
                continue
            self.region_selector.cb_province.setCurrentIndex(province_index)
            remainder = address[len(pname):].lstrip()
            for city_index, city in enumerate(province.get('children', []), 1):
                cname = city.get('name', '')
                if not remainder.startswith(cname):
                    continue
                self.region_selector.cb_city.setCurrentIndex(city_index)
                remainder = remainder[len(cname):].lstrip()
                for district_index, district in enumerate(city.get('children', []), 1):
                    dname = district.get('name', '')
                    if remainder.startswith(dname):
                        self.region_selector.cb_district.setCurrentIndex(district_index)
                        remainder = remainder[len(dname):].lstrip()
                        break
                self.edt_address.setText(remainder)
                return
            self.edt_address.setText(remainder)
            return

    def _load_inspection_params(self, p: dict):
        std_id = p.get("standard_id")
        if std_id:
            idx = self.cb_standard.findData(std_id)
            if idx >= 0:
                self.cb_standard.setCurrentIndex(idx)

        interval = p.get("interval_size_m")
        if interval is not None:
            idx = self.cb_interval.findData(float(interval))
            if idx >= 0:
                self.cb_interval.setCurrentIndex(idx)

        length = p.get("ruler_length_m")
        if length is not None:
            idx = self.cb_ruler_length.findData(float(length))
            if idx >= 0:
                self.cb_ruler_length.setCurrentIndex(idx)

        self.spin_ruler_width.setValue(float(p.get("ruler_width_m", 0.055)))
        self.spin_step_longitudinal.setValue(float(p.get("step_longitudinal_m", 2.0)))
        self.spin_step_transverse.setValue(float(p.get("step_transverse_m", 0.055)))
        self.spin_flatness_limit.setValue(float(p.get("flatness_limit_mm", 4.0)))
        self.spin_verticality_limit.setValue(float(p.get("verticality_limit_mm", 4.0)))
        self.spin_select_band.setValue(float(p.get("select_band_m", 0.01)))
        self.spin_hole_band.setValue(float(p.get("hole_band_m", 0.02)))

        self.chk_sor.setChecked(bool(p.get("sor_enabled", True)))
        self.spin_sor_sigma.setValue(float(p.get("sor_sigma", 4.0)))
        self.spin_sor_k.setValue(int(p.get("sor_k", 8)))
        method = p.get("sor_method", "local")
        idx = self.cb_sor_method.findData(method)
        if idx >= 0:
            self.cb_sor_method.setCurrentIndex(idx)
        self.spin_sor_w_weight.setValue(float(p.get("sor_w_weight", 50.0)))

    # ------------------------------------------------------------------
    # 校验与取值
    # ------------------------------------------------------------------
    def _on_accept(self):
        if not self.edt_name.text().strip():
            QMessageBox.warning(self, "提示", "项目名称为必填项！")
            self.tabs.setCurrentIndex(0)
            self.edt_name.setFocus()
            return
        self.accept()

    def values(self) -> dict:
        prov, city, dist = self.region_selector.get_selected()
        region_parts = [p for p in (prov, city, dist) if p]
        region_str = "".join(region_parts)
        detail = self.edt_address.text().strip()
        full_address = region_str
        if detail:
            full_address = (region_str + " " + detail) if region_str else detail
        full_address = full_address.strip() or None

        inspection_params = {
            "standard_id": self.cb_standard.currentData(),
            "interval_size_m": self.cb_interval.currentData(),
            "ruler_length_m": self.cb_ruler_length.currentData(),
            "ruler_width_m": self.spin_ruler_width.value(),
            "step_longitudinal_m": self.spin_step_longitudinal.value(),
            "step_transverse_m": self.spin_step_transverse.value(),
            "flatness_limit_mm": self.spin_flatness_limit.value(),
            "verticality_limit_mm": self.spin_verticality_limit.value(),
            "select_band_m": self.spin_select_band.value(),
            "hole_band_m": self.spin_hole_band.value(),
            "sor_enabled": self.chk_sor.isChecked(),
            "sor_sigma": self.spin_sor_sigma.value(),
            "sor_k": self.spin_sor_k.value(),
            "sor_method": self.cb_sor_method.currentData(),
            "sor_w_weight": self.spin_sor_w_weight.value(),
        }

        return {
            "name": self.edt_name.text().strip(),
            "org_unit": self.edt_org.text().strip() or None,
            "address": full_address,
            "building_floor": self.edt_building.text().strip() or None,
            "remarks": self.edt_remarks.toPlainText().strip() or None,
            "construction_unit": self.edt_construction_unit.text().strip() or None,
            "construction_unit_executor": self.edt_construction_unit_executor.text().strip() or None,
            "inspection_unit": self.edt_inspection_unit.text().strip() or None,
            "supervision_unit": self.edt_supervision_unit.text().strip() or None,
            "client_unit": self.edt_client_unit.text().strip() or None,
            "report_no": self.edt_report_no.text().strip() or None,
            "inspection_date": self.date_inspection.date().toPython() if self.date_inspection.date() else None,
            "report_date": self.date_report.date().toPython() if self.date_report.date() else None,
            "inspection_params_json": json.dumps(inspection_params, ensure_ascii=False),
            "fls_directories": self._resource_values('fls_directories'),
            "pointcloud_files": self._resource_values('pointcloud_files'),
            "photo_files": self._resource_values('photo_files'),
        }