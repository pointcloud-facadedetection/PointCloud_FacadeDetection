from __future__ import annotations

import json
import os
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QTextEdit,
    QVBoxLayout,
    QLabel,
    QMessageBox,
    QComboBox,
    QFrame,
    QWidget,
    QHBoxLayout,
)


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
        # 三级区域选择沿用全局下拉按钮，避免出现 Windows 原生减号式箭头。
        for combo in (self.cb_province, self.cb_city, self.cb_district):
            combo.setProperty('uiRole', 'projectRegionInput')
            combo.setMinimumHeight(36)

        # 布局：水平排列三个下拉框
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.cb_province)
        layout.addWidget(self.cb_city)
        layout.addWidget(self.cb_district)

        # 填充省份
        self.cb_province.addItem("请选择省份")
        for prov in self.data:
            self.cb_province.addItem(prov["name"])

        # 信号连接
        self.cb_province.currentIndexChanged.connect(self._on_province_changed)
        self.cb_city.currentIndexChanged.connect(self._on_city_changed)

        # 初始化城市和区县
        self._on_province_changed(0)

    def _load_data(self) -> list:
        """从 utils/pca-code.json 加载数据"""
        # 当前文件位于 facadeDetection/ui/dialogs，向上两级到达包根目录。
        current_dir = os.path.dirname(os.path.abspath(__file__))
        package_dir = os.path.dirname(os.path.dirname(current_dir))
        json_path = os.path.join(package_dir, "utils", "pca-code.json")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            # 如果文件不存在，返回空列表，并显示错误提示
            QMessageBox.critical(self, "错误", "未找到省市区数据")
            return []
        except json.JSONDecodeError:
            QMessageBox.critical(self, "错误", "省市区数据文件格式错误")
            return []

    def _on_province_changed(self, idx: int):
        """省份变化 -> 更新城市列表"""
        self.cb_city.clear()
        self.cb_city.addItem("请选择城市")
        self.cb_district.clear()
        self.cb_district.addItem("请选择区县")

        if idx > 0 and self.data:
            prov = self.data[idx - 1]
            cities = prov.get("children", [])
            for city in cities:
                self.cb_city.addItem(city["name"])

        # 触发信号
        self._emit_selection()

    def _on_city_changed(self, idx: int):
        """城市变化 -> 更新区县列表"""
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
        """发射当前选中的省市区名称（未选时返回空字符串）"""
        prov = self.cb_province.currentText()
        city = self.cb_city.currentText()
        dist = self.cb_district.currentText()
        # 如果为占位文本，视为未选择
        if prov == "请选择省份":
            prov = ""
        if city == "请选择城市":
            city = ""
        if dist == "请选择区县":
            dist = ""
        self.selectionChanged.emit(prov, city, dist)

    def get_selected(self) -> tuple[str, str, str]:
        """返回 (省, 市, 区) 三元组，未选则为空字符串"""
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
    创建项目表单对话框：
    - 必填：项目名称
    - 选填：单位、省市区（三级联动）、详细地址、备注
    """

    def __init__(self, parent=None, project=None):
        super().__init__(parent)
        self._project = project
        self.setObjectName('projectCreateDialog')
        self.setWindowTitle("编辑项目" if project is not None else "创建项目")
        self.setModal(True)
        self._build_ui()
        if project is not None:
            self._load_project(project)

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 20)
        lay.setSpacing(16)

        # 与主工作台相同的深蓝标题层，明确区分标题、说明与表单内容。
        hero = QFrame()
        hero.setObjectName('projectDialogHero')
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(18, 14, 18, 14)
        hero_layout.setSpacing(4)
        title_text = '编辑项目信息' if self._project is not None else '创建新项目'
        title = QLabel(title_text)
        title.setObjectName('projectDialogTitle')
        subtitle = QLabel('填写工程识别信息，后续点云与检测结果将归档到该项目。')
        subtitle.setObjectName('projectDialogSubtitle')
        subtitle.setWordWrap(True)
        hero_layout.addWidget(title)
        hero_layout.addWidget(subtitle)
        lay.addWidget(hero)

        form_panel = QFrame()
        form_panel.setObjectName('projectFormPanel')
        form = QFormLayout(form_panel)
        form.setContentsMargins(20, 18, 20, 18)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(13)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        # 项目名称（必填）
        self.edt_name = QLineEdit()
        self.edt_name.setObjectName('projectNameInput')
        self.edt_name.setPlaceholderText("请输入项目名称")
        form.addRow("项目名称 *", self.edt_name)

        # 单位（选填）
        self.edt_org = QLineEdit()
        self.edt_org.setPlaceholderText("请输入所属单位（选填）")
        form.addRow("所属单位", self.edt_org)

        # 省市区三级联动（选填）
        self.region_selector = RegionSelector()
        self.region_selector.setObjectName('projectRegionSelector')
        form.addRow("省市区", self.region_selector)

        # 详细地址（选填）
        self.edt_address = QLineEdit()
        self.edt_address.setPlaceholderText("具体地址，如街道")
        form.addRow("详细地址", self.edt_address)

        # 楼栋号（选填）
        self.edt_building = QLineEdit()
        self.edt_building.setPlaceholderText('如：1号楼')
        form.addRow('楼栋号信息', self.edt_building)

        # 备注（选填）
        self.edt_remarks = QTextEdit()
        self.edt_remarks.setPlaceholderText("补充项目说明（选填）")
        self.edt_remarks.setFixedHeight(92)
        form.addRow("备注", self.edt_remarks)

        lay.addWidget(form_panel, 1)

        # 按钮
        self.btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self.btns.setObjectName('projectDialogButtons')
        save_button = self.btns.button(QDialogButtonBox.StandardButton.Save)
        cancel_button = self.btns.button(QDialogButtonBox.StandardButton.Cancel)
        save_button.setText('保存项目')
        save_button.setProperty('buttonRole', 'primary')
        save_button.setMinimumSize(104, 38)
        cancel_button.setText('取消')
        cancel_button.setProperty('buttonRole', 'tonal')
        cancel_button.setMinimumSize(88, 38)
        self.btns.accepted.connect(self._on_accept)
        self.btns.rejected.connect(self.reject)
        lay.addWidget(self.btns)

        self.resize(660, 560)
        self.setMinimumSize(620, 520)
        self.edt_name.setFocus()

    def _load_project(self, project):
        """回填项目详情，编辑时保持创建表单字段一致。"""
        self.edt_name.setText(str(getattr(project, 'name', '') or ''))
        self.edt_org.setText(str(getattr(project, 'org_unit', '') or ''))
        self.edt_address.setText(str(getattr(project, 'address', '') or ''))
        self.edt_building.setText(str(getattr(project, 'building_floor', '') or ''))
        self.edt_remarks.setPlainText(str(getattr(project, 'remarks', '') or ''))

    def _on_accept(self):
        if not self.edt_name.text().strip():
            QMessageBox.warning(self, "提示", "项目名称为必填项！")
            self.edt_name.setFocus()
            return
        self.accept()

    def values(self) -> dict:
        """返回表单数据，地址由省市区 + 详细地址拼接而成"""
        prov, city, dist = self.region_selector.get_selected()
        # 组合省市区（只拼接非空部分）
        region_parts = [p for p in (prov, city, dist) if p]
        region_str = "".join(region_parts)  
        detail = self.edt_address.text().strip()
        full_address = region_str
        if detail:
            full_address = (region_str + " " + detail) if region_str else detail
        # 如果全部为空，则返回 None
        full_address = full_address.strip() or None

        return {
            "name": self.edt_name.text().strip(),
            "org_unit": self.edt_org.text().strip() or None,
            "address": full_address,
            "building_floor": self.edt_building.text().strip() or None,
            "remarks": self.edt_remarks.toPlainText().strip() or None,
        }
