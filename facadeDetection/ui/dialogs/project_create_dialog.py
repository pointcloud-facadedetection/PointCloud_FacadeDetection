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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("创建项目")
        self.setModal(True)
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        title = QLabel("请输入项目信息")
        title.setStyleSheet("font-size:16px; font-weight:600; color:#333;")
        lay.addWidget(title)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)

        # 项目名称（必填）
        self.edt_name = QLineEdit()
        self.edt_name.setPlaceholderText("必填")
        form.addRow("项目名称：", self.edt_name)

        # 单位（选填）
        self.edt_org = QLineEdit()
        self.edt_org.setPlaceholderText("选填")
        form.addRow("所属单位：", self.edt_org)

        # 省市区三级联动（选填）
        self.region_selector = RegionSelector()
        form.addRow("省市区：", self.region_selector)

        # 详细地址（选填）
        self.edt_address = QLineEdit()
        self.edt_address.setPlaceholderText("具体地址，如街道")
        form.addRow("详细地址：", self.edt_address)

        # 楼栋号（选填）
        self.edt_building = QLineEdit(); 
        self.edt_building.setPlaceholderText('如：1号楼')
        form.addRow('楼栋号信息：', self.edt_building)

        # 备注（选填）
        self.edt_remarks = QTextEdit()
        self.edt_remarks.setPlaceholderText("选填")
        self.edt_remarks.setFixedHeight(80)
        form.addRow("备注：", self.edt_remarks)

        lay.addLayout(form)

        # 按钮
        self.btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self.btns.accepted.connect(self._on_accept)
        self.btns.rejected.connect(self.reject)
        lay.addWidget(self.btns)

        self.resize(480, 360)
        self.edt_name.setFocus()

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