from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QDate, QUrl, Qt, Signal
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
    QSizePolicy,
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


class ResourceListWidget(QListWidget):
    """支持从资源管理器拖入文件/文件夹的列表。

    交互契约：拖入的内容只作为"待选路径"交给对话框校验，与"新增"
    按钮走同一条 ``_add_paths`` 通道，因此校验、去重、落库行为完全一致，
    不引入第二套资源管理逻辑。
    """

    pathsDropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DragDropMode.DropOnly)
        self.setSelectionMode(QListWidget.SelectionMode.SingleSelection)

    @staticmethod
    def _local_paths(event) -> list:
        mime = event.mimeData()
        if not mime.hasUrls():
            return []
        return [url.toLocalFile() for url in mime.urls()
                if url.isLocalFile() and url.toLocalFile()]

    def dragEnterEvent(self, event):
        if self._local_paths(event):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if self._local_paths(event):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event):
        paths = self._local_paths(event)
        if not paths:
            super().dropEvent(event)
            return
        event.acceptProposedAction()
        self.pathsDropped.emit(paths)


class ProjectCreateDialog(QDialog):
    """
    创建/编辑项目四页签表单：
      - 基础信息
      - 报告信息
      - 检测参数
      - 文件导入

    改造要点（对外接口不变：仍只暴露 ``values()`` 与既有控件名）：
    - 必填项即时校验：项目名称失焦即提示，错误态用动态属性驱动 QSS；
    - 文件导入支持拖拽 + 计数提示 + 空态引导；
    - 检测参数按"尺度参数 / 判定阈值 / 离群剔除"分成三组卡片，
      仅调整排版，取值逻辑与键名保持原样。
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
        self.setObjectName('projectCreateDialog')
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 12)
        lay.setSpacing(10)

        header = QVBoxLayout()
        header.setSpacing(2)
        title = QLabel("请输入项目信息")
        title.setObjectName('formDialogTitle')
        subtitle = QLabel("检测参数将随项目保存，并在【项目操作】中统一下发使用")
        subtitle.setObjectName('formDialogSubtitle')
        header.addWidget(title)
        header.addWidget(subtitle)
        lay.addLayout(header)

        self.tabs = QTabWidget()
        self.tabs.setObjectName('formTabs')
        self.tabs.addTab(self._build_basic_tab(), "基础信息")
        self.tabs.addTab(self._build_report_tab(), "报告信息")
        self.tabs.addTab(self._build_inspection_tab(), "检测参数")
        self.tabs.addTab(self._build_import_tab(), "文件导入")
        lay.addWidget(self.tabs, 1)

        # 底部操作条：左侧常驻校验状态，右侧 Save/Cancel。
        # 单独用一个容器是为了让"当前 Tab 的名称"始终可见，
        # 用户切页后不会忘记要保存哪一页的修改。
        footer = QWidget()
        footer.setObjectName('formFooter')
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(10, 8, 10, 8)
        footer_layout.setSpacing(10)
        self.lbl_tab_hint = QLabel()
        self.lbl_tab_hint.setProperty('uiRole', 'formFooterHint')
        footer_layout.addWidget(self.lbl_tab_hint, 1)

        self.btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self.btns.accepted.connect(self._on_accept)
        self.btns.rejected.connect(self.reject)
        save_button = self.btns.button(QDialogButtonBox.StandardButton.Save)
        save_button.setProperty('buttonRole', 'primary')
        save_button.setText('保存')
        self.btns.button(QDialogButtonBox.StandardButton.Cancel).setText('取消')
        footer_layout.addWidget(self.btns)
        lay.addWidget(footer)

        self.tabs.currentChanged.connect(self._on_tab_changed)
        self._on_tab_changed(0)

        self.resize(640, 620)
        self.setMinimumSize(600, 520)

    def _on_tab_changed(self, index: int):
        name = self.tabs.tabText(index) if index >= 0 else ''
        self.lbl_tab_hint.setText(f'当前编辑：{name}')

    # ---------- 基础信息 ----------
    def _build_basic_tab(self) -> QWidget:
        page = QWidget()
        page.setObjectName('formTabPage')
        # 表单贴顶聚拢，多余高度留在底部，避免行间被均摊撑开
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        host = QWidget()
        host.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Maximum)
        outer.addWidget(host)
        outer.addStretch(1)
        form = QFormLayout(host)
        form.setContentsMargins(14, 14, 14, 14)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setSpacing(10)
        form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        name_box = QWidget()
        name_layout = QVBoxLayout(name_box)
        name_layout.setContentsMargins(0, 0, 0, 0)
        name_layout.setSpacing(3)
        self.edt_name = QLineEdit()
        self.edt_name.setPlaceholderText("必填，如：XX花园1号楼外立面")
        self.edt_name.textChanged.connect(self._update_report_no)
        self.edt_name.textChanged.connect(self._clear_name_error)
        self.edt_name.editingFinished.connect(self._validate_name)
        name_layout.addWidget(self.edt_name)
        self.lbl_name_hint = QLabel("项目名称用于生成报告编号与报告封面，建议包含楼栋信息")
        self.lbl_name_hint.setProperty('uiRole', 'fieldHint')
        name_layout.addWidget(self.lbl_name_hint)
        form.addRow(self._required_label("项目名称："), name_box)

        self.edt_org = QLineEdit()
        self.edt_org.setPlaceholderText("选填")
        form.addRow("所属单位：", self.edt_org)

        self.region_selector = RegionSelector()
        form.addRow("省市区：", self.region_selector)

        self.edt_address = QLineEdit()
        self.edt_address.setPlaceholderText("具体地址，如街道门牌号")
        form.addRow("详细地址：", self.edt_address)

        self.edt_building = QLineEdit()
        self.edt_building.setPlaceholderText("如：1号楼 / A区3栋")
        form.addRow("楼栋号信息：", self.edt_building)

        self.edt_remarks = QTextEdit()
        self.edt_remarks.setPlaceholderText("选填，可填写现场情况说明")
        self.edt_remarks.setFixedHeight(80)
        form.addRow("备注：", self.edt_remarks)

        return page

    def _required_label(self, text: str) -> QWidget:
        """带红色星号的字段标签，让必填项在视觉上先被看到。"""
        holder = QWidget()
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(2)
        layout.addStretch(1)
        label = QLabel(text)
        mark = QLabel("*")
        mark.setProperty('uiRole', 'requiredMark')
        layout.addWidget(label)
        layout.addWidget(mark)
        return holder

    @staticmethod
    def _repolish(widget: QWidget):
        """动态属性改变后必须手动重新应用样式表，Qt 不会自动刷新。"""
        style = widget.style()
        style.unpolish(widget)
        style.polish(widget)
        widget.update()

    # ---------- 内联校验 ----------
    def _validate_name(self) -> bool:
        """校验项目名称并把结果写进控件的 ``fieldState`` 动态属性。

        返回是否通过，供 ``_on_accept`` 复用，保证"失焦提示"与
        "保存拦截"使用的是同一套判定，不会出现口径不一致。
        """
        ok = bool(self.edt_name.text().strip())
        state = '' if ok else 'error'
        self.edt_name.setProperty('fieldState', state)
        self.lbl_name_hint.setProperty('fieldState', state)
        self.lbl_name_hint.setText(
            "项目名称用于生成报告编号与报告封面，建议包含楼栋信息" if ok
            else "项目名称为必填项，请填写后再保存")
        self._repolish(self.edt_name)
        self._repolish(self.lbl_name_hint)
        return ok

    def _clear_name_error(self, text: str):
        """用户开始输入后立即消除错误态，避免红框一直挂着。"""
        if text.strip() and self.edt_name.property('fieldState') == 'error':
            self._validate_name()

    # ---------- 文件导入 ----------
    def _build_import_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)
        self._resource_lists = {}
        self._resource_counts = {}
        self._resource_empties = {}
        sections = (
            ('fls_directories', 'FLS 目录导入', '上传单站点原生 FLS 文件夹', 'directory'),
            ('pointcloud_files', '点云文件上传', '上传多站点拼接处理后点云文件 (E57/PLY)', 'pointcloud'),
            ('photo_files', '现场照片上传', '上传对应站点拍摄 2D 照片', 'photo'),
        )
        for key, title, hint, kind in sections:
            box = QGroupBox(title)
            box.setObjectName('importGroup')
            layout = QVBoxLayout(box)
            layout.setSpacing(6)

            # 顶部一行：左侧说明，右侧实时计数 + 操作按钮，列表区域保持干净。
            head = QHBoxLayout()
            hint_label = QLabel(hint)
            hint_label.setProperty('uiRole', 'importEmptyHint')
            head.addWidget(hint_label, 1)
            counter = QLabel()
            counter.setProperty('uiRole', 'importEmptyHint')
            head.addWidget(counter, 0, Qt.AlignmentFlag.AlignRight)
            add = QPushButton('新增')
            clear = QPushButton('清空')
            add.setProperty('buttonRole', 'primary')
            clear.setProperty('buttonRole', 'danger')
            add.clicked.connect(lambda _=False, k=key, t=title, ty=kind: self._add_resource(k, t, ty))
            clear.clicked.connect(lambda _=False, k=key, t=title: self._clear_resources(k, t))
            head.addWidget(add)
            head.addWidget(clear)
            layout.addLayout(head)

            # 列表支持从资源管理器直接拖入，交互路径与"新增"按钮完全一致。
            listing = ResourceListWidget()
            listing.setObjectName(f'{key}List')
            listing.setProperty('uiRole', 'resourceList')
            listing.setMinimumHeight(72)
            listing.setToolTip('可直接从资源管理器拖入文件或文件夹')
            listing.pathsDropped.connect(
                lambda paths, k=key, t=title: self._add_paths(k, t, paths))
            self._resource_lists[key] = listing
            self._resource_counts[key] = counter
            layout.addWidget(listing, 1)

            empty = QLabel('尚未添加内容，可点击「新增」或直接拖拽文件到此处')
            empty.setProperty('uiRole', 'importEmptyHint')
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._resource_empties[key] = empty
            layout.addWidget(empty)

            outer.addWidget(box, 1)
            self._refresh_resource_state(key)
        outer.addStretch(1)
        return page

    def _refresh_resource_state(self, key: str):
        """同步"计数标签 + 空态提示"，让列表当前状态一眼可见。"""
        count = self._resource_lists[key].count()
        counter = self._resource_counts.get(key)
        if counter is not None:
            counter.setText(f'已添加 {count} 项')
        empty = self._resource_empties.get(key)
        if empty is not None:
            empty.setVisible(count == 0)

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
        self._add_paths(key, title, paths)

    def _add_paths(self, key: str, title: str, paths) -> int:
        """统一的路径入库通道：拖拽与「新增」按钮共用，校验口径完全一致。

        返回实际新增条目数，供拖拽结束后的会话内反馈使用。
        """
        added = 0
        for path in paths or ():
            if not path:
                continue
            try:
                candidate = str(Path(path).expanduser().resolve())
            except OSError:
                continue
            if not Path(candidate).exists():
                QMessageBox.warning(self, title, f'路径不存在：\n{candidate}')
                continue
            if candidate in self._resource_values(key):
                continue
            self._append_resource_row(key, candidate)
            added += 1
        return added

    def _append_resource_row(self, key: str, path: str):
        listing = self._resource_lists[key]
        item = QListWidgetItem(listing)
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(6, 3, 4, 3)
        # 完整路径整行显示会撑破列表宽度，改为"文件名 + 原路径"，
        # 完整路径仍写入 tooltip，落库时从 tooltip 取值不受显示影响。
        name = Path(path).name or path
        label = QLabel(f'{name}    {path}' if name != path else path)
        label.setProperty('uiRole', 'resourcePath')
        label.setToolTip(path)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(label, 1)
        remove = QPushButton('移除')
        remove.setProperty('uiRole', 'resourceRemove')
        remove.setToolTip('仅从列表移除，不会删除磁盘文件')
        remove.clicked.connect(
            lambda _=False, i=item, l=listing, title=key:
            self._remove_resource(l, i, title))
        layout.addWidget(remove)
        item.setSizeHint(row.sizeHint())
        listing.addItem(item)
        listing.setItemWidget(item, row)
        self._refresh_resource_state(key)

    def _remove_resource(self, listing: QListWidget, item: QListWidgetItem,
                         title: str):
        if QMessageBox.question(
                self, '删除资源', f'确定删除该{title}资源条目吗？',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            listing.takeItem(listing.row(item))
            self._refresh_resource_state(self._resource_key_of(listing))

    def _resource_key_of(self, listing: QListWidget) -> str:
        """由控件反查分组键，供移除条目后刷新对应计数使用。"""
        for key, widget in self._resource_lists.items():
            if widget is listing:
                return key
        return ''

    def _resource_values(self, key: str) -> list[str]:
        """读取资源路径。

        行内文本为排版做了缩略，因此路径以 tooltip 为准；显示文本只在
        tooltip 缺失时兜底，保证落库值与用户在列表中看到的内容一致。
        """
        listing = self._resource_lists[key]
        values = []
        for index in range(listing.count()):
            row = listing.itemWidget(listing.item(index))
            if row is None:
                continue
            label = row.findChild(QLabel)
            if label is None:
                continue
            path = label.toolTip() or label.text()
            if path:
                values.append(path)
        return values

    def _clear_resources(self, key: str, title: str):
        if not self._resource_values(key):
            return
        if QMessageBox.question(self, title, '确定清空本板块全部资源吗？',
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            self._resource_lists[key].clear()
            self._refresh_resource_state(key)

    def _load_resources(self, fls, pointclouds, photos):
        for key, values in (('fls_directories', fls), ('pointcloud_files', pointclouds), ('photo_files', photos)):
            for path in dict.fromkeys(str(value) for value in (values or []) if value):
                self._append_resource_row(key, path)
            self._refresh_resource_state(key)

    # ---------- 报告信息 ----------
    def _build_report_tab(self) -> QWidget:
        page = QWidget()
        # 表单贴顶聚拢，多余高度留在底部
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        host = QWidget()
        host.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Maximum)
        outer.addWidget(host)
        outer.addStretch(1)
        form = QFormLayout(host)
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

    def _make_param_group(self, title: str) -> tuple[QGroupBox, QGridLayout]:
        """检测参数按语义分组的卡片容器，内部仍是两列表单网格。

        分组只影响排版：控件名、取值范围与 ``values()`` 的键名一律不变，
        因此不会影响参数向下传递与既有算法取值。
        """
        group = QGroupBox(title)
        group.setObjectName('paramGroup')
        grid = QGridLayout(group)
        grid.setContentsMargins(10, 14, 10, 8)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        return group, grid

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
        # 右侧给垂直滚动条让位，避免压到分组卡片边线
        layout.setContentsMargins(4, 4, 14, 4)

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

        # 标准对应的限值用"信息条"呈现，比灰色小字更易扫读。
        self.lbl_standard_hint = QLabel()
        self.lbl_standard_hint.setObjectName('standardHintChip')
        self.lbl_standard_hint.setWordWrap(True)
        layout.addWidget(self.lbl_standard_hint)

        # --- 参数分组：尺度参数 / 判定阈值 / SOR，逐段卡片化 ---
        scale_group, scale_grid = self._make_param_group('尺度参数')
        grids = {'current': scale_grid}

        def _add_row(row: int, label: str, widget: QWidget):
            lbl = QLabel(label)
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            current = grids['current']
            current.addWidget(lbl, row, 0)
            current.addWidget(widget, row, 1)
            return row + 1

        r = 0
        self.cb_interval = QComboBox()
        for value in (4.0, 8.0, 10.0, 20.0):
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
        layout.addWidget(scale_group)

        limit_group, limit_grid = self._make_param_group('判定阈值（由检测标准带出，可微调）')
        grids['current'] = limit_grid
        r = 0

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

        layout.addWidget(limit_group)

        # --- SOR 参数 ---
        sor_frame = QGroupBox("SOR 离群剔除")
        sor_frame.setObjectName('paramGroup')
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
        # 与失焦提示共用同一校验函数：不再用模态框遮挡字段，只在表单内
        # 高亮并切到出错的 Tab，用户修正后可直接再次保存。
        if not self._validate_name():
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