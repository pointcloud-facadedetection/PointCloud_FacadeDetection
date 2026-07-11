"""Left-side controls based on the project's first UI prototype."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QMouseEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QColorDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


SUPPORTED_POINT_CLOUDS = {".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".pts"}


class DropZone(QFrame):
    """Clickable point-cloud drop target."""

    files_selected = Signal(list)
    clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(118)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon = QLabel("＋")
        icon.setObjectName("dropIcon")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel("点击或拖拽点云文件")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint = QLabel("支持 PLY / PCD / XYZ / PTS")
        hint.setObjectName("dropHint")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon)
        layout.addWidget(title)
        layout.addWidget(hint)

    @staticmethod
    def _paths_from_mime(event: QDragEnterEvent | QDropEvent) -> list[str]:
        paths: list[str] = []
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path and Path(path).suffix.lower() in SUPPORTED_POINT_CLOUDS:
                    paths.append(path)
        return paths

    def _set_drag_active(self, active: bool) -> None:
        self.setProperty("dragActive", active)
        self.style().unpolish(self)
        self.style().polish(self)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if self._paths_from_mime(event):
            event.acceptProposedAction()
            self._set_drag_active(True)
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self._set_drag_active(False)
        event.accept()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        self._set_drag_active(False)
        paths = self._paths_from_mime(event)
        if paths:
            event.acceptProposedAction()
            self.files_selected.emit(paths)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class SectionCard(QFrame):
    """Small reusable section card used throughout the sidebar."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sectionCard")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(13, 12, 13, 13)
        self.body.setSpacing(8)
        heading = QLabel(title)
        heading.setObjectName("sectionTitle")
        self.body.addWidget(heading)


class ControlPanel(QFrame):
    """All controls reserved by the HTML design, exposed through Qt signals."""

    files_selected = Signal(list)
    cloud_selected = Signal(str)
    action_requested = Signal(str)
    value_changed = Signal(str, float)
    color_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("controlPanel")
        self.setMinimumWidth(340)
        self.setMaximumWidth(340)
        self._cloud_buttons: list[QPushButton] = []
        self._workflow_buttons: list[QPushButton] = []
        self._file_placeholder: QListWidgetItem | None = None
        self._selected_color = "#667eea"

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        content.setStyleSheet("background: transparent;")
        self.sections = QVBoxLayout(content)
        self.sections.setContentsMargins(14, 14, 14, 16)
        self.sections.setSpacing(12)
        self._build_sections()
        self.sections.addStretch(1)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("sidebarHeader")
        layout = QVBoxLayout(header)
        layout.setContentsMargins(19, 17, 18, 15)
        layout.setSpacing(4)
        title = QLabel("点云立面检测系统")
        title.setObjectName("appTitle")
        subtitle = QLabel("大规模点云处理 · 建筑立面检测 · 质量评估")
        subtitle.setObjectName("appSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        return header

    def _build_sections(self) -> None:
        upload = SectionCard("文件上传")
        self.drop_zone = DropZone()
        self.drop_zone.clicked.connect(self._choose_files)
        self.drop_zone.files_selected.connect(self.files_selected)
        upload.body.addWidget(self.drop_zone)
        self.sections.addWidget(upload)

        parameters = SectionCard("处理参数")
        self.voxel_slider = self._add_slider(
            parameters, "体素下采样大小", 0, 100, 5, lambda value: value / 100.0, "voxel_size"
        )
        self.point_size_slider = self._add_slider(
            parameters, "点大小缩放", 1, 500, 100, lambda value: value / 100.0, "point_size"
        )
        self.sections.addWidget(parameters)

        files = SectionCard("文件列表")
        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(92)
        self.file_list.setMaximumHeight(170)
        self._file_placeholder = QListWidgetItem("暂无文件，请上传")
        self._file_placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
        self.file_list.addItem(self._file_placeholder)
        self.file_list.currentItemChanged.connect(self._emit_cloud_selected)
        files.body.addWidget(self.file_list)
        self.sections.addWidget(files)

        processing = SectionCard("处理操作")
        method_row = QHBoxLayout()
        radius = QRadioButton("半径去噪")
        statistical = QRadioButton("统计滤波")
        radius.setChecked(True)
        self.denoise_group = QButtonGroup(self)
        self.denoise_group.addButton(radius, 0)
        self.denoise_group.addButton(statistical, 1)
        method_row.addWidget(radius)
        method_row.addWidget(statistical)
        processing.body.addLayout(method_row)
        processing.body.addWidget(self._action_button("去噪处理", "denoise"))
        processing.body.addWidget(self._action_button("计算法向量", "normals"))
        processing.body.addWidget(self._action_button("显示边界框", "bbox"))
        processing.body.addLayout(
            self._button_row(
                self._action_button("清空", "clear", danger=True),
                self._action_button("重置视角", "reset_view", requires_cloud=False),
            )
        )
        self.sections.addWidget(processing)

        facade = SectionCard("立面检测")
        self.min_area_slider = self._add_slider(
            facade, "最小立面面积 (m²)", 2, 100, 10, lambda value: value / 2.0, "min_area"
        )
        facade.body.addWidget(self._action_button("检测建筑立面", "detect_facades", primary=True))
        self.facade_list = QListWidget()
        self.facade_list.setMinimumHeight(72)
        self.facade_list.addItem(QListWidgetItem("检测结果将在这里显示"))
        self.facade_list.item(0).setFlags(Qt.ItemFlag.NoItemFlags)
        facade.body.addWidget(self.facade_list)
        facade.body.addLayout(
            self._button_row(
                self._action_button("恢复显示", "reset_facades", auto_enable=False),
                self._action_button("质量报告", "quality_report", auto_enable=False),
            )
        )
        self.sections.addWidget(facade)

        segmentation = SectionCard("框选细分")
        segmentation.body.addWidget(self._action_button("进入框选模式", "selection_mode"))
        segmentation.body.addWidget(
            self._action_button("分割框选区域", "segment_selection", primary=True, auto_enable=False)
        )
        segmentation.body.addWidget(
            self._action_button("恢复分割前颜色", "reset_segments", auto_enable=False)
        )
        self.sections.addWidget(segmentation)

        color = SectionCard("颜色设置")
        preset_row = QHBoxLayout()
        preset_row.setSpacing(6)
        self.color_preview = QLabel()
        self.color_preview.setFixedSize(28, 28)
        preset_row.addWidget(self.color_preview)
        for hex_color in ("#667eea", "#f56565", "#48bb78", "#ed8936", "#38b2ac", "#9f7aea"):
            button = QToolButton()
            button.setToolTip(hex_color)
            button.setFixedSize(28, 28)
            button.setStyleSheet(
                f"QToolButton {{ background: {hex_color}; border: 2px solid #383864; border-radius: 14px; }}"
                "QToolButton:hover { border-color: white; }"
            )
            button.clicked.connect(lambda checked=False, value=hex_color: self._select_color(value))
            preset_row.addWidget(button)
        preset_row.addStretch(1)
        choose_color = QToolButton()
        choose_color.setText("…")
        choose_color.setToolTip("选择其他颜色")
        choose_color.clicked.connect(self._choose_color)
        preset_row.addWidget(choose_color)
        color.body.addLayout(preset_row)
        color.body.addLayout(
            self._button_row(
                self._action_button("应用颜色", "apply_color"),
                self._action_button("恢复颜色", "reset_color", auto_enable=False),
            )
        )
        self.sections.addWidget(color)

        registration = SectionCard("点云配准")
        registration.body.addLayout(
            self._button_row(
                self._action_button("设为目标", "set_target"),
                self._action_button("设为源", "set_source"),
            )
        )
        registration.body.addLayout(
            self._button_row(
                self._action_button("进入配准", "registration_mode"),
                self._action_button("退出配准", "exit_registration", danger=True, auto_enable=False),
            )
        )
        registration.body.addLayout(
            self._button_row(
                self._action_button("添加点对", "add_correspondence", auto_enable=False),
                self._action_button("撤销点对", "undo_correspondence", auto_enable=False),
            )
        )
        registration.body.addLayout(
            self._button_row(
                self._action_button("执行配准", "execute_registration", primary=True, auto_enable=False),
                self._action_button("ICP 精配准", "icp_refine", auto_enable=False),
            )
        )
        self.registration_status = QLabel("对应点：0 对  |  源：未选  |  目标：未选")
        self.registration_status.setObjectName("mutedLabel")
        self.registration_status.setWordWrap(True)
        registration.body.addWidget(self.registration_status)
        self.sections.addWidget(registration)

        export = SectionCard("导出")
        export.body.addWidget(self._action_button("保存配准结果", "save_registration", auto_enable=False))
        self.sections.addWidget(export)

        self.set_cloud_controls_enabled(False)
        self._select_color(self._selected_color)

    def _add_slider(
        self,
        card: SectionCard,
        title: str,
        minimum: int,
        maximum: int,
        value: int,
        converter,
        key: str,
    ) -> QSlider:
        label_row = QHBoxLayout()
        label = QLabel(title)
        label.setObjectName("mutedLabel")
        value_label = QLabel(f"{converter(value):g}")
        value_label.setObjectName("sceneValue")
        label_row.addWidget(label)
        label_row.addStretch(1)
        label_row.addWidget(value_label)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)

        def changed(raw_value: int) -> None:
            converted = float(converter(raw_value))
            value_label.setText(f"{converted:g}")
            self.value_changed.emit(key, converted)

        slider.valueChanged.connect(changed)
        card.body.addLayout(label_row)
        card.body.addWidget(slider)
        return slider

    def _action_button(
        self,
        text: str,
        action: str,
        *,
        primary: bool = False,
        danger: bool = False,
        requires_cloud: bool = True,
        auto_enable: bool = True,
    ) -> QPushButton:
        button = QPushButton(text)
        if primary:
            button.setProperty("primary", True)
        if danger:
            button.setProperty("danger", True)
        button.clicked.connect(lambda checked=False, name=action: self.action_requested.emit(name))
        if requires_cloud and auto_enable:
            self._cloud_buttons.append(button)
        elif requires_cloud:
            button.setEnabled(False)
            self._workflow_buttons.append(button)
        return button

    @staticmethod
    def _button_row(*buttons: QPushButton) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(7)
        for button in buttons:
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            row.addWidget(button)
        return row

    def _choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "导入点云",
            "",
            "点云文件 (*.ply *.pcd *.xyz *.xyzn *.xyzrgb *.pts);;所有文件 (*)",
        )
        if paths:
            self.files_selected.emit(paths)

    def _select_color(self, color: str) -> None:
        self._selected_color = color
        if hasattr(self, "color_preview"):
            self.color_preview.setStyleSheet(
                f"background: {color}; border: 2px solid #e0e0ff; border-radius: 14px;"
            )
            self.color_preview.setToolTip(f"当前颜色：{color}")
        self.color_changed.emit(color)

    def _emit_cloud_selected(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        del previous
        if current is None:
            return
        name = current.data(Qt.ItemDataRole.UserRole)
        if name:
            self.cloud_selected.emit(str(name))

    def _choose_color(self) -> None:
        color = QColorDialog.getColor(QColor(self._selected_color), self, "选择点云颜色")
        if color.isValid():
            self._select_color(color.name())

    def set_cloud_controls_enabled(self, enabled: bool) -> None:
        for button in self._cloud_buttons:
            button.setEnabled(enabled)

    def register_cloud(self, name: str, point_count: int) -> None:
        if self._file_placeholder is not None:
            row = self.file_list.row(self._file_placeholder)
            self.file_list.takeItem(row)
            self._file_placeholder = None
        item = QListWidgetItem(f"{name}\n{point_count:,} 点")
        item.setData(Qt.ItemDataRole.UserRole, name)
        self.file_list.addItem(item)
        self.file_list.setCurrentItem(item)
        self.set_cloud_controls_enabled(True)

    def clear_clouds(self) -> None:
        self.file_list.clear()
        self._file_placeholder = QListWidgetItem("暂无文件，请上传")
        self._file_placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
        self.file_list.addItem(self._file_placeholder)
        self.set_cloud_controls_enabled(False)

    @property
    def selected_color(self) -> str:
        return self._selected_color

    @property
    def denoise_method(self) -> str:
        return "radius" if self.denoise_group.checkedId() == 0 else "statistical"
