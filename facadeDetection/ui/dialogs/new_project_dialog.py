"""New-project placeholder dialog matching the reference prototype."""

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ui.styles import PLACEHOLDER, make_button


PROJECT_FIELDS = (
    ('项目名称', 'project_name'),
    ('报告编号', 'report_number'),
    ('建设单位', 'developer'),
    ('施工单位', 'contractor'),
    ('检测单位', 'inspection_unit'),
    ('监理单位', 'supervisor'),
    ('检测位置', 'location'),
    ('起始楼层', 'start_floor'),
    ('结束楼层', 'end_floor'),
    ('标准层高', 'floor_height'),
    ('检测标准', 'inspection_standard'),
    ('面层材质', 'surface_material'),
    ('测量日期', 'measurement_date'),
    ('报告日期', 'report_date'),
)


class NewProjectDialog(QDialog):
    project_created = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('newProjectDialog')
        self.setWindowTitle('新建项目')
        self.setModal(False)
        self.setMinimumSize(760, 680)
        self.resize(820, 760)
        self.fields = {}
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(18)

        title_row = QHBoxLayout()
        title_group = QVBoxLayout()
        title = QLabel('新建检测项目')
        title.setProperty('role', 'pageTitle')
        subtitle = QLabel('先建立项目框架，具体信息可在后续阶段补充。')
        subtitle.setProperty('role', 'muted')
        title_group.addWidget(title)
        title_group.addWidget(subtitle)
        title_row.addLayout(title_group)
        title_row.addStretch(1)
        close_button = make_button('✕', 'close_new_project', 'secondary')
        close_button.setFixedWidth(42)
        close_button.clicked.connect(self.reject)
        title_row.addWidget(close_button)
        root.addLayout(title_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        panel = QFrame()
        panel.setProperty('role', 'panel')
        form = QGridLayout(panel)
        form.setContentsMargins(24, 22, 24, 22)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(14)

        for index, (label_text, field_name) in enumerate(PROJECT_FIELDS):
            row = index // 2
            column = (index % 2) * 2
            field_box = QWidget()
            field_layout = QVBoxLayout(field_box)
            field_layout.setContentsMargins(0, 0, 0, 0)
            field_layout.setSpacing(6)
            label = QLabel(label_text)
            label.setProperty('role', 'muted')
            edit = QLineEdit(PLACEHOLDER)
            edit.setObjectName(f'new_project_{field_name}')
            edit.setFixedHeight(44)
            self.fields[field_name] = edit
            field_layout.addWidget(label)
            field_layout.addWidget(edit)
            form.addWidget(field_box, row, column, 1, 2)

        scroll.setWidget(panel)
        root.addWidget(scroll, 1)

        actions = QHBoxLayout()
        actions.addStretch(1)
        cancel_button = make_button('取消', 'cancel_new_project', 'secondary', height=44)
        create_button = make_button(
            '创建项目并进入工作台',
            'create_project_and_enter',
            'accent',
            height=44,
        )
        cancel_button.clicked.connect(self.reject)
        create_button.clicked.connect(self._submit)
        actions.addWidget(cancel_button)
        actions.addWidget(create_button)
        root.addLayout(actions)

        self.close_button = close_button
        self.cancel_button = cancel_button
        self.create_button = create_button

    def _submit(self):
        data = {
            name: self.fields[name].text().strip()
            for _, name in PROJECT_FIELDS
        }
        self.project_created.emit(data)
        self.accept()
