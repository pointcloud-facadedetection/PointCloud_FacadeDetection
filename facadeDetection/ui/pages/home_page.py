"""Project overview page for the facade detection desktop application."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..styles import PLACEHOLDER, make_button


PROJECT_FIELDS = (
    ('检测位置', 'location'),
    ('检测范围', 'scope'),
    ('检测日期', 'date'),
    ('墙体数量', 'wall_count'),
    ('合格率', 'pass_rate'),
    ('更新时间', 'updated_at'),
)


class ProjectCard(QFrame):
    """One project card shown in the two-column project grid."""

    enter_requested = Signal(str)
    report_requested = Signal(str)
    delete_requested = Signal(str)
    action_requested = Signal(str, str)

    def __init__(
        self,
        project_id: str,
        project_data: dict | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.project_id = project_id
        self.project_data = dict(project_data or {})
        self.setObjectName(f'projectCard_{project_id}')
        self.setProperty('role', 'card')
        self.setMinimumHeight(250)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(15)

        heading = QHBoxLayout()
        heading.setSpacing(12)

        title = QLabel(self.project_data.get('project_name', ''))
        title.setObjectName(f'project_{project_id}_title')
        title.setProperty('role', 'cardTitle')
        heading.addWidget(title)
        heading.addStretch(1)

        status_text = self.project_data.get('status', '')
        status = QLabel(status_text)
        status.setObjectName(f'project_{project_id}_status')
        status.setProperty('role', 'statusPill')
        status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status.setVisible(bool(status_text))
        heading.addWidget(status)
        layout.addLayout(heading)

        divider = QFrame()
        divider.setObjectName(f'project_{project_id}_divider')
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet('color: #E8EAED;')
        layout.addWidget(divider)

        details = QGridLayout()
        details.setContentsMargins(0, 0, 0, 0)
        details.setHorizontalSpacing(28)
        details.setVerticalSpacing(14)
        for index, (field_name, field_key) in enumerate(PROJECT_FIELDS):
            field = QWidget()
            field.setObjectName(f'project_{project_id}_{field_key}')
            field_layout = QVBoxLayout(field)
            field_layout.setContentsMargins(0, 0, 0, 0)
            field_layout.setSpacing(4)

            label = QLabel(field_name)
            label.setObjectName(f'project_{project_id}_{field_key}_label')
            label.setProperty('role', 'muted')
            field_layout.addWidget(label)

            value = QLabel(self._field_value(field_key))
            value.setObjectName(f'project_{project_id}_{field_key}_value')
            value.setProperty('role', 'placeholder')
            field_layout.addWidget(value)

            details.addWidget(field, index // 2, index % 2)
        details.setColumnStretch(0, 1)
        details.setColumnStretch(1, 1)
        layout.addLayout(details)
        layout.addStretch(1)

        actions = QHBoxLayout()
        actions.setSpacing(10)
        enter_button = make_button(
            '进入项目', f'enter_project_{project_id}', 'primary', height=38
        )
        export_button = make_button(
            '导出PDF', f'export_project_{project_id}', 'secondary', height=38
        )
        delete_button = make_button(
            '删除', f'delete_project_{project_id}', 'danger', height=38
        )
        actions.addWidget(enter_button)
        actions.addWidget(export_button)
        actions.addStretch(1)
        actions.addWidget(delete_button)
        layout.addLayout(actions)

        enter_button.clicked.connect(self._request_enter)
        export_button.clicked.connect(self._request_report)
        delete_button.clicked.connect(self._request_delete)

    def _field_value(self, field_key: str) -> str:
        def text_value(source_key: str) -> str:
            value = self.project_data.get(source_key, '')
            return '' if value is None else str(value).strip()

        if field_key == 'scope':
            start = text_value('start_floor')
            end = text_value('end_floor')
            if start and end:
                return f'{start} 至 {end}'
            return start or end

        source_keys = {
            'location': 'location',
            'date': 'measurement_date',
            'wall_count': 'wall_count',
            'pass_rate': 'pass_rate',
            'updated_at': 'updated_at',
        }
        return text_value(source_keys[field_key])

    def _request_enter(self) -> None:
        self.action_requested.emit('enter_project', '进入项目')
        self.enter_requested.emit(self.project_id)

    def _request_report(self) -> None:
        self.action_requested.emit('export_project_pdf', '导出PDF')
        self.report_requested.emit(self.project_id)

    def _request_delete(self) -> None:
        self.action_requested.emit('delete_project', '删除')
        self.delete_requested.emit(self.project_id)


class HomePage(QWidget):
    """Application landing page containing the project overview grid."""

    new_project_requested = Signal()
    enter_project_requested = Signal(str)
    report_requested = Signal(str)
    action_requested = Signal(str, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName('homePage')
        self._next_project_number = 1
        self._project_cards: dict[str, ProjectCard] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())
        root.addWidget(self._build_project_list(), 1)

    @property
    def project_count(self) -> int:
        return len(self._project_cards)

    @property
    def project_cards(self) -> tuple[ProjectCard, ...]:
        return tuple(self._project_cards.values())

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName('homeHeader')
        header.setProperty('role', 'topbar')
        header.setFixedHeight(64)

        layout = QHBoxLayout(header)
        layout.setContentsMargins(32, 0, 32, 0)
        layout.setSpacing(12)

        title = QLabel('外立面激光检测工作台')
        title.setObjectName('homeAppTitle')
        title.setStyleSheet(
            'color: #FFFFFF; font-size: 19px; font-weight: 700; background: transparent;'
        )
        layout.addWidget(title)
        layout.addStretch(1)

        device_status = QLabel(f'设备状态：{PLACEHOLDER}')
        device_status.setObjectName('deviceStatusPill')
        device_status.setProperty('role', 'devicePill')
        layout.addWidget(device_status)

        connect_button = make_button(
            '连接设备', 'connectDeviceButton', 'topGhost', height=38
        )
        new_project_button = make_button(
            '＋ 新建项目', 'newProjectButton', 'accent', height=38
        )
        settings_button = make_button('⚙', 'settingsButton', 'topGhost', height=38)
        settings_button.setFixedWidth(42)
        settings_button.setToolTip('设置')
        layout.addWidget(connect_button)
        layout.addWidget(new_project_button)
        layout.addWidget(settings_button)

        connect_button.clicked.connect(
            lambda: self.action_requested.emit('connect_device', '连接设备')
        )
        new_project_button.clicked.connect(self._request_new_project)
        settings_button.clicked.connect(
            lambda: self.action_requested.emit('settings', '设置')
        )
        return header

    def _build_project_list(self) -> QWidget:
        body = QWidget()
        body.setObjectName('homeBody')
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(32, 28, 32, 28)
        body_layout.setSpacing(20)

        heading = QHBoxLayout()
        heading.setSpacing(10)
        title = QLabel('项目列表')
        title.setObjectName('projectListTitle')
        title.setProperty('role', 'pageTitle')
        heading.addWidget(title)

        self.project_count_label = QLabel('共 0 个项目')
        self.project_count_label.setObjectName('projectCount')
        self.project_count_label.setProperty('role', 'muted')
        heading.addWidget(self.project_count_label)
        heading.addStretch(1)
        body_layout.addLayout(heading)

        scroll = QScrollArea()
        scroll.setObjectName('projectListScroll')
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        grid_host = QWidget()
        grid_host.setObjectName('projectGridHost')
        self.project_grid = QGridLayout(grid_host)
        self.project_grid.setObjectName('projectGrid')
        self.project_grid.setContentsMargins(0, 0, 0, 0)
        self.project_grid.setHorizontalSpacing(20)
        self.project_grid.setVerticalSpacing(20)
        self.project_grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.project_grid.setColumnStretch(0, 1)
        self.project_grid.setColumnStretch(1, 1)
        scroll.setWidget(grid_host)
        body_layout.addWidget(scroll, 1)
        return body

    def add_project(self, project_data: dict | None = None) -> ProjectCard:
        """Append a project card populated with the submitted form data."""
        project_id = str(self._next_project_number)
        self._next_project_number += 1

        card = ProjectCard(project_id, project_data)
        card.enter_requested.connect(self.enter_project_requested)
        card.report_requested.connect(self.report_requested)
        card.delete_requested.connect(self._delete_project)
        card.action_requested.connect(self.action_requested)
        self._project_cards[project_id] = card
        self._relayout_cards()
        self._update_count()
        return card

    def add_placeholder_project(self, project_data: dict | None = None) -> ProjectCard:
        """Backward-compatible alias used by existing callers and tests."""
        return self.add_project(project_data)

    def get_project_data(self, project_id: str) -> dict:
        card = self._project_cards.get(project_id)
        return dict(card.project_data) if card is not None else {}

    def _request_new_project(self) -> None:
        self.action_requested.emit('new_project', '新建项目')
        self.new_project_requested.emit()

    def _delete_project(self, project_id: str) -> None:
        card = self._project_cards.pop(project_id, None)
        if card is None:
            return
        self.project_grid.removeWidget(card)
        card.deleteLater()
        self._relayout_cards()
        self._update_count()

    def _relayout_cards(self) -> None:
        for card in self._project_cards.values():
            self.project_grid.removeWidget(card)
        for index, card in enumerate(self._project_cards.values()):
            self.project_grid.addWidget(card, index // 2, index % 2)

    def _update_count(self) -> None:
        self.project_count_label.setText(f'共 {self.project_count} 个项目')
