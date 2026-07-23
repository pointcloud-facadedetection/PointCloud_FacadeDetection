"""Report preview page for the first-version desktop UI prototype."""

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ui.styles import PLACEHOLDER, make_button, set_active


REPORT_SECTIONS = (
    ('封面', 'cover'),
    ('一、项目基本信息', 'project_info'),
    ('二、现场照片', 'site_photos'),
    ('三、检测设备及原理', 'equipment'),
    ('四、整体情况汇总', 'overall_summary'),
    ('五、平整度检测结果', 'flatness'),
    ('六、垂直度检测结果', 'verticality'),
    ('七、检测区间缺陷说明', 'defect_notes'),
    ('八、结论与建议', 'conclusion'),
)

REPORT_EDITOR_FIELDS = (
    ('项目名称', 'project_name'),
    ('报告编号', 'report_number'),
    ('建设单位', 'developer'),
    ('施工单位', 'contractor'),
    ('检测单位', 'inspection_unit'),
    ('监理单位', 'supervisor'),
    ('检测日期', 'inspection_date'),
    ('报告日期', 'report_date'),
)


class ReportPage(QWidget):
    """Report page with navigation, paper preview and an editable side drawer."""

    back_requested = Signal()
    action_requested = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('reportPage')
        self.toc_buttons = {}
        self.editor_fields = {}
        self.paper_metadata_values = {}
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())

        body = QWidget()
        body.setObjectName('reportBody')
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        body_layout.addWidget(self._build_toc())
        body_layout.addWidget(self._build_preview(), 1)
        body_layout.addWidget(self._build_editor())
        root.addWidget(body, 1)

    def _build_header(self):
        header = QFrame()
        header.setObjectName('reportHeader')
        header.setProperty('role', 'topbar')
        header.setFixedHeight(60)

        layout = QHBoxLayout(header)
        layout.setContentsMargins(18, 8, 18, 8)
        layout.setSpacing(10)

        back_button = make_button(
            '← 返回工作台', 'report_back', 'topGhost', height=40
        )
        back_button.clicked.connect(lambda _checked=False: self.back_requested.emit())
        layout.addWidget(back_button)

        title_box = QVBoxLayout()
        title_box.setContentsMargins(6, 0, 0, 0)
        title_box.setSpacing(0)
        title = QLabel('检测报告预览')
        title.setObjectName('reportHeaderTitle')
        title.setStyleSheet('color: white; font-size: 16px; font-weight: 700;')
        report_number = QLabel(f'报告编号：{PLACEHOLDER}')
        report_number.setObjectName('reportHeaderNumber')
        report_number.setStyleSheet('color: #AEB9C6; font-size: 11px;')
        self.report_header_number = report_number
        title_box.addWidget(title)
        title_box.addWidget(report_number)
        layout.addLayout(title_box)
        layout.addStretch(1)

        refresh_button = make_button(
            '刷新报告数据', 'report_refresh', 'topGhost', height=40
        )
        edit_button = make_button(
            '编辑报告信息', 'report_edit', 'topGhost', height=40
        )
        export_button = make_button(
            '导出 PDF', 'report_export_pdf', 'accent', height=40
        )
        print_button = make_button('打印', 'report_print', 'topGhost', height=40)

        self._connect_action(refresh_button, 'refresh_report_data')
        edit_button.clicked.connect(self._toggle_editor)
        self._connect_action(export_button, 'export_report_pdf')
        self._connect_action(print_button, 'print_report')

        layout.addWidget(refresh_button)
        layout.addWidget(edit_button)
        layout.addWidget(export_button)
        layout.addWidget(print_button)

        self.back_button = back_button
        self.refresh_button = refresh_button
        self.edit_button = edit_button
        self.export_button = export_button
        self.print_button = print_button
        return header

    def _build_toc(self):
        toc = QFrame()
        toc.setObjectName('reportToc')
        toc.setFixedWidth(250)

        layout = QVBoxLayout(toc)
        layout.setContentsMargins(16, 18, 16, 18)
        layout.setSpacing(5)

        title = QLabel('报告目录')
        title.setObjectName('reportTocTitle')
        title.setProperty('role', 'sectionTitle')
        layout.addWidget(title)

        hint = QLabel('点击章节定位预览内容')
        hint.setProperty('role', 'muted')
        layout.addWidget(hint)
        layout.addSpacing(9)

        for index, (label, section_name) in enumerate(REPORT_SECTIONS):
            button = make_button(
                label,
                f'toc_{section_name}',
                'toc',
                height=40,
            )
            button.clicked.connect(
                lambda _checked=False, name=section_name, text=label: self._select_section(
                    name, text
                )
            )
            self.toc_buttons[section_name] = button
            layout.addWidget(button)
            set_active(button, index == 0)

        layout.addStretch(1)

        status = QFrame()
        status.setProperty('role', 'soft')
        status_layout = QVBoxLayout(status)
        status_layout.setContentsMargins(12, 10, 12, 10)
        status_layout.setSpacing(4)
        status_title = QLabel('报告状态')
        status_title.setProperty('role', 'muted')
        status_value = QLabel(PLACEHOLDER)
        status_value.setObjectName('reportStatusValue')
        status_value.setProperty('role', 'placeholder')
        status_layout.addWidget(status_title)
        status_layout.addWidget(status_value)
        layout.addWidget(status)
        return toc

    def _build_preview(self):
        preview = QFrame()
        preview.setObjectName('reportPreview')
        preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        layout = QVBoxLayout(preview)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_preview_toolbar())

        scroll = QScrollArea()
        scroll.setObjectName('reportPreviewScroll')
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)

        canvas = QWidget()
        canvas.setObjectName('reportCanvas')
        canvas_layout = QVBoxLayout(canvas)
        canvas_layout.setContentsMargins(28, 26, 28, 40)
        canvas_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        canvas_layout.addWidget(self._build_report_paper(), 0, Qt.AlignmentFlag.AlignTop)
        canvas_layout.addStretch(1)
        scroll.setWidget(canvas)
        layout.addWidget(scroll, 1)

        self.preview_scroll = scroll
        return preview

    def _build_preview_toolbar(self):
        toolbar = QFrame()
        toolbar.setObjectName('reportPreviewToolbar')
        toolbar.setFixedHeight(58)
        toolbar.setStyleSheet(
            'QFrame#reportPreviewToolbar {'
            'background: #FFFFFF; border-bottom: 1px solid #D8DDE4;}'
        )

        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(18, 8, 18, 8)
        layout.setSpacing(8)

        previous_button = make_button(
            '‹ 上一页', 'report_previous_page', 'secondary', height=36
        )
        next_button = make_button(
            '下一页 ›', 'report_next_page', 'secondary', height=36
        )
        page_label = QLabel(f'第 {PLACEHOLDER} 页 / 共 {PLACEHOLDER} 页')
        page_label.setObjectName('reportPageIndicator')
        page_label.setProperty('role', 'placeholder')

        zoom_out_button = make_button(
            '−', 'report_zoom_out', 'secondary', height=36
        )
        zoom_out_button.setFixedWidth(42)
        zoom_label = QLabel(PLACEHOLDER)
        zoom_label.setObjectName('reportZoomIndicator')
        zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        zoom_label.setFixedWidth(50)
        zoom_label.setProperty('role', 'placeholder')
        zoom_in_button = make_button('+', 'report_zoom_in', 'secondary', height=36)
        zoom_in_button.setFixedWidth(42)
        fullscreen_button = make_button(
            '⤢ 全屏', 'report_fullscreen', 'secondary', height=36
        )

        for button, action_name in (
            (previous_button, 'previous_report_page'),
            (next_button, 'next_report_page'),
            (zoom_out_button, 'report_zoom_out'),
            (zoom_in_button, 'report_zoom_in'),
            (fullscreen_button, 'report_fullscreen'),
        ):
            self._connect_action(button, action_name)

        layout.addWidget(previous_button)
        layout.addWidget(next_button)
        layout.addWidget(page_label)
        layout.addStretch(1)
        layout.addWidget(zoom_out_button)
        layout.addWidget(zoom_label)
        layout.addWidget(zoom_in_button)
        layout.addWidget(fullscreen_button)

        self.previous_button = previous_button
        self.next_button = next_button
        self.zoom_out_button = zoom_out_button
        self.zoom_in_button = zoom_in_button
        self.fullscreen_button = fullscreen_button
        return toolbar

    def _build_report_paper(self):
        paper = QFrame()
        paper.setObjectName('reportPaper')
        paper.setFixedWidth(640)
        paper.setMinimumHeight(905)
        paper.setStyleSheet(
            'QFrame#reportPaper {'
            'background: #FFFFFF; border: 1px solid #D8DDE4; border-radius: 2px;}'
        )

        layout = QVBoxLayout(paper)
        layout.setContentsMargins(56, 54, 56, 54)
        layout.setSpacing(17)

        organization = QLabel(PLACEHOLDER)
        organization.setObjectName('reportPaperOrganization')
        organization.setProperty('role', 'muted')
        organization.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.report_paper_organization = organization
        layout.addWidget(organization)

        title = QLabel('建筑外立面激光检测报告')
        title.setObjectName('reportPaperTitle')
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setWordWrap(True)
        title.setStyleSheet('color: #13294B; font-size: 25px; font-weight: 700;')
        layout.addWidget(title)

        report_number = QLabel(f'报告编号：{PLACEHOLDER}')
        report_number.setObjectName('reportPaperNumber')
        report_number.setAlignment(Qt.AlignmentFlag.AlignCenter)
        report_number.setProperty('role', 'placeholder')
        self.report_paper_number = report_number
        layout.addWidget(report_number)
        layout.addSpacing(32)

        metadata = QGridLayout()
        metadata.setHorizontalSpacing(24)
        metadata.setVerticalSpacing(18)
        metadata_field_names = (
            'project_name',
            'developer',
            'contractor',
            'inspection_unit',
            'measurement_date',
            'report_date',
        )
        for row, label_text in enumerate(
            ('项目名称', '建设单位', '施工单位', '检测单位', '检测日期', '报告日期')
        ):
            label = QLabel(label_text)
            label.setProperty('role', 'muted')
            value = QLabel(PLACEHOLDER)
            value.setObjectName(f'reportPaperValue{row + 1}')
            value.setProperty('role', 'placeholder')
            self.paper_metadata_values[metadata_field_names[row]] = value
            metadata.addWidget(label, row, 0)
            metadata.addWidget(value, row, 1)
        metadata.setColumnStretch(1, 1)
        layout.addLayout(metadata)
        layout.addStretch(1)

        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet('background: #D8DDE4; border: none;')
        layout.addWidget(divider)

        footer = QHBoxLayout()
        footer_left = QLabel(PLACEHOLDER)
        footer_left.setProperty('role', 'muted')
        footer_page = QLabel(PLACEHOLDER)
        footer_page.setProperty('role', 'muted')
        footer.addWidget(footer_left)
        footer.addStretch(1)
        footer.addWidget(footer_page)
        layout.addLayout(footer)

        self.report_paper = paper
        return paper

    def _build_editor(self):
        editor = QFrame()
        editor.setObjectName('reportEditor')
        editor.setFixedWidth(360)

        layout = QVBoxLayout(editor)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        header = QHBoxLayout()
        title = QLabel('编辑报告信息')
        title.setProperty('role', 'sectionTitle')
        close_button = make_button(
            '✕', 'report_editor_close', 'secondary', height=36
        )
        close_button.setFixedWidth(40)
        close_button.clicked.connect(lambda _checked=False: self.set_editor_visible(False))
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(close_button)
        layout.addLayout(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        form_host = QWidget()
        form = QVBoxLayout(form_host)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(12)

        for label_text, field_name in REPORT_EDITOR_FIELDS:
            field_box = QWidget()
            field_layout = QVBoxLayout(field_box)
            field_layout.setContentsMargins(0, 0, 0, 0)
            field_layout.setSpacing(5)
            label = QLabel(label_text)
            label.setProperty('role', 'muted')
            edit = QLineEdit(PLACEHOLDER)
            edit.setObjectName(f'report_editor_{field_name}')
            edit.setFixedHeight(42)
            self.editor_fields[field_name] = edit
            field_layout.addWidget(label)
            field_layout.addWidget(edit)
            form.addWidget(field_box)

        conclusion_label = QLabel('结论文字')
        conclusion_label.setProperty('role', 'muted')
        conclusion_edit = QTextEdit()
        conclusion_edit.setObjectName('report_editor_conclusion')
        conclusion_edit.setPlainText('')
        conclusion_edit.setMinimumHeight(130)
        self.editor_fields['conclusion'] = conclusion_edit
        form.addWidget(conclusion_label)
        form.addWidget(conclusion_edit)
        form.addStretch(1)
        scroll.setWidget(form_host)
        layout.addWidget(scroll, 1)

        actions = QHBoxLayout()
        cancel_button = make_button(
            '取消', 'report_editor_cancel', 'secondary', height=42
        )
        save_button = make_button(
            '保存', 'report_editor_save', 'accent', height=42
        )
        cancel_button.clicked.connect(
            lambda _checked=False: self.set_editor_visible(False)
        )
        save_button.clicked.connect(self._save_editor)
        actions.addWidget(cancel_button)
        actions.addWidget(save_button)
        layout.addLayout(actions)

        self.editor = editor
        self.editor_close_button = close_button
        self.editor_cancel_button = cancel_button
        self.editor_save_button = save_button
        editor.hide()
        return editor

    def _connect_action(self, button, action_name):
        label = button.text()
        button.clicked.connect(
            lambda _checked=False, name=action_name, text=label: self.action_requested.emit(
                name, text
            )
        )

    def _select_section(self, section_name, label):
        for name, button in self.toc_buttons.items():
            set_active(button, name == section_name)
        self.action_requested.emit('select_report_section', label)

    def _toggle_editor(self, _checked=False):
        self.set_editor_visible(self.editor.isHidden())
        self.action_requested.emit('edit_report_info', self.edit_button.text())

    def set_editor_visible(self, visible):
        self.editor.setVisible(visible)
        set_active(self.edit_button, visible)

    def _save_editor(self, _checked=False):
        self.action_requested.emit('save_report_info', self.editor_save_button.text())
        self.set_editor_visible(False)

    def set_project_data(self, data: dict):
        """Populate report preview and editor fields from project form data."""
        project_data = data or {}

        def value_for(field_name):
            value = project_data.get(field_name, '')
            if value is None or value == PLACEHOLDER:
                return ''
            return str(value).strip()

        report_number = value_for('report_number')
        inspection_unit = value_for('inspection_unit')

        self.report_header_number.setText(f'报告编号：{report_number}')
        self.report_paper_organization.setText(inspection_unit)
        self.report_paper_number.setText(f'报告编号：{report_number}')

        for field_name, label in self.paper_metadata_values.items():
            label.setText(value_for(field_name))

        editor_mapping = {
            'project_name': 'project_name',
            'report_number': 'report_number',
            'developer': 'developer',
            'contractor': 'contractor',
            'inspection_unit': 'inspection_unit',
            'supervisor': 'supervisor',
            'inspection_date': 'measurement_date',
            'report_date': 'report_date',
        }
        for editor_name, project_field_name in editor_mapping.items():
            self.editor_fields[editor_name].setText(value_for(project_field_name))

        self.editor_fields['conclusion'].setPlainText('')
