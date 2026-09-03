from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from qtwebview2 import QtWebView2Widget

from ui.main_window_config import PAGE_DEFINITIONS
from ui.widgets.window_chrome import ElidedLabel
from ui.widgets.technical_canvas import TechnicalCanvas
from services.report_export import ReportDataService, PdfReportRenderer


REPORT_PDF_FILTER = 'PDF 文件 (*.pdf)'
REPORT_EMPTY_TITLE = '请选择PDF上传'


class ReportPageMixin:
    def _export_quality_report(self):
        if self.current_project is None:
            QMessageBox.information(self, '导出报告', '请先创建或选择项目。')
            return
        default_dir = Path(self.current_project.directory_path) / 'reports'
        default_dir.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(self, '导出质量报告',
            str(default_dir / f'{self.current_project.name}_质量检测报告.pdf'), REPORT_PDF_FILTER)
        if not path:
            return
        try:
            PdfReportRenderer.write_pdf(self._report_html, path)
            self.report_export_service.register_pdf(
                getattr(self.current_project, 'project_id', None),
                path,
                '建筑外立面质量检测报告',
            )
            self._set_report_pdf_status(f'已导出：{Path(path).name}', 'success')
            self._current_report_pdf_name = Path(path).name
            self.statusBar().showMessage(f'报告已导出：{path}', 6000)
        except Exception as exc:
            QMessageBox.warning(self, '导出报告', f'报告导出失败：{exc}')

    def _refresh_report_preview(self):
        if not hasattr(self, 'report_preview_browser'):
            return
        facades = self.project_operation_service.last_facade_results or []
        self._report_snapshot = ReportDataService.build(
            self.current_project, facades,
            getattr(self.current_project, 'directory_path', None))
        self._report_html = PdfReportRenderer.html(self._report_snapshot)
        self.report_document_title_label.setText('建筑外立面质量检测报告')
        self.report_preview_browser.setHtml(self._report_html)
        self.report_preview_browser.setVisible(True)
        self.report_webview.setVisible(False)
        self.report_preview_state_stack.setCurrentIndex(1)
        self._set_report_pdf_status('在线报告预览已更新', 'success')

    def _create_report_export_page(self, page_title, page_key):
        page, body_layout = self._create_page_shell(
            page_title,
            page_key,
        )
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        self.report_navigation_stack = QStackedWidget()
        self.report_navigation_stack.setObjectName('reportNavigationStack')

        document_header = QWidget()
        document_header.setObjectName('reportDocumentHeader')
        document_header.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        document_header.setFixedHeight(48)
        document_header_layout = QHBoxLayout(document_header)
        document_header_layout.setContentsMargins(16, 0, 16, 0)
        document_header_layout.setSpacing(16)
        self.report_document_title_label = ElidedLabel(REPORT_EMPTY_TITLE)
        self.report_document_title_label.setObjectName('reportDocumentTitleLabel')
        self.report_document_title_label.setProperty('uiRole', 'sectionTitle')
        document_header_layout.addWidget(self.report_document_title_label, 1)

        self.report_pdf_status_label = QLabel('PDF未加载')
        self.report_pdf_status_label.setObjectName('reportPdfStatusLabel')
        self.report_pdf_status_label.setProperty('uiRole', 'supportingText')
        self.report_pdf_status_label.setProperty('statusState', 'neutral')
        document_header_layout.addWidget(self.report_pdf_status_label)
        body_layout.addWidget(document_header)

        report_preview_page = QWidget()
        report_preview_page.setObjectName('reportPreviewPage')
        report_preview_page.setProperty('uiRole', 'contentArea')
        report_preview_page.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground,
            True,
        )
        report_preview_layout = QVBoxLayout(report_preview_page)
        report_preview_layout.setContentsMargins(0, 0, 0, 0)
        report_preview_layout.setSpacing(0)

        self.report_preview_state_stack = QStackedWidget()
        self.report_preview_state_stack.setObjectName('reportPreviewStateStack')

        report_empty_state = TechnicalCanvas('document')
        report_empty_state.setObjectName('reportEmptyState')
        self.report_preview_state_stack.addWidget(report_empty_state)

        report_document_page = QWidget()
        report_document_page.setObjectName('reportDocumentPage')
        report_document_layout = QVBoxLayout(report_document_page)
        report_document_layout.setContentsMargins(0, 0, 0, 0)
        report_document_layout.setSpacing(0)
        # QtWebView2Widget 只嵌入系统 WebView2 内容区，不创建浏览器地址栏。
        self.report_preview_browser = QTextBrowser(report_document_page)
        self.report_preview_browser.setOpenExternalLinks(False)
        self.report_preview_browser.setStyleSheet('QTextBrowser { background: #f8fafc; border: 0; }')
        self.report_preview_browser.setHtml('<h2>报告预览</h2><p>创建或选择项目后将自动生成报告预览。</p>')
        report_document_layout.addWidget(self.report_preview_browser, 1)
        self.report_webview = QtWebView2Widget(
            url='about:blank',
            debug=False,
            context_menus=False,
            background_color='#f8fafc',
            parent=report_document_page,
        )
        self.report_webview.setObjectName('reportPdfWebView')
        self.report_webview.bridge.initialization_done.connect(
            self._on_report_webview_initialized
        )
        self.report_webview.bridge.domContentLoaded.connect(
            self._on_report_pdf_loaded
        )
        self.report_webview.setVisible(False)
        self.report_preview_state_stack.addWidget(report_document_page)
        report_preview_layout.addWidget(self.report_preview_state_stack, 1)

        heatmap_page = QWidget()
        heatmap_page.setObjectName('reportHeatmapPage')
        heatmap_page.setProperty('uiRole', 'contentArea')
        heatmap_page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        heatmap_layout = QVBoxLayout(heatmap_page)
        heatmap_layout.setContentsMargins(0, 0, 0, 0)

        # 保留稳定的控件接口，后续算法视口可以直接替换此占位控件。
        heatmap_placeholder = TechnicalCanvas('heatmap')
        heatmap_placeholder.setObjectName('heatmapPlaceholder')
        heatmap_layout.addWidget(heatmap_placeholder, 1)

        self.report_navigation_stack.addWidget(report_preview_page)
        self.report_navigation_stack.addWidget(heatmap_page)
        body_layout.addWidget(self.report_navigation_stack, 1)
        return page

    def _create_report_navigation(self):
        """创建报告预览和热力图之间的页面内导航。"""
        panel = QWidget()
        panel.setObjectName('reportNavigation')
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        navigation_layout = QHBoxLayout(panel)
        navigation_layout.setContentsMargins(0, 0, 0, 0)
        navigation_layout.setSpacing(8)

        self.report_navigation_group = QButtonGroup(self)
        self.report_navigation_group.setExclusive(True)
        navigation_items = (
            ('报告预览', 'btn_report_preview_navigation'),
            ('热力图', 'btn_heatmap_navigation'),
        )
        for index, (label, object_name) in enumerate(navigation_items):
            button = QPushButton(label)
            button.setObjectName(object_name)
            button.setProperty('uiRole', 'navigationItem')
            button.setProperty('navigationLevel', 'internal')
            button.setCheckable(True)
            button.setChecked(index == self._report_navigation_index)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(
                lambda _checked=False, target_index=index:
                self._set_report_navigation(target_index)
            )
            self.report_navigation_group.addButton(button, index)
            navigation_layout.addWidget(button)

        return panel

    def _set_report_navigation(self, page_index):
        """切换报告页内部内容，不改变底部四个主页面。"""
        if not 0 <= page_index < self.report_navigation_stack.count():
            return

        self._report_navigation_index = page_index
        self.report_navigation_stack.setCurrentIndex(page_index)
        button = self.report_navigation_group.button(page_index)
        if button is not None:
            button.setChecked(True)

        title = (
            '热力图'
            if page_index == 1
            else self._current_report_pdf_name or REPORT_EMPTY_TITLE
        )
        self.report_document_title_label.setText(title)
        current_page_key = PAGE_DEFINITIONS[
            self.page_stack.currentIndex()
        ][1]
        if current_page_key == 'report_export':
            self._update_window_title('report_export')

    def _open_report_pdf(self):
        pdf_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            '选择 PDF 报告',
            self._last_upload_directory,
            REPORT_PDF_FILTER,
        )
        if not pdf_path:
            return

        self._last_upload_directory = str(Path(pdf_path).parent)
        self.show_report_pdf(pdf_path)

    def _set_report_pdf_status(self, text, state='neutral'):
        """Update PDF status text and its Corporate Clean semantic color."""
        self.report_pdf_status_label.setText(text)
        self.report_pdf_status_label.setProperty('statusState', state)
        # Dynamic QSS properties need a repolish before their selector updates.
        style = self.report_pdf_status_label.style()
        style.unpolish(self.report_pdf_status_label)
        style.polish(self.report_pdf_status_label)
        self.report_pdf_status_label.update()

    def show_report_pdf(self, pdf_path):
        try:
            document = self.report_export_service.prepare_pdf(pdf_path)
        except ValueError as exc:
            QMessageBox.warning(
                self,
                '打开 PDF',
                str(exc),
            )
            return

        self._set_report_pdf_status(
            f'正在加载：{document.name}',
            'loading',
        )
        self.report_preview_state_stack.setCurrentIndex(0)
        self.report_preview_browser.setVisible(False)
        self.report_webview.setVisible(True)
        self._current_report_pdf_name = document.name
        self._set_report_navigation(0)
        self.report_webview.load_url(document.uri)

    def _on_report_webview_initialized(self, success, error_message):
        if success:
            if self._current_report_pdf_name is None:
                self._set_report_pdf_status('PDF未加载')
            return

        self._report_webview_error = error_message
        self.report_preview_state_stack.setCurrentIndex(0)
        if self._current_report_pdf_name is None:
            self._set_report_pdf_status('PDF未加载')
            return

        self._set_report_pdf_status('WebView2 初始化失败', 'error')
        QMessageBox.warning(
            self,
            'WebView2 初始化失败',
            error_message or '请检查 Microsoft Edge WebView2 Runtime。',
        )

    def _on_report_pdf_loaded(self):
        if self._current_report_pdf_name is not None:
            self._set_report_pdf_status('PDF 已加载', 'success')
            self.report_preview_state_stack.setCurrentIndex(1)
