from pathlib import Path

from PySide6.QtCore import Qt, QTimer
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
REPORT_EMPTY_TITLE = '建筑外立面质量检测报告'
MODEL_EXPORT_FILTER = 'PLY 点云 (*.ply)'



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
            # 预览刷新是按节拍合并的，导出前确保拿到最新快照
            if getattr(self, '_report_preview_pending', False):
                self._rebuild_report_preview()
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
        # 一次项目激活会沿多条信号连锁触发刷新（项目切换 / 立面恢复 /
        # 预览请求），合并到同一事件循环节拍内只真正构建一次。
        if getattr(self, '_report_preview_pending', False):
            return
        self._report_preview_pending = True
        QTimer.singleShot(0, self._rebuild_report_preview)

    def _rebuild_report_preview(self):
        self._report_preview_pending = False
        if not hasattr(self, 'report_preview_browser'):
            return
        # 使用全量聚合数据生成报告，支持多站点增量拓展
        all_facades_by_station = getattr(
            self.project_operation_service, 'all_facade_results', None)
        if all_facades_by_station:
            self._report_snapshot = ReportDataService.build(
                self.current_project,
                facades_by_station=all_facades_by_station,
                project_root=getattr(self.current_project, 'directory_path', None))
        else:
            # 降级：回退到旧模式（当前活动站点）
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

        # 按改造要求删除热力图展示切换：报告页只保留 PDF 报告预览这一种内容。
        # report_navigation_stack 作为稳定的单页容器保留，避免影响既有装配
        # 顺序与 _set_report_navigation(0) 的调用方。
        self.report_navigation_stack.addWidget(report_preview_page)
        body_layout.addWidget(self.report_navigation_stack, 1)
        return page

    def _create_report_navigation(self):
        """创建报告页命令栏右侧的视图切换区。

        改造后报告页只剩 PDF 报告预览一种内容，热力图切换按钮已删除；
        这里返回的容器保留稳定对象名，仅承载“预览状态”提示，不再参与页面切换。
        """
        panel = QWidget()
        panel.setObjectName('reportNavigation')
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        navigation_layout = QHBoxLayout(panel)
        navigation_layout.setContentsMargins(0, 0, 0, 0)
        navigation_layout.setSpacing(8)

        self.report_navigation_group = QButtonGroup(self)
        self.report_navigation_group.setExclusive(True)
        # 只保留“报告预览”一项；热力图导航按钮已按改造要求移除。
        self.report_navigation_group.addButton(QPushButton(panel), 0)
        return panel

    def _set_report_navigation(self, page_index):
        """报告页内容唯一（PDF 预览），保留该方法作为稳定调用入口。"""
        if not 0 <= page_index < self.report_navigation_stack.count():
            return

        self._report_navigation_index = page_index
        self.report_navigation_stack.setCurrentIndex(page_index)
        self.report_document_title_label.setText(
            self._current_report_pdf_name or REPORT_EMPTY_TITLE)
        current_page_key = PAGE_DEFINITIONS[
            self.page_stack.currentIndex()
        ][1]
        if current_page_key == 'report_export':
            self._update_window_title('report_export')

    def _open_report_pdf(self):
        """【打开 PDF】已重构为【导出模型】。

        业务：选择当前项目的站点（或全部站点）→ 对源点云做 voxel=0.2 的
        downsample → 以 PLY 格式导出到用户指定路径。耗时步骤全部在后台
        线程池执行，并由 TaskProgressController 提供模态进度弹窗。
        """
        if self.current_project is None:
            QMessageBox.information(self, '导出模型', '请先创建或选择项目。')
            return
        stations = self._exportable_stations()
        if not stations:
            QMessageBox.information(
                self, '导出模型',
                '当前项目没有可用站点，请先导入并处理点云。')
            return

        station = self._prompt_export_station(stations)
        if station is None:
            return

        default_name = f'{self._sanitize_export_name(station.display_name)}_export.ply'
        default_dir = Path(self.current_project.directory_path) / 'exports'
        default_dir.mkdir(parents=True, exist_ok=True)
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            '导出模型',
            str(default_dir / default_name),
            MODEL_EXPORT_FILTER,
        )
        if not path:
            return
        self.model_export_controller.export_station(station, path)

    # ------------------------------------------------------------------
    # 【导出模型】入口辅助：站点枚举 / 站点选择 / 文件名清洗
    # ------------------------------------------------------------------
    def _exportable_stations(self):
        """返回当前项目可导出的站点（排除资产失效项）。"""
        controller = getattr(self, 'model_export_controller', None)
        if controller is not None:
            return controller.exportable_stations()
        return []

    def _prompt_export_station(self, stations):
        """选择要导出的站点；单站点直接返回，多站点弹对话框。"""
        controller = getattr(self, 'model_export_controller', None)
        if controller is None:
            return stations[0] if stations else None
        return controller.prompt_station(self, stations)

    @staticmethod
    def _sanitize_export_name(name):
        """把站点名转换为可安全用于文件名的字符串。"""
        text = str(name or '').strip() or 'station'
        return ''.join(ch if ch not in '\\/:*?"<>|' else '_' for ch in text)

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
