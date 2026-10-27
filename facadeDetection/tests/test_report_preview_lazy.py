from types import SimpleNamespace

from PySide6.QtWidgets import QLabel, QStackedWidget, QTextBrowser, QWidget

from ui.main_window_config import PAGE_DEFINITIONS
from ui.pages.report_page import ReportPageMixin


def _harness():
    harness = ReportPageMixin()
    harness.page_stack = QStackedWidget()
    for _ in PAGE_DEFINITIONS:
        harness.page_stack.addWidget(QWidget())
    harness.page_stack.setCurrentIndex(1)
    harness.report_preview_browser = QTextBrowser()
    harness.report_document_title_label = QLabel()
    harness.report_webview = QWidget()
    harness.report_preview_state_stack = QStackedWidget()
    harness.report_preview_state_stack.addWidget(QWidget())
    harness.project_operation_service = SimpleNamespace(
        all_facade_results={}, last_facade_results=[])
    harness.current_project = None
    harness._set_report_pdf_status = lambda *_args: None
    return harness


def test_hidden_report_defers_build_until_page_visible(qapp):
    harness = _harness()
    harness._refresh_report_preview()
    qapp.processEvents()
    assert harness._report_preview_dirty is True
    assert getattr(harness, '_report_preview_pending', False) is False
    assert not hasattr(harness, '_report_snapshot')

    harness.page_stack.setCurrentIndex(3)
    harness._refresh_report_preview()
    qapp.processEvents()
    assert hasattr(harness, '_report_snapshot')
    assert harness._report_preview_dirty is False
    assert getattr(harness, '_report_preview_pending', False) is False

    harness.page_stack.setCurrentIndex(1)
    previous_html = harness._report_html
    harness._refresh_report_preview()
    qapp.processEvents()
    assert harness._report_preview_dirty is True
    assert harness._report_html == previous_html
