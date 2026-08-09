"""Corporate Clean theme shared by every PySide6 workspace page.

The visual hierarchy follows the reference site's functional-first rules:
one white workspace surface, thin borders, restrained blue anchors and no
decorative gradients or nested-card shadows.
"""

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication


# QPalette uses these tokens directly; the QSS below mirrors the same values
# so business widgets never carry page-specific color declarations.
COLORS = {
    "primary": "#1E40AF",
    "primary_hover": "#1D4ED8",
    "accent": "#3B82F6",
    "accent_soft": "#EFF6FF",
    "background": "#F8FAFC",
    "surface": "#FFFFFF",
    "surface_muted": "#F1F5F9",
    "text": "#0F172A",
    "text_secondary": "#334155",
    "text_muted": "#64748B",
    "border": "#E2E8F0",
    "border_strong": "#CBD5E1",
    "success": "#047857",
    "danger": "#B91C1C",
    "danger_hover": "#DC2626",
    "danger_soft": "#FEF2F2",
    "viewport": "#111827",
}


APPLICATION_STYLE_SHEET = """
QMainWindow#mainWindow,
QWidget#applicationShell,
QStackedWidget#pageStack,
QWidget[pageRole="workspace"],
QWidget[uiRole="contentArea"] {
    background-color: #F8FAFC;
}

QWidget#applicationHeader {
    background-color: #FFFFFF;
    border: none;
    border-bottom: 1px solid #E2E8F0;
}

QLabel#applicationBrandMark {
    color: #FFFFFF;
    background-color: #1E40AF;
    border: 1px solid #1E40AF;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 700;
}

QLabel#applicationBrandTitle {
    color: #0F172A;
    background-color: transparent;
    font-size: 16px;
    font-weight: 700;
}

QLabel#currentProjectLabel {
    color: #334155;
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 12px;
    font-weight: 500;
}

QWidget[uiRole="pageHeadingRow"] {
    background-color: transparent;
    border: none;
}

QLabel[uiRole="pageTitle"] {
    min-height: 32px;
    color: #0F172A;
    background-color: transparent;
    border: none;
    padding: 0;
    font-size: 24px;
    font-weight: 700;
}

QLabel[uiRole="sectionTitle"] {
    color: #0F172A;
    background-color: transparent;
    font-size: 16px;
    font-weight: 600;
}

QLabel[uiRole="supportingText"] {
    color: #64748B;
    background-color: transparent;
    font-size: 12px;
    font-weight: 400;
}

QFrame[uiRole="workspaceSurface"] {
    background-color: #FFFFFF;
    border: 1px solid #DCE3EC;
    border-radius: 8px;
}

QWidget[uiRole="workspaceBody"],
QWidget#overviewColumns {
    background-color: #FFFFFF;
    border: none;
    border-bottom-left-radius: 7px;
    border-bottom-right-radius: 7px;
}

QWidget[uiRole="pageHeader"] {
    background-color: #F8FAFC;
    border: none;
    border-bottom: 1px solid #E2E8F0;
    border-top-left-radius: 7px;
    border-top-right-radius: 7px;
}

QFrame[uiRole="commandGroup"] {
    background-color: transparent;
    border: none;
    border-right: 1px solid #E2E8F0;
    border-radius: 0;
}

QFrame[uiRole="commandGroup"][groupLast="true"] {
    border-right: none;
}

QLabel[uiRole="commandGroupLabel"] {
    color: #64748B;
    background-color: transparent;
    border: none;
    padding: 0 2px;
    font-size: 12px;
    font-weight: 500;
}

QPushButton {
    min-height: 20px;
    padding: 7px 14px;
    color: #334155;
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    font-size: 14px;
    font-weight: 500;
}

QPushButton:hover {
    color: #1E40AF;
    background-color: #F8FAFC;
    border-color: #93C5FD;
}

QPushButton:pressed {
    color: #1E3A8A;
    background-color: #EFF6FF;
    border-color: #3B82F6;
}

QPushButton:focus {
    border-color: #3B82F6;
}

QPushButton:disabled {
    color: #94A3B8;
    background-color: #F1F5F9;
    border-color: #E2E8F0;
}

QPushButton[uiRole="headerAction"] {
    min-height: 20px;
    padding: 7px 14px;
}

QPushButton[buttonRole="primary"],
QPushButton[buttonRole="primary"]:checked {
    color: #FFFFFF;
    background-color: #1E40AF;
    border-color: #1E40AF;
}

QPushButton[buttonRole="primary"]:hover {
    color: #FFFFFF;
    background-color: #1D4ED8;
    border-color: #1D4ED8;
}

QPushButton[buttonRole="primary"]:pressed {
    color: #FFFFFF;
    background-color: #1E3A8A;
    border-color: #1E3A8A;
}

QPushButton[buttonRole="primary"]:disabled {
    color: #94A3B8;
    background-color: #E2E8F0;
    border-color: #E2E8F0;
}

QPushButton[buttonRole="danger"] {
    color: #B91C1C;
    background-color: #FFFFFF;
    border-color: #FCA5A5;
}

QPushButton[buttonRole="danger"]:hover {
    color: #FFFFFF;
    background-color: #DC2626;
    border-color: #DC2626;
}

QPushButton[buttonRole="danger"]:pressed {
    color: #FFFFFF;
    background-color: #B91C1C;
    border-color: #B91C1C;
}

QPushButton[buttonRole="danger"]:disabled {
    color: #94A3B8;
    background-color: #F1F5F9;
    border-color: #E2E8F0;
}

/* Global pages use a flat top indicator; no tab-shaped blue cards. */
QPushButton[uiRole="navigationItem"][navigationLevel="main"] {
    min-width: 96px;
    min-height: 46px;
    padding: 0 18px;
    color: #475569;
    background-color: transparent;
    border: none;
    border-top: 2px solid transparent;
    border-radius: 0;
    font-size: 14px;
    font-weight: 500;
}

QPushButton[uiRole="navigationItem"][navigationLevel="main"]:hover {
    color: #1E40AF;
    background-color: #F8FAFC;
    border-top-color: #BFDBFE;
}

QPushButton[uiRole="navigationItem"][navigationLevel="main"]:checked {
    color: #1E40AF;
    background-color: #FFFFFF;
    border-top-color: #1E40AF;
    font-weight: 600;
}

QPushButton[uiRole="navigationItem"][navigationLevel="main"]:disabled {
    color: #94A3B8;
    background-color: transparent;
    border-top-color: transparent;
}

/* Report sub-pages use the same flat navigation language at a smaller scale. */
QPushButton[uiRole="navigationItem"][navigationLevel="internal"] {
    min-width: 88px;
    min-height: 34px;
    padding: 0 12px;
    color: #475569;
    background-color: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    border-radius: 0;
    font-size: 13px;
    font-weight: 500;
}

QPushButton[uiRole="navigationItem"][navigationLevel="internal"]:hover {
    color: #1E40AF;
    background-color: #F8FAFC;
    border-bottom-color: #BFDBFE;
}

QPushButton[uiRole="navigationItem"][navigationLevel="internal"]:checked {
    color: #1E40AF;
    background-color: transparent;
    border-bottom-color: #1E40AF;
    font-weight: 600;
}

QDockWidget#bottomDock {
    border: none;
}

QWidget#bottomDockPanel,
QWidget#bottomNavigation {
    background-color: #FFFFFF;
    border: none;
}

QWidget#bottomDockPanel {
    border-top: 1px solid #E2E8F0;
}

/* Overview: a flat project list and one muted summary rail. */
QFrame[uiRole="workspaceSection"] {
    background-color: #FFFFFF;
    border: none;
    border-right: 1px solid #E2E8F0;
    border-radius: 0;
}

QWidget#projectActivityHeader {
    background-color: #FFFFFF;
    border: none;
    border-bottom: 1px solid #E2E8F0;
}

QFrame[uiRole="workspaceAside"] {
    background-color: #F8FAFC;
    border: none;
    border-radius: 0;
}

QLabel#overviewWorkspaceNameLabel {
    color: #0F172A;
    background-color: transparent;
    font-size: 18px;
    font-weight: 600;
}

QLabel#overviewWorkspacePathLabel,
QLabel#overviewWorkspaceFileLabel,
QLabel[uiRole="metricTitle"] {
    color: #64748B;
    background-color: transparent;
    font-size: 12px;
    font-weight: 400;
}

QLabel[uiRole="summaryMetricValue"] {
    color: #0F172A;
    background-color: transparent;
    font-size: 22px;
    font-weight: 700;
}

QFrame#workspaceSummaryDivider {
    color: #E2E8F0;
    background-color: #E2E8F0;
    border: none;
}

QScrollArea#projectListScrollArea,
QScrollArea#projectListScrollArea > QWidget > QWidget,
QWidget#projectListContainer {
    background-color: transparent;
    border: none;
}

QWidget#projectRow {
    background-color: #FFFFFF;
    border: none;
    border-bottom: 1px solid #E2E8F0;
    border-radius: 0;
}

QWidget#projectRow:hover {
    background-color: #F8FAFC;
    border-bottom-color: #CBD5E1;
}

QWidget#projectInfo,
QWidget#projectActionPanel {
    background-color: transparent;
    border: none;
}

QLabel#projectNameLabel {
    color: #0F172A;
    background-color: transparent;
    font-size: 15px;
    font-weight: 600;
}

QLabel#projectPathLabel,
QLabel#projectMetaLabel {
    color: #64748B;
    background-color: transparent;
    font-size: 12px;
    font-weight: 400;
}

QWidget#projectActionPanel QPushButton {
    min-width: 0;
    min-height: 0;
    padding: 0 12px;
    font-size: 13px;
}

QLabel#emptyProjectLabel {
    min-height: 120px;
    color: #64748B;
    background-color: transparent;
    border: none;
    font-size: 14px;
    padding: 24px 16px;
}

/* Operation workspace keeps the three-panel engineering layout. */
QSplitter#operationPageSplitter {
    background-color: #FFFFFF;
    border: none;
}

QSplitter#operationPageSplitter::handle {
    background-color: #E2E8F0;
}

QFrame[uiRole="sidebar"] {
    background-color: #FFFFFF;
    border: none;
    border-radius: 0;
}

QFrame#leftDock {
    border-right: 1px solid #E2E8F0;
}

QFrame#rightDock {
    border-left: 1px solid #E2E8F0;
}

QWidget[uiRole="sidebarTitleBar"] {
    background-color: #F1F5F9;
    border: none;
    border-bottom: 1px solid #E2E8F0;
}

QWidget[uiRole="sidebarBody"] {
    background-color: #FFFFFF;
    border: none;
}

QToolButton#btn_collapse_left_sidebar,
QToolButton#btn_collapse_right_sidebar {
    color: #475569;
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    font-size: 14px;
}

QToolButton#btn_collapse_left_sidebar:hover,
QToolButton#btn_collapse_right_sidebar:hover {
    color: #1E40AF;
    background-color: #EFF6FF;
    border-color: #BFDBFE;
}

QToolButton#btn_expand_left_sidebar,
QToolButton#btn_expand_right_sidebar {
    color: #FFFFFF;
    background-color: #1E40AF;
    border: 1px solid #1E40AF;
    border-radius: 6px;
    font-size: 15px;
}

QToolButton#btn_expand_left_sidebar:hover,
QToolButton#btn_expand_right_sidebar:hover {
    background-color: #1D4ED8;
    border-color: #1D4ED8;
}

QWidget#viewportPanel {
    background-color: #FFFFFF;
    border: none;
}

QLabel#viewportStateLabel {
    color: #475569;
    background-color: #F1F5F9;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 12px;
}

QWidget#open3dViewport {
    background-color: #111827;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
}

/* Report workspace: stable page title, flat sub-navigation and one viewer. */
QWidget#reportNavigation,
QWidget#reportDocumentHeader {
    background-color: transparent;
    border: none;
}

QLabel#reportPdfStatusLabel {
    color: #64748B;
    background-color: transparent;
    font-size: 12px;
}

QLabel#reportPdfStatusLabel[statusState="loading"] {
    color: #1E40AF;
}

QLabel#reportPdfStatusLabel[statusState="success"] {
    color: #047857;
}

QLabel#reportPdfStatusLabel[statusState="error"] {
    color: #B91C1C;
}

QStackedWidget#reportNavigationStack,
QStackedWidget#reportPreviewStateStack,
QWidget#reportPreviewPage,
QWidget#reportHeatmapPage,
QWidget#reportDocumentPage {
    background-color: #FFFFFF;
    border: none;
}

QWidget#reportEmptyState,
QLabel#heatmapPlaceholder {
    color: #64748B;
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
}

QWidget#reportPdfWebView {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
}

QLineEdit,
QComboBox,
QSpinBox,
QDoubleSpinBox,
QPlainTextEdit,
QTextEdit {
    min-height: 22px;
    padding: 6px 10px;
    color: #0F172A;
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    selection-color: #FFFFFF;
    selection-background-color: #1E40AF;
}

QLineEdit:focus,
QComboBox:focus,
QSpinBox:focus,
QDoubleSpinBox:focus,
QPlainTextEdit:focus,
QTextEdit:focus {
    border-color: #3B82F6;
}

QListView,
QTreeView,
QTableView {
    color: #0F172A;
    background-color: #FFFFFF;
    alternate-background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    selection-color: #FFFFFF;
    selection-background-color: #1E40AF;
}

QHeaderView::section {
    color: #334155;
    background-color: #F1F5F9;
    border: none;
    border-bottom: 1px solid #E2E8F0;
    padding: 8px;
    font-weight: 600;
}

QScrollBar:vertical {
    width: 10px;
    background: transparent;
    margin: 2px;
}

QScrollBar::handle:vertical {
    min-height: 28px;
    background: #CBD5E1;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #94A3B8;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {
    height: 0;
    background: transparent;
}

QMenu {
    color: #0F172A;
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    padding: 5px;
}

QMenu::item {
    padding: 7px 24px 7px 10px;
    border-radius: 5px;
}

QMenu::item:selected {
    color: #1E40AF;
    background-color: #EFF6FF;
}

QToolTip {
    color: #FFFFFF;
    background-color: #0F172A;
    border: 1px solid #334155;
    padding: 5px 8px;
}
"""


def apply_application_theme(app: QApplication) -> None:
    """Apply one sans-serif font stack, palette and QSS to every Qt window."""
    app.setStyle("Fusion")

    available_families = {
        family.casefold(): family
        for family in QFontDatabase.families()
    }
    requested_families = (
        "Segoe UI Variable Text",
        "Segoe UI Variable",
        "Microsoft YaHei UI",
        "DengXian",
        "等线",
        "Segoe UI",
    )
    font_families = [
        available_families[family.casefold()]
        for family in requested_families
        if family.casefold() in available_families
    ]
    font = QFont(font_families[0] if font_families else "Sans Serif", 10)
    if font_families:
        font.setFamilies(font_families)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(COLORS["background"]))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(COLORS["text"]))
    palette.setColor(QPalette.ColorRole.Base, QColor(COLORS["surface"]))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(COLORS["background"]))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(COLORS["text"]))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(COLORS["surface"]))
    palette.setColor(QPalette.ColorRole.Text, QColor(COLORS["text"]))
    palette.setColor(QPalette.ColorRole.Button, QColor(COLORS["surface"]))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(COLORS["text"]))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(COLORS["primary"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(COLORS["surface"]))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(COLORS["text_muted"]))
    app.setPalette(palette)
    app.setStyleSheet(APPLICATION_STYLE_SHEET)
