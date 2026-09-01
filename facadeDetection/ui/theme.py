"""Corporate Clean theme shared by every PySide6 workspace page.

The visual hierarchy follows the reference site's functional-first rules:
one white workspace surface, thin borders, restrained blue anchors and no
decorative gradients or nested-card shadows.
"""

from pathlib import Path

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication


# QPalette uses these tokens directly; the QSS below mirrors the same values
# so business widgets never carry page-specific color declarations.
COLORS = {
    "primary": "#1E40AF",
    "primary_hover": "#1D4ED8",
    "accent": "#3B82F6",
    "accent_soft": "#EFF6FF",
    "background": "#EEF2F7",
    "surface": "#FFFFFF",
    "surface_muted": "#F5F7FB",
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
    background-color: #FFFFFF;
}

QWidget#applicationHeader {
    background-color: #0B1220;
    border: none;
    border-bottom: 1px solid #243044;
}

QLabel#applicationBrandMark {
    color: #FFFFFF;
    background-color: #2F6BFF;
    border: 1px solid #4B7DFF;
    border-radius: 10px;
    font-size: 14px;
    font-weight: 700;
}

QLabel#applicationPageTitle {
    color: #F8FAFC;
    background-color: transparent;
    font-size: 22px;
    font-weight: 700;
}

QLabel#currentProjectLabel {
    color: #DCE6F5;
    background-color: #111C2E;
    border: 1px solid #2B3A52;
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 13px;
    font-weight: 500;
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
    font-size: 13px;
    font-weight: 400;
}

QFrame[uiRole="workspaceSurface"] {
    background-color: #FFFFFF;
    border: none;
    border-radius: 0;
}

QWidget[uiRole="workspaceBody"],
QWidget#overviewColumns {
    background-color: #FFFFFF;
    border: none;
    border-radius: 0;
}

QWidget[uiRole="pageHeader"] {
    background-color: #F6F8FB;
    border: none;
    border-bottom: 1px solid #DDE5EF;
    border-radius: 0;
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
    font-size: 13px;
    font-weight: 500;
}

QPushButton {
    min-height: 20px;
    padding: 7px 14px;
    color: #27364A;
    background-color: #FFFFFF;
    border: 1px solid #C8D3E1;
    border-radius: 7px;
    font-size: 14px;
    font-weight: 500;
}

QPushButton:hover {
    color: #1E40AF;
    background-color: #F2F6FF;
    border-color: #8EABF6;
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
    color: #FFFFFF;
    background-color: #2457D6;
    border: 1px solid #2457D6;
    min-height: 20px;
    padding: 7px 14px;
}

/* 顶部命令栏是高频操作区，概览页与项目操作页统一使用主蓝色。 */
QPushButton[uiRole="headerAction"]:hover {
    color: #FFFFFF;
    background-color: #1E48B8;
    border-color: #1E48B8;
}

QPushButton[uiRole="headerAction"]:pressed {
    color: #FFFFFF;
    background-color: #1E3A8A;
    border-color: #1E3A8A;
}

QPushButton[uiRole="headerAction"]:focus {
    color: #FFFFFF;
    border-color: #93C5FD;
}

QPushButton[uiRole="headerAction"]:disabled {
    color: #94A3B8;
    background-color: #E2E8F0;
    border-color: #E2E8F0;
}

QToolButton[uiRole="sidebarToggle"] {
    color: #FFFFFF;
    background-color: #2457D6;
    border: 1px solid #2457D6;
    border-radius: 7px;
    font-size: 16px;
    font-weight: 700;
}

QToolButton[uiRole="sidebarToggle"]:hover {
    background-color: #1E48B8;
    border-color: #1E48B8;
}

QToolButton[uiRole="sidebarToggle"]:pressed,
QToolButton[uiRole="sidebarToggle"]:checked {
    background-color: #1E3A8A;
    border-color: #1E3A8A;
}

QToolButton[uiRole="windowControl"] {
    color: #E5EDF8;
    background-color: transparent;
    border: none;
    border-radius: 0;
    font-family: "Segoe UI Symbol", "Microsoft YaHei UI";
    font-size: 20px;
    font-weight: 600;
}

QToolButton[uiRole="windowControl"]:hover {
    color: #FFFFFF;
    background-color: #243044;
}

QToolButton[uiRole="windowControl"]:pressed {
    background-color: #334155;
}

QToolButton[uiRole="windowControl"][windowAction="close"]:hover {
    color: #FFFFFF;
    background-color: #C42B1C;
}

QToolButton[uiRole="windowControl"][windowAction="close"]:pressed {
    background-color: #A91F14;
}

QPushButton[buttonRole="primary"],
QPushButton[buttonRole="primary"]:checked {
    color: #FFFFFF;
    background-color: #2457D6;
    border-color: #2457D6;
}

QPushButton[buttonRole="primary"]:hover {
    color: #FFFFFF;
    background-color: #1E48B8;
    border-color: #1E48B8;
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

/* 低频工具动作使用雾蓝色调，和主蓝按钮形成清晰但统一的层级。 */
QPushButton[buttonRole="tonal"] {
    color: #1E40AF;
    background-color: #E7EEFF;
    border-color: #BFD0FF;
    font-weight: 600;
}

QPushButton[buttonRole="tonal"]:hover {
    color: #173A9A;
    background-color: #D9E5FF;
    border-color: #91ABF8;
}

QPushButton[buttonRole="tonal"]:pressed {
    color: #FFFFFF;
    background-color: #2457D6;
    border-color: #2457D6;
}

QPushButton[buttonRole="tonal"]:disabled {
    color: #94A3B8;
    background-color: #E8EDF3;
    border-color: #D4DDE8;
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

/* Global navigation is a compact command dock instead of four full-screen tabs. */
QPushButton[uiRole="navigationItem"][navigationLevel="main"] {
    min-width: 96px;
    min-height: 42px;
    padding: 0 18px;
    color: #475569;
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 7px;
    font-size: 14px;
    font-weight: 500;
}

QPushButton[uiRole="navigationItem"][navigationLevel="main"]:hover {
    color: #1E40AF;
    background-color: #F2F6FF;
    border-color: #D5E0FF;
}

QPushButton[uiRole="navigationItem"][navigationLevel="main"]:checked {
    color: #FFFFFF;
    background-color: #1E40AF;
    border-color: #1E40AF;
    font-weight: 600;
}

QPushButton[uiRole="navigationItem"][navigationLevel="main"]:disabled {
    color: #94A3B8;
    background-color: transparent;
    border-color: transparent;
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
    border: none;
}

QWidget#bottomDockPanel {
    background-color: #E9EEF5;
    border-top: 1px solid #D2DBE7;
}

QWidget#bottomNavigation {
    background-color: #F1F5FA;
    border: 1px solid #C6D2E1;
    border-radius: 11px;
}

QGroupBox {
    color: #334155;
    background-color: #F1F6FB;
    border: 1px solid #CFDAE8;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 8px;
    font-size: 13px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
    color: #334155;
    background-color: #F1F6FB;
}

QSlider::groove:horizontal {
    height: 4px;
    background-color: #CBD7E6;
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background-color: #2F6BFF;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    width: 14px;
    height: 14px;
    margin: -5px 0;
    background-color: #FFFFFF;
    border: 2px solid #2F6BFF;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background-color: #E7EEFF;
    border-color: #1E48B8;
}

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
    background-color: #F5F8FC;
    border: none;
    border-radius: 0;
}

QFrame[uiRole="accentLine"] {
    background-color: #2F6BFF;
    border: none;
    border-radius: 1px;
}

QLabel#overviewWorkspaceNameLabel {
    color: #0F172A;
    background-color: transparent;
    font-size: 18px;
    font-weight: 600;
}

QLabel#overviewWorkspacePathLabel,
QLabel#overviewWorkspaceFileLabel {
    color: #64748B;
    background-color: transparent;
    font-size: 12px;
    font-weight: 400;
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
    background-color: #F4F7FC;
    border-bottom-color: #BFCBDC;
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

QLabel#projectMarkerLabel {
    color: #1E40AF;
    background-color: #E7EEFF;
    border: 1px solid #C9D8FF;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 700;
}

QLabel#projectPathLabel,
QLabel#projectMetaLabel {
    color: #64748B;
    background-color: transparent;
    font-size: 12px;
    font-weight: 400;
}

QWidget#projectEmptyState,
QWidget#reviewTechnicalCanvas,
QWidget#reportEmptyState,
QWidget#heatmapPlaceholder {
    background-color: #FBFCFE;
    border: none;
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
    background-color: #F7F9FC;
    border: none;
    border-radius: 0;
}

QFrame#leftDock {
    border-right: 1px solid #E2E8F0;
}

QFrame#rightDock {
    border-left: 1px solid #E2E8F0;
}

QWidget[uiRole="sidebarBody"] {
    background-color: #F7F9FC;
    border: none;
}

QWidget#viewportPanel {
    background-color: #FFFFFF;
    border: none;
}

QFrame#inspectionConfigBar {
    background-color: #EDF3FA;
    border: none;
    border-top: 1px solid #C8D5E4;
    border-bottom: 1px solid #C8D5E4;
}

QFrame#inspectionConfigBar QLabel {
    color: #475569;
    background-color: transparent;
}

QLabel[uiRole="inspectionFieldLabel"] {
    color: #334155;
    font-size: 13px;
    font-weight: 600;
}

QComboBox#standardProfileCombo,
QComboBox#inspectionIntervalCombo {
    color: #173A7A;
    background-color: #E7EEFF;
    border: 1px solid #B3C6E6;
    border-radius: 7px;
    padding-left: 12px;
    font-weight: 600;
}

QComboBox#standardProfileCombo:hover,
QComboBox#inspectionIntervalCombo:hover {
    background-color: #DCE7FF;
    border-color: #86A4DF;
}

QComboBox#standardProfileCombo:focus,
QComboBox#inspectionIntervalCombo:focus {
    background-color: #F6F9FF;
    border-color: #2F6BFF;
}

QLabel#standardSummary {
    color: #475569;
    background-color: #F7FAFD;
    border: 1px solid #D3DEEA;
    border-radius: 7px;
    padding: 7px 10px;
    font-size: 12px;
}

QLabel#viewportStateLabel {
    color: #1E40AF;
    background-color: #E7EEFF;
    border: 1px solid #BFD0FF;
    border-radius: 7px;
    padding: 4px 10px;
    font-size: 13px;
    font-weight: 600;
}

QLabel#viewportStateLabel[statusState="loading"] {
    color: #1E40AF;
    background-color: #E7EEFF;
    border-color: #9FB8FF;
}

QWidget#open3dViewport {
    background-color: #111827;
    border: 1px solid #B7C4D5;
    border-radius: 0;
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

QWidget#reportPdfWebView {
    background-color: #FFFFFF;
    border: 1px solid #D5DEEA;
    border-radius: 0;
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
    background-color: #EEF4FA;
    border: 1px solid #B8C7DA;
    border-radius: 6px;
    selection-color: #FFFFFF;
    selection-background-color: #1E40AF;
}

QLineEdit:hover,
QComboBox:hover,
QSpinBox:hover,
QDoubleSpinBox:hover,
QPlainTextEdit:hover,
QTextEdit:hover {
    background-color: #F4F8FC;
    border-color: #8FA8C8;
}

QLineEdit:focus,
QComboBox:focus,
QSpinBox:focus,
QDoubleSpinBox:focus,
QPlainTextEdit:focus,
QTextEdit:focus {
    background-color: #F8FBFF;
    border-color: #3B82F6;
}

QLineEdit:disabled,
QComboBox:disabled,
QSpinBox:disabled,
QDoubleSpinBox:disabled,
QPlainTextEdit:disabled,
QTextEdit:disabled {
    color: #94A3B8;
    background-color: #E8EDF3;
    border-color: #D4DDE8;
}

/* 下拉框与数字输入统一成一体式雾蓝控件，替换不稳定的系统箭头。 */
QComboBox {
    padding-right: 36px;
}

QSpinBox,
QDoubleSpinBox {
    padding-right: 34px;
}

QComboBox::drop-down {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 30px;
    background-color: #E1EAF5;
    border: none;
    border-left: 1px solid #B8C7DA;
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
}

QComboBox::drop-down:hover {
    background-color: #D6E3F3;
    border-left-color: #8FA8C8;
}

QComboBox::drop-down:on {
    background-color: #C9D9EE;
    border-left-color: #6F8FB9;
}

QComboBox::down-arrow {
    image: url(__CHEVRON_DOWN__);
    width: 12px;
    height: 8px;
}

QSpinBox::up-button,
QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 28px;
    background-color: #E1EAF5;
    border: none;
    border-left: 1px solid #B8C7DA;
    border-bottom: 1px solid #C6D3E3;
    border-top-right-radius: 6px;
}

QSpinBox::down-button,
QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 28px;
    background-color: #E1EAF5;
    border: none;
    border-left: 1px solid #B8C7DA;
    border-bottom-right-radius: 6px;
}

QSpinBox::up-button:hover,
QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover,
QDoubleSpinBox::down-button:hover {
    background-color: #D6E3F3;
    border-left-color: #8FA8C8;
}

QSpinBox::up-button:pressed,
QDoubleSpinBox::up-button:pressed,
QSpinBox::down-button:pressed,
QDoubleSpinBox::down-button:pressed {
    background-color: #C9D9EE;
}

QSpinBox::up-arrow,
QDoubleSpinBox::up-arrow {
    image: url(__CHEVRON_UP__);
    width: 10px;
    height: 7px;
}

QSpinBox::down-arrow,
QDoubleSpinBox::down-arrow {
    image: url(__CHEVRON_DOWN__);
    width: 10px;
    height: 7px;
}

QComboBox QAbstractItemView {
    color: #0F172A;
    background-color: #F8FBFF;
    border: 1px solid #AFC0D6;
    selection-color: #FFFFFF;
    selection-background-color: #2457D6;
    outline: 0;
}

QCheckBox {
    color: #334155;
    spacing: 7px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    background-color: #EEF4FA;
    border: 1px solid #9FB0C7;
    border-radius: 4px;
}

QCheckBox::indicator:hover {
    border-color: #3B82F6;
}

QCheckBox::indicator:checked {
    background-color: #2457D6;
    border-color: #2457D6;
}

QFrame#qualityParameterPanel {
    background-color: #F1F6FB;
    border: 1px solid #CFDAE8;
    border-radius: 8px;
}

/* 紧凑参数输入保留足够的数字区，不让右侧增减按钮压住文本。 */
QSpinBox[uiRole="qualityParameterInput"],
QDoubleSpinBox[uiRole="qualityParameterInput"],
QComboBox[uiRole="qualityParameterInput"] {
    min-height: 32px;
    color: #173A7A;
    background-color: #F8FBFF;
    border-color: #B3C6E6;
    font-size: 12px;
    font-weight: 600;
}

QDialog#projectCreateDialog {
    background-color: #E9EEF5;
}

QFrame#projectDialogHero {
    background-color: #0F1B2D;
    border: 1px solid #2A3A52;
    border-radius: 10px;
}

QLabel#projectDialogTitle {
    color: #F8FAFC;
    background-color: transparent;
    font-size: 20px;
    font-weight: 700;
}

QLabel#projectDialogSubtitle {
    color: #B9C7DA;
    background-color: transparent;
    font-size: 12px;
}

QFrame#projectFormPanel {
    background-color: #F7FAFD;
    border: 1px solid #CBD7E6;
    border-radius: 10px;
}

QDialog#projectCreateDialog QFrame#projectFormPanel QLabel {
    color: #334155;
    background-color: transparent;
    font-size: 13px;
    font-weight: 600;
}

QDialog#projectCreateDialog QLineEdit,
QDialog#projectCreateDialog QComboBox,
QDialog#projectCreateDialog QTextEdit {
    min-height: 26px;
    color: #172033;
    background-color: #EEF4FA;
    border-color: #B8C7DA;
    font-size: 13px;
}

QWidget#projectRegionSelector QComboBox {
    min-height: 30px;
    padding-left: 10px;
    color: #173A7A;
    background-color: #EEF4FA;
    border-color: #B8C7DA;
    font-weight: 600;
}

QDialog#projectCreateDialog QLineEdit:hover,
QDialog#projectCreateDialog QComboBox:hover,
QDialog#projectCreateDialog QTextEdit:hover {
    background-color: #F8FBFF;
    border-color: #8FA8C8;
}

QDialog#projectCreateDialog QLineEdit:focus,
QDialog#projectCreateDialog QComboBox:focus,
QDialog#projectCreateDialog QTextEdit:focus {
    background-color: #FFFFFF;
    border-color: #2F6BFF;
}

QDialogButtonBox#projectDialogButtons {
    background-color: transparent;
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


def _theme_style_sheet() -> str:
    """Resolve reusable SVG controls to stable absolute QSS paths."""
    icon_dir = Path(__file__).resolve().parent / "assets" / "icons"
    return (
        APPLICATION_STYLE_SHEET
        .replace("__CHEVRON_DOWN__", f'"{(icon_dir / "chevron_down.svg").as_posix()}"')
        .replace("__CHEVRON_UP__", f'"{(icon_dir / "chevron_up.svg").as_posix()}"')
    )


def apply_application_theme(app: QApplication) -> None:
    """Apply one sans-serif font stack, palette and QSS to every Qt window."""
    app.setStyle("Fusion")

    available_families = {
        family.casefold(): family
        for family in QFontDatabase.families()
    }
    requested_families = (
        "DengXian",
        "等线",
        "Microsoft YaHei UI",
        "Bahnschrift",
        "Segoe UI Variable Text",
        "Segoe UI Variable",
        "Segoe UI",
    )
    font_families = [
        available_families[family.casefold()]
        for family in requested_families
        if family.casefold() in available_families
    ]
    # 11pt 在老师演示用的 1920×1080 屏幕上更易读，Qt 仍会随系统缩放自适应。
    font = QFont(font_families[0] if font_families else "Sans Serif", 11)
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
    app.setStyleSheet(_theme_style_sheet())
