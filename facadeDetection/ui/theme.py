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
    font-size: 17px;
    font-weight: 500;
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
    background-color: #FFFFFF;
    border: 1px solid #D4DDE9;
    border-radius: 11px;
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

/* 三维视口（外层容器）：主工作区"画框"。
   白底 + 1px 描边 + 10px 圆角把白色页面与深色视口隔开。
   注意：内嵌的 Open3D 原生 GLFW surface 本身是直角矩形，因此外层圆角
   必须由 8px 的布局留白（operation_page 的 viewport_layout）喂出来，
   而不是靠面板自身裁切；内层再由 open3dViewport 的描边与圆角收边，
   四角才不会露出灰缝。 */
QWidget#viewportPanel {
    /* 视口面板是三维视图的卡片外框：只在四角做圆角、四周描一圈细边，
       面板本身用中性底色，让内部的深色 Open3D 视图成为唯一视觉主体。
       配合 operation_page 中 8px 的内容边距，卡片与左右侧栏之间始终留白，
       圆角不会被相邻面板裁掉。 */
    background-color: #FFFFFF;
    border: 1px solid #DCE3EC;
    border-radius: 10px;
}

QLabel#viewportStateLabel {
    color: #34445A;
    background-color: #EEF2F7;
    border: 1px solid #D5DEEA;
    border-radius: 7px;
    padding: 4px 8px;
    font-size: 13px;
}

QWidget#open3dViewport {
    /* 深色画布与面板圆角保持同一个视觉半径；描边取比底色略亮的冷灰，
       在浅色工作台上形成清晰的画布边界，不再是"深块直接切边"。 */
    background-color: #111827;
    border: 1px solid #C3CFDF;
    border-radius: 9px;
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

/* ------------------------------------------------------------------
   次级按钮：介于主蓝与危险红之间，用于取消、更多、辅助操作。
   进度窗"取消"与顶部"更多"菜单按钮都复用该角色，保证视觉一致。
   ------------------------------------------------------------------ */
QPushButton[buttonRole="secondary"] {
    color: #1E40AF;
    background-color: #FFFFFF;
    border: 1px solid #BFD3F5;
    border-radius: 6px;
    padding: 6px 16px;
    min-height: 20px;
    font-weight: 600;
}

QPushButton[buttonRole="secondary"]:hover {
    color: #1D4ED8;
    background-color: #EFF6FF;
    border-color: #93B4F0;
}

QPushButton[buttonRole="secondary"]:pressed {
    color: #1E3A8A;
    background-color: #DBEAFE;
    border-color: #7BA4EC;
}

QPushButton[buttonRole="secondary"]:disabled {
    color: #94A3B8;
    background-color: #F8FAFC;
    border-color: #E2E8F0;
}

/* ------------------------------------------------------------------
   统一进度窗（TaskProgressDialog / LoadingDialog）
   ------------------------------------------------------------------ */
QDialog QLabel#taskProgressTitle {
    color: #0F172A;
    font-size: 14px;
    font-weight: 700;
}

QDialog QLabel#taskProgressMessage {
    color: #334155;
    font-size: 12px;
}

QDialog QLabel#taskProgressStatus {
    color: #64748B;
    font-size: 11px;
}

QProgressBar {
    color: #0F172A;
    background-color: #E8EDF5;
    border: 1px solid #D8E0EC;
    border-radius: 6px;
    min-height: 18px;
    text-align: center;
    font-size: 11px;
    font-weight: 600;
}

QProgressBar::chunk {
    background-color: #2F6BFF;
    border-radius: 5px;
    margin: 1px;
}

/* 不定量模式（setRange(0,0)）下 Qt 会绘制忙碌条纹，隐藏文字避免重叠 */
QProgressBar[busy="true"] {
    text-align: center;
}

/* ------------------------------------------------------------------
   顶部"更多"下拉菜单按钮：与步骤栏同高，视觉权重低于主按钮。
   ------------------------------------------------------------------ */
QToolButton[uiRole="headerMore"] {
    color: #1E40AF;
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: 600;
}

QToolButton[uiRole="headerMore"]:hover {
    color: #1D4ED8;
    background-color: #EFF6FF;
    border-color: #93B4F0;
}

QToolButton[uiRole="headerMore"]:pressed,
QToolButton[uiRole="headerMore"]:checked {
    color: #1E3A8A;
    background-color: #DBEAFE;
    border-color: #7BA4EC;
}

QToolButton[uiRole="headerMore"]:disabled {
    color: #94A3B8;
    background-color: #F8FAFC;
    border-color: #E2E8F0;
}

QToolButton[uiRole="headerMore"]::menu-indicator {
    image: none;
    width: 0px;
}

/* ------------------------------------------------------------------
   顶部步骤导航栏（StepNavBar）
   四步业务漏斗：数据处理 → 区域选取 → 立面提取 → 质量评估
   视觉状态由动态属性 [stepState] / [badgeState] 驱动，组件本身不感知业务。
   ------------------------------------------------------------------ */
QWidget#stepNavBar {
    background-color: transparent;
    border: none;
}

QFrame#stepChip {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    min-height: 44px;
}

QFrame#stepChip:hover {
    background-color: #F5F7FB;
    border-color: #CBD5E1;
}

QFrame#stepChip[stepState="active"] {
    background-color: #EFF6FF;
    border-color: #3B82F6;
}

/* 已完成步骤：淡蓝底 + 蓝色描边，与主色体系保持同一层级 */
QFrame#stepChip[stepState="done"] {
    background-color: #F8FAFF;
    border-color: #BFDBFE;
}

QLabel#stepChipTitle {
    color: #0F172A;
    font-size: 15px;
    font-weight: 600;
    background-color: transparent;
}

QFrame#stepChip[stepState="active"] QLabel#stepChipTitle {
    color: #1E40AF;
}

QFrame#stepChip[stepState="done"] QLabel#stepChipTitle {
    color: #334155;
}

QLabel#stepBadge {
    color: #FFFFFF;
    background-color: #94A3B8;
    border: none;
    border-radius: 17px;
    font-size: 15px;
    font-weight: 700;
}

QLabel#stepBadge[badgeState="active"] {
    background-color: #1E40AF;
}

QLabel#stepBadge[badgeState="done"] {
    background-color: #2F6BFF;
    font-size: 17px;
}

/* 步骤失败：徽标 ✗ 用醒目的玫红，与"完成"的蓝、进行中的深蓝明确区分。 */
QLabel#stepBadge[badgeState="failed"] {
    background-color: #E11D48;
    font-size: 17px;
}

QFrame#stepChip[stepState="failed"] {
    border: 1px solid #FDA4AF;
    background-color: #FFF1F2;
}

QFrame#stepChip[stepState="failed"] QLabel#stepChipTitle {
    color: #9F1239;
}

/* 三步等宽均分：卡片最小宽度保证窄窗口下文字不被裁切。 */
QFrame#stepChip {
    min-width: 132px;
}

/* ------------------------------------------------------------------
   ROI 确认悬浮卡片
   ------------------------------------------------------------------ */
QFrame#roiConfirmCard {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 8px;
    padding: 4px;
}

/* 项目操作页占位提示 */
QWidget#operationPlaceholder {
    background-color: #F8FAFC;
    border: 2px dashed #CBD5E1;
    border-radius: 12px;
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
    # 表单样式单列在 form_styles 中，此处与其拼接成一份完整 QSS，
    # 保证全局只调用一次 setStyleSheet（重复设置会让已有样式表失效）。
    from .form_styles import FORM_STYLE_SHEET

    app.setStyleSheet(APPLICATION_STYLE_SHEET + FORM_STYLE_SHEET)
