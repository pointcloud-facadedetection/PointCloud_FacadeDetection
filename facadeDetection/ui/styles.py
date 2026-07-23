"""Visual tokens and widget helpers for the desktop prototype shell."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton, QWidget


PLACEHOLDER = ''

COLORS = {
    'navy': '#13294B',
    'navy_dark': '#0E2038',
    'blue': '#1D4E89',
    'blue_soft': '#EDF2F9',
    'orange': '#E8823A',
    'green': '#1E9E5A',
    'red': '#D64545',
    'ink': '#1F2937',
    'muted': '#5B6472',
    'subtle': '#8A93A0',
    'line': '#D8DDE4',
    'surface': '#FFFFFF',
    'canvas': '#F2F4F7',
}


APP_STYLESHEET = """
QMainWindow, QWidget#appRoot {
    background: #F2F4F7;
    color: #1F2937;
}
QWidget {
    font-family: "Microsoft YaHei UI";
    font-size: 13px;
}
QFrame[role="topbar"] {
    background: #13294B;
    border: none;
}
QFrame[role="card"], QFrame[role="panel"] {
    background: #FFFFFF;
    border: 1px solid #E1E5EA;
    border-radius: 10px;
}
QFrame[role="dropzone"] {
    background: #FFFFFF;
    border: 2px dashed #C7CCD3;
    border-radius: 8px;
}
QFrame[role="soft"] {
    background: #F7F9FB;
    border: 1px solid #E8EAED;
    border-radius: 8px;
}
QFrame#navRail {
    background: #0E2038;
    border: none;
}
QFrame#reportToc {
    background: #FFFFFF;
    border-right: 1px solid #D8DDE4;
}
QFrame#rightInspector, QFrame#reportEditor {
    background: #FFFFFF;
    border-left: 1px solid #D8DDE4;
}
QLabel[role="pageTitle"] {
    color: #13294B;
    font-size: 18px;
    font-weight: 700;
}
QLabel[role="sectionTitle"] {
    color: #13294B;
    font-size: 15px;
    font-weight: 700;
}
QLabel[role="cardTitle"] {
    color: #13294B;
    font-size: 16px;
    font-weight: 700;
}
QLabel[role="muted"] {
    color: #8A93A0;
    font-size: 12px;
}
QLabel[role="placeholder"] {
    color: #3D4756;
    font-weight: 600;
}
QLabel[role="statusPill"] {
    color: #E8823A;
    background: #FCEBDD;
    border-radius: 12px;
    padding: 4px 10px;
    font-size: 12px;
    font-weight: 600;
}
QLabel[role="devicePill"] {
    color: #DCE3EC;
    background: rgba(255, 255, 255, 0.09);
    border-radius: 15px;
    padding: 6px 13px;
    font-size: 12px;
}
QPushButton {
    min-height: 38px;
    padding: 0 14px;
    border-radius: 6px;
    border: 1px solid #C7CCD3;
    background: #FFFFFF;
    color: #3D4756;
    font-weight: 600;
}
QPushButton:hover {
    border-color: #8FA0B8;
    background: #F7F9FB;
}
QPushButton:pressed {
    background: #E9EDF2;
}
QPushButton[variant="primary"] {
    background: #13294B;
    border-color: #13294B;
    color: #FFFFFF;
}
QPushButton[variant="primary"]:hover {
    background: #1D4E89;
    border-color: #1D4E89;
}
QPushButton[variant="accent"] {
    background: #E8823A;
    border-color: #E8823A;
    color: #FFFFFF;
}
QPushButton[variant="accent"]:hover {
    background: #D8742E;
    border-color: #D8742E;
}
QPushButton[variant="topGhost"] {
    background: transparent;
    border-color: rgba(255, 255, 255, 0.35);
    color: #FFFFFF;
}
QPushButton[variant="topGhost"]:hover {
    background: rgba(255, 255, 255, 0.10);
}
QPushButton[variant="danger"] {
    background: #FFFFFF;
    border-color: #F0D3D3;
    color: #C24545;
}
QPushButton[variant="nav"] {
    min-width: 62px;
    max-width: 62px;
    min-height: 58px;
    padding: 4px 2px;
    border: none;
    border-radius: 6px;
    background: transparent;
    color: #AEB9C6;
    font-size: 12px;
}
QPushButton[variant="nav"][active="true"] {
    background: #1D4E89;
    color: #FFFFFF;
}
QPushButton[variant="nav"]:hover {
    background: rgba(255, 255, 255, 0.08);
    color: #FFFFFF;
}
QPushButton[variant="toc"] {
    min-height: 38px;
    padding: 0 12px;
    border: none;
    border-radius: 5px;
    background: transparent;
    color: #3D4756;
    text-align: left;
    font-weight: 500;
}
QPushButton[variant="toc"][active="true"] {
    background: #13294B;
    color: #FFFFFF;
    font-weight: 700;
}
QPushButton[variant="chip"] {
    min-height: 32px;
    padding: 0 12px;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.10);
    border-color: rgba(255, 255, 255, 0.18);
    color: #C7D2E0;
}
QPushButton[variant="chip"][active="true"] {
    background: #1D4E89;
    border-color: #1D4E89;
    color: #FFFFFF;
}
QLineEdit, QTextEdit, QComboBox {
    min-height: 38px;
    padding: 0 10px;
    border: 1px solid #C7CCD3;
    border-radius: 6px;
    background: #FFFFFF;
    color: #1F2937;
    selection-background-color: #1D4E89;
}
QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
    border-color: #1D4E89;
}
QScrollArea {
    border: none;
    background: transparent;
}
QScrollBar:vertical {
    width: 10px;
    background: transparent;
}
QScrollBar::handle:vertical {
    min-height: 30px;
    border-radius: 5px;
    background: #C7CCD3;
}
QTableWidget {
    background: #FFFFFF;
    alternate-background-color: #F7F9FB;
    border: 1px solid #E1E5EA;
    border-radius: 8px;
    gridline-color: #E8EAED;
    color: #3D4756;
}
QHeaderView::section {
    min-height: 38px;
    padding: 0 8px;
    background: #EDF2F9;
    border: none;
    border-right: 1px solid #D8DDE4;
    border-bottom: 1px solid #D8DDE4;
    color: #13294B;
    font-weight: 700;
}
QDialog {
    background: #F2F4F7;
}
"""


def make_button(
    text: str,
    object_name: str,
    variant: str = 'secondary',
    *,
    height: int = 40,
) -> QPushButton:
    button = QPushButton(text)
    button.setObjectName(object_name)
    button.setProperty('variant', variant)
    button.setFixedHeight(height)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    return button


def set_active(widget: QWidget, active: bool) -> None:
    widget.setProperty('active', 'true' if active else 'false')
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()
