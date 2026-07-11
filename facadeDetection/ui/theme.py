"""Application-wide visual theme.

The palette mirrors the first-version HTML prototype that ships with the
repository, while keeping the desktop implementation native to PySide6.
"""

APP_STYLESHEET = r"""
QMainWindow, QWidget {
    background-color: #0a0a1a;
    color: #e0e0ff;
    font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
    font-size: 12px;
}
QLabel { background-color: transparent; }

QMenuBar {
    background-color: #101025;
    border-bottom: 1px solid #29294c;
    padding: 3px 8px;
}
QMenuBar::item {
    background: transparent;
    padding: 6px 12px;
    border-radius: 5px;
}
QMenuBar::item:selected { background-color: #27274a; }
QMenu {
    background-color: #171731;
    border: 1px solid #30305c;
    padding: 5px;
}
QMenu::item { padding: 7px 28px 7px 12px; border-radius: 5px; }
QMenu::item:selected { background-color: #667eea; color: white; }

QScrollArea { border: none; background: transparent; }
QScrollBar:vertical {
    background: #12122a;
    width: 8px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #34345d;
    min-height: 28px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover { background: #667eea; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QFrame#controlPanel {
    background-color: #12122a;
    border-right: 1px solid #29294c;
}
QFrame#sidebarHeader {
    background-color: #171733;
    border-bottom: 1px solid #29294c;
}
QLabel#appTitle {
    color: #8fa2ff;
    font-size: 18px;
    font-weight: 700;
}
QLabel#appSubtitle, QLabel#mutedLabel { color: #8892b0; }
QLabel#sectionTitle {
    color: #8195ff;
    font-size: 12px;
    font-weight: 700;
}
QFrame#sectionCard {
    background-color: #17172f;
    border: 1px solid #29294c;
    border-radius: 12px;
}
QFrame#dropZone {
    background-color: #15152e;
    border: 2px dashed #353568;
    border-radius: 10px;
}
QFrame#dropZone[dragActive="true"] {
    background-color: #1d2148;
    border-color: #667eea;
}
QLabel#dropIcon { color: #8fa2ff; font-size: 26px; font-weight: 700; }
QLabel#dropHint { color: #8892b0; font-size: 11px; }

QPushButton {
    background-color: #24243f;
    color: #e0e0ff;
    border: 1px solid #393963;
    border-radius: 7px;
    padding: 8px 10px;
    font-weight: 600;
}
QPushButton:hover { background-color: #30305a; border-color: #667eea; }
QPushButton:pressed { background-color: #1d1d37; }
QPushButton:disabled { color: #5d647e; background-color: #19192e; border-color: #252542; }
QPushButton[primary="true"] {
    background-color: #667eea;
    color: white;
    border-color: #7588ed;
}
QPushButton[primary="true"]:hover { background-color: #764ba2; }
QPushButton[danger="true"] { color: #f58a8a; border-color: #683848; }
QPushButton:checked { background-color: #4258bd; color: white; border-color: #8fa2ff; }

QToolButton {
    background-color: #20203b;
    color: #e0e0ff;
    border: 1px solid #383864;
    border-radius: 7px;
    min-width: 30px;
    min-height: 30px;
    font-weight: 600;
}
QToolButton:hover { background-color: #30305a; border-color: #667eea; }
QToolButton:checked { background-color: #667eea; color: white; }

QListWidget {
    background-color: #111126;
    border: 1px solid #29294c;
    border-radius: 8px;
    padding: 4px;
    outline: none;
}
QListWidget::item { color: #cfd3f7; padding: 7px; border-radius: 5px; }
QListWidget::item:selected { background-color: #313e83; color: white; }
QListWidget::item:disabled { color: #6f7691; }

QSlider::groove:horizontal {
    height: 5px;
    background: #080817;
    border-radius: 2px;
}
QSlider::sub-page:horizontal { background: #667eea; border-radius: 2px; }
QSlider::handle:horizontal {
    background: #e8eaff;
    border: 2px solid #667eea;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}

QRadioButton { color: #bfc5e7; spacing: 6px; }
QRadioButton::indicator {
    width: 13px;
    height: 13px;
    border: 1px solid #51517f;
    border-radius: 7px;
    background: #101023;
}
QRadioButton::indicator:checked { background: #667eea; border: 3px solid #afbaff; }

QFrame#viewportPanel { background-color: #080814; }
QFrame#viewportToolbar, QFrame#sceneCard, QFrame#viewportBadge, QFrame#bottomHint {
    background-color: rgba(18, 18, 42, 232);
    border: 1px solid #33335d;
    border-radius: 10px;
}
QLabel#viewportTitle { color: #d9dcff; font-weight: 700; }
QLabel#readyDot { color: #48bb78; font-size: 14px; }
QLabel#sceneTitle { color: #e0e0ff; font-weight: 700; font-size: 13px; }
QLabel#sceneKey { color: #8892b0; }
QLabel#sceneValue { color: #cdd2ff; font-family: Consolas, monospace; }
QLabel#viewportHint { color: #8892b0; }

QStatusBar {
    background-color: #101025;
    color: #9fa7c7;
    border-top: 1px solid #29294c;
}
QStatusBar::item { border: none; }

QMessageBox { background-color: #171731; }
QMessageBox QLabel { color: #e0e0ff; }
"""
