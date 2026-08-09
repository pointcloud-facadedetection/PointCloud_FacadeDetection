import sys
from PySide6.QtWidgets import QApplication

from ui.main_window import APPLICATION_TITLE, MainWindow
from ui.theme import apply_application_theme
from db.connection import init_index_db


def main():
    # Ensure global index database and base folders exist before GUI starts
    init_index_db()
    app = QApplication(sys.argv)
    app.setApplicationName(APPLICATION_TITLE)
    app.setOrganizationName('PointCloud FacadeDetection')
    apply_application_theme(app)
    # Open3D 原生窗口嵌入 Qt 时会短暂触发“最后一个窗口关闭”。
    app.setQuitOnLastWindowClosed(False)
    window = MainWindow()
    # 默认最大化，同时保留 MainWindow 的常规尺寸供用户退出最大化后使用。
    window.showMaximized()
    app.setQuitOnLastWindowClosed(True)
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
