import sys
from PySide6.QtWidgets import QApplication

from ui.main_window import APPLICATION_TITLE, MainWindow
from ui.theme import apply_application_theme
from db.connection import init_index_db

def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APPLICATION_TITLE)
    app.setOrganizationName('PointCloud FacadeDetection')
    apply_application_theme(app)
    # 窗口创建前初始化索引数据库，保证项目列表与最新持久化结构可用。
    init_index_db()
    app.setQuitOnLastWindowClosed(False)
    window = MainWindow()
    # 默认最大化，同时保留 MainWindow 的常规尺寸供用户退出最大化后使用。
    window.show()
    from PySide6.QtCore import QTimer
    QTimer.singleShot(0, window.showMaximized)
    app.setQuitOnLastWindowClosed(True)
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
