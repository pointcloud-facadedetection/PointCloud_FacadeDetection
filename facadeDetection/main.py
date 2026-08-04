import sys
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow
from db.connection import init_index_db


def main():
    # Ensure global index database and base folders exist before GUI starts
    init_index_db()
    app = QApplication(sys.argv)
    # Open3D 原生窗口嵌入 Qt 时会短暂触发“最后一个窗口关闭”。
    app.setQuitOnLastWindowClosed(False)
    window = MainWindow()
    window.show()
    app.setQuitOnLastWindowClosed(True)
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
