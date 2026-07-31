import sys
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow
from db.connection import init_index_db


def main():
    # Ensure global index database and base folders exist before GUI starts
    init_index_db()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
