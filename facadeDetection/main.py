import sys
from PySide6.QtWidgets import QApplication

try:
    from .ui.main_window import MainWindow
except ImportError:
    from ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName('PointCloud FacadeDetection')
    app.setOrganizationName('pointcloud-facadedetection')
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
