import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'facadeDetection'))

from PySide6.QtCore import QPoint, QRect
from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from ui.main_window import MainWindow


class MainWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        self.window.show()
        self._process_layout()

    def tearDown(self):
        self.window.close()
        self.app.processEvents()

    def _process_layout(self):
        self.app.processEvents()
        self.app.processEvents()

    def test_header_contains_required_business_buttons(self):
        self.assertEqual(
            [button.text() for button in self.window.header_buttons.values()],
            ['上传文件', '点云去噪', '立面检测'],
        )
        self.assertEqual(
            list(self.window.header_buttons),
            ['upload_file', 'point_cloud_denoise', 'facade_detection'],
        )

    def test_header_buttons_start_top_left_and_wrap_without_overflow(self):
        layout = self.window.header_layout
        layout.setGeometry(QRect(0, 0, 280, 200))
        geometries = [layout.itemAt(index).geometry() for index in range(layout.count())]

        self.assertEqual((geometries[0].x(), geometries[0].y()), (10, 10))
        self.assertGreater(len({geometry.y() for geometry in geometries}), 1)
        self.assertTrue(all(geometry.right() <= 270 for geometry in geometries))

        layout.setGeometry(QRect(0, 0, 900, 100))
        self.assertEqual(
            {layout.itemAt(index).geometry().y() for index in range(layout.count())},
            {10},
        )

    def test_header_height_calculation_tracks_wrapped_rows(self):
        self.assertGreater(
            self.window.header_layout.heightForWidth(280),
            self.window.header_layout.heightForWidth(900),
        )

    def test_header_spans_above_both_sidebars(self):
        header_geometry = self.window.header_dock.geometry()
        left_geometry = self.window.left_dock.geometry()
        right_geometry = self.window.right_dock.geometry()

        self.assertLess(header_geometry.bottom(), left_geometry.top())
        self.assertLess(header_geometry.bottom(), right_geometry.top())
        self.assertLessEqual(header_geometry.left(), left_geometry.left())
        self.assertGreaterEqual(header_geometry.right(), right_geometry.right())

    def test_sidebars_hide_completely_and_expand_from_edge_buttons(self):
        viewport = self.window.centralWidget()
        for collapse_button, expand_button, dock, side in (
            (
                self.window.left_sidebar_button,
                self.window.left_sidebar_expand_button,
                self.window.left_dock,
                'left',
            ),
            (
                self.window.right_sidebar_button,
                self.window.right_sidebar_expand_button,
                self.window.right_dock,
                'right',
            ),
        ):
            content = dock.widget()
            self.assertTrue(dock.isVisible())
            self.assertIs(collapse_button.parentWidget(), dock.titleBarWidget())
            self.assertIs(expand_button.parentWidget(), viewport)
            self.assertFalse(expand_button.isVisible())
            title_label = dock.titleBarWidget().findChild(
                QLabel,
                f'{dock.objectName()}TitleLabel',
            )
            self.assertEqual(
                title_label.text(),
                'Left Sidebar' if side == 'left' else 'Right Sidebar',
            )
            self.assertIn(
                'background-color: #d9d9d9',
                dock.titleBarWidget().styleSheet(),
            )
            self.assertEqual(collapse_button.text(), '◀' if side == 'left' else '▶')
            self.assertEqual(expand_button.text(), '▶' if side == 'left' else '◀')
            self.assertIsNone(
                self.window.header_panel.findChild(
                    QPushButton,
                    collapse_button.objectName(),
                )
            )
            self.assertEqual(content.findChildren(QLabel), [])
            expanded_viewport_width = viewport.width()
            expanded_dock_width = dock.width()

            collapse_button.click()
            self._process_layout()
            self.assertFalse(dock.isVisible())
            self.assertTrue(dock.visibleRegion().isEmpty())
            self.assertFalse(content.isVisible())
            self.assertTrue(expand_button.isVisible())
            self.assertTrue(expand_button.isEnabled())
            self.assertGreaterEqual(
                viewport.width(),
                expanded_viewport_width + expanded_dock_width,
            )

            button_origin = expand_button.mapToGlobal(QPoint(0, 0))
            viewport_origin = viewport.mapToGlobal(QPoint(0, 0))
            if side == 'left':
                self.assertLessEqual(abs(button_origin.x() - viewport_origin.x()), 2)
            else:
                button_right = button_origin.x() + expand_button.width()
                viewport_right = viewport_origin.x() + viewport.width()
                self.assertLessEqual(abs(button_right - viewport_right), 2)

            hidden_viewport_width = viewport.width()
            expand_button.click()
            self._process_layout()
            self.assertTrue(dock.isVisible())
            self.assertFalse(expand_button.isVisible())
            self.assertTrue(content.isVisible())
            self.assertGreaterEqual(dock.width(), 180)
            self.assertLess(viewport.width(), hidden_viewport_width)

    def test_sidebars_can_hide_together_and_restore_independently(self):
        viewport = self.window.centralWidget()
        initial_viewport_width = viewport.width()

        self.window.left_sidebar_button.click()
        self.window.right_sidebar_button.click()
        self._process_layout()

        self.assertFalse(self.window.left_dock.isVisible())
        self.assertFalse(self.window.right_dock.isVisible())
        self.assertTrue(self.window.left_sidebar_expand_button.isVisible())
        self.assertTrue(self.window.right_sidebar_expand_button.isVisible())
        self.assertGreater(viewport.width(), initial_viewport_width)

        both_hidden_width = viewport.width()
        self.window.left_sidebar_expand_button.click()
        self._process_layout()

        self.assertTrue(self.window.left_dock.isVisible())
        self.assertFalse(self.window.right_dock.isVisible())
        self.assertFalse(self.window.left_sidebar_expand_button.isVisible())
        self.assertTrue(self.window.right_sidebar_expand_button.isVisible())
        self.assertLess(viewport.width(), both_hidden_width)

        self.window.close()
        self._process_layout()
        self.assertFalse(self.window.left_sidebar_expand_button.isVisible())
        self.assertFalse(self.window.right_sidebar_expand_button.isVisible())

    def test_bottom_bar_is_visible_and_contains_no_buttons(self):
        self.assertTrue(self.window.bottom_dock.isVisible())
        self.assertEqual(self.window.bottom_dock.objectName(), 'bottomDock')
        self.assertEqual(
            self.window.bottom_dock.widget().findChildren(QPushButton),
            [],
        )

    def test_business_buttons_trigger_matching_service_methods(self):
        output = io.StringIO()
        with redirect_stdout(output):
            for button in self.window.header_buttons.values():
                button.click()

        self.assertEqual(
            output.getvalue().splitlines(),
            [
                'upload_file被点击了',
                'point_cloud_denoise被点击了',
                'facade_detection被点击了',
            ],
        )


if __name__ == '__main__':
    unittest.main()
