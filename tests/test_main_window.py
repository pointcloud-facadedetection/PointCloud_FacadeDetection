import os
import sys
import unittest
from contextlib import redirect_stdout
import io
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'facadeDetection'))

from PySide6.QtCore import QPoint, QRect
from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from ui.main_window import (
    MainWindow,
    PAGE_BUTTON_NAMES,
    PAGE_DEFINITIONS,
    PAGE_HEADER_ACTIONS,
    UPLOAD_FILE_FILTER,
)


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

    def _visible_header_buttons(self):
        return [
            button
            for button in self.window.header_buttons.values()
            if button.isVisible()
        ]

    def test_header_actions_follow_current_page(self):
        self.assertEqual(
            [button.text() for button in self._visible_header_buttons()],
            ['上传文件'],
        )
        self.assertEqual(
            [
                button_name
                for button_name, button in self.window.header_buttons.items()
                if button.isVisible()
            ],
            ['btn_upload'],
        )

        self.window.page_buttons['project_operation'].click()
        self._process_layout()
        self.assertEqual(
            [button.text() for button in self._visible_header_buttons()],
            [
                label
                for label, _button_name, _action_name
                in PAGE_HEADER_ACTIONS['project_operation']
            ],
        )

        self.window.page_buttons['inspection_review'].click()
        self._process_layout()
        self.assertEqual(self._visible_header_buttons(), [])

    def test_header_buttons_start_top_left_and_wrap_without_overflow(self):
        self.window.page_buttons['project_operation'].click()
        self._process_layout()
        layout = self.window.header_layout
        layout.setGeometry(QRect(0, 0, 280, 200))
        visible_items = [
            layout.itemAt(index)
            for index in range(layout.count())
            if not layout.itemAt(index).isEmpty()
        ]
        geometries = [item.geometry() for item in visible_items]

        self.assertEqual((geometries[0].x(), geometries[0].y()), (10, 10))
        self.assertGreater(len({geometry.y() for geometry in geometries}), 1)
        self.assertTrue(all(geometry.right() <= 270 for geometry in geometries))

        layout.setGeometry(QRect(0, 0, 1400, 100))
        self.assertEqual(
            {item.geometry().y() for item in visible_items},
            {10},
        )

    def test_header_height_calculation_tracks_wrapped_rows(self):
        self.window.page_buttons['project_operation'].click()
        self._process_layout()
        self.assertGreater(
            self.window.header_layout.heightForWidth(280),
            self.window.header_layout.heightForWidth(1400),
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
            self.assertIs(expand_button.parentWidget(), self.window)
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
            self.assertEqual(
                collapse_button.objectName(),
                f'btn_collapse_{side}_sidebar',
            )
            self.assertEqual(
                expand_button.objectName(),
                f'btn_expand_{side}_sidebar',
            )
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

            button_rect = QRect(
                expand_button.mapToGlobal(QPoint(0, 0)),
                expand_button.size(),
            )
            viewport_rect = QRect(
                viewport.mapToGlobal(QPoint(0, 0)),
                viewport.size(),
            )
            self.assertTrue(viewport_rect.contains(button_rect))
            self.assertLessEqual(button_rect.top() - viewport_rect.top(), 8)
            if side == 'left':
                self.assertLessEqual(button_rect.left() - viewport_rect.left(), 8)
            else:
                self.assertLessEqual(viewport_rect.right() - button_rect.right(), 8)

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

    def test_bottom_bar_contains_four_mutually_exclusive_page_tabs(self):
        self.assertTrue(self.window.bottom_dock.isVisible())
        self.assertEqual(self.window.bottom_dock.objectName(), 'bottomDock')
        self.assertEqual(
            [button.text() for button in self.window.page_buttons.values()],
            [title for title, _key in PAGE_DEFINITIONS],
        )
        self.assertEqual(
            [button.objectName() for button in self.window.page_buttons.values()],
            [
                PAGE_BUTTON_NAMES[page_key]
                for _title, page_key in PAGE_DEFINITIONS
            ],
        )
        self.assertEqual(self.window.centralWidget().currentIndex(), 0)
        self.assertIs(
            self.window.centralWidget().widget(1),
            self.window.viewport.get_widget(),
        )

        for index, (_title, page_key) in enumerate(PAGE_DEFINITIONS):
            button = self.window.page_buttons[page_key]
            button.click()
            self._process_layout()

            self.assertEqual(self.window.centralWidget().currentIndex(), index)
            self.assertTrue(button.isChecked())
            self.assertEqual(
                sum(tab.isChecked() for tab in self.window.page_buttons.values()),
                1,
            )

    def test_business_buttons_trigger_matching_service_methods(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.window.page_buttons['project_operation'].click()
            self._process_layout()
            for (
                _label,
                button_name,
                _action_name,
            ) in PAGE_HEADER_ACTIONS['project_operation']:
                button = self.window.header_buttons[button_name]
                button.click()

        self.assertEqual(
            output.getvalue().splitlines(),
            [
                'reset triggered',
                'change_colors triggered',
                'denoise triggered',
                'registration triggered',
                'facade_detection triggered',
                'compute_quality triggered',
                'segmentation triggered',
                'compute_detail triggered',
                'align_2d_3d triggered',
            ],
        )

    def test_upload_button_opens_file_dialog_and_triggers_upload_chain(self):
        selected_files = [
            'C:/data/facade.ply',
            'C:/data/facade-photo.png',
        ]
        output = io.StringIO()
        with patch(
            'ui.main_window.QFileDialog.getOpenFileNames',
            return_value=(selected_files, '项目支持文件'),
        ) as file_dialog:
            with redirect_stdout(output):
                self.window.header_buttons['btn_upload'].click()

        file_dialog.assert_called_once_with(
            self.window,
            '选择点云或图像文件',
            str(Path.home()),
            UPLOAD_FILE_FILTER,
        )
        self.assertEqual(
            self.window.button_service.selected_file_paths,
            selected_files,
        )
        self.assertEqual(
            self.window.button_service.extracted_file_paths,
            selected_files,
        )
        self.assertEqual(self.window._last_upload_directory, 'C:\\data')
        self.assertEqual(
            output.getvalue().splitlines(),
            ['upload_files triggered', 'extract_files triggered'],
        )

    def test_canceling_upload_dialog_does_not_trigger_service(self):
        output = io.StringIO()
        with patch(
            'ui.main_window.QFileDialog.getOpenFileNames',
            return_value=([], ''),
        ):
            with redirect_stdout(output):
                self.window.header_buttons['btn_upload'].click()

        self.assertEqual(self.window.button_service.selected_file_paths, [])
        self.assertEqual(self.window.button_service.extracted_file_paths, [])
        self.assertEqual(output.getvalue(), '')


if __name__ == '__main__':
    unittest.main()
