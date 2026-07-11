"""Fast UI checks that do not require a visible OpenGL window."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("FACADE_DISABLE_OPEN3D", "1")

from PySide6.QtWidgets import QApplication  # noqa: E402

from facadeDetection.ui.main_window import MainWindow  # noqa: E402
from facadeDetection.view3d.open3d_viewport import Open3DViewport  # noqa: E402


class MainWindowSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.window = MainWindow()
        self.window.show()
        self.app.processEvents()

    def tearDown(self) -> None:
        self.window.close()
        self.app.processEvents()

    def test_expected_ui_regions_exist(self) -> None:
        self.assertIn("PointCloud FacadeDetection", self.window.windowTitle())
        self.assertEqual(self.window.control_panel.minimumWidth(), 340)
        self.assertEqual(self.window.control_panel.file_list.count(), 1)
        self.assertEqual(
            set(self.window.viewport_panel.tool_buttons),
            {"rotate", "pan", "zoom", "grid", "axes", "screenshot", "fullscreen"},
        )

    def test_algorithm_controls_start_disabled(self) -> None:
        self.assertTrue(self.window.control_panel._cloud_buttons)
        self.assertTrue(all(not button.isEnabled() for button in self.window.control_panel._cloud_buttons))

    def test_workflow_buttons_remain_locked_after_cloud_registration(self) -> None:
        panel = self.window.control_panel
        panel.register_cloud("demo", 42)
        self.assertTrue(all(button.isEnabled() for button in panel._cloud_buttons))
        self.assertTrue(all(not button.isEnabled() for button in panel._workflow_buttons))

    def test_navigation_modes_are_exclusive(self) -> None:
        buttons = self.window.viewport_panel.tool_buttons
        buttons["pan"].click()
        buttons["zoom"].click()
        self.assertFalse(buttons["rotate"].isChecked())
        self.assertFalse(buttons["pan"].isChecked())
        self.assertTrue(buttons["zoom"].isChecked())

    def test_reference_grid_is_real_open3d_geometry(self) -> None:
        grid = Open3DViewport._make_grid(extent=10, step=1)
        self.assertEqual(len(grid.points), 84)
        self.assertEqual(len(grid.lines), 42)


if __name__ == "__main__":
    unittest.main()
