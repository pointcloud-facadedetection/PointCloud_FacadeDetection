"""Behaviour tests for the three-page first-version desktop UI."""

from __future__ import annotations

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'facadeDetection'))

from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QTableWidget,
    QWidget,
)

from ui.dialogs.new_project_dialog import PROJECT_FIELDS as DIALOG_FIELDS
from ui.dialogs.new_project_dialog import NewProjectDialog
from ui.main_window import MainWindow
from ui.styles import PLACEHOLDER


PROJECT_INPUT = {
    'project_name': '测试项目',
    'report_number': 'BG-001',
    'developer': '某建设单位',
    'inspection_unit': '某检测单位',
    'location': '东立面',
    'start_floor': '1层',
    'end_floor': '10层',
    'inspection_standard': 'JGJ/T 132',
    'measurement_date': '2026-07-21',
}


REQUIRED_WINDOW_BUTTONS = {
    # Home.
    'connectDeviceButton',
    'newProjectButton',
    'settingsButton',
    # Workbench header and navigation.
    'back_to_home',
    'save_project',
    'recalculate_project',
    'generate_pdf_report',
    'nav_data',
    'nav_walls',
    'nav_review',
    'nav_results',
    'nav_report',
    # Data import.
    'connect_device',
    'disconnect_device',
    'import_point_cloud',
    'import_site_photos',
    'import_facade_drawing',
    *(f'view_imported_file_{index}' for index in range(1, 4)),
    *(f'reimport_file_{index}' for index in range(1, 4)),
    # Wall list and visual tools.
    *(f'select_wall_{index}' for index in range(1, 7)),
    'view_flatness',
    'view_verticality',
    'reset_3d_view',
    'fit_3d_view',
    'pan_3d_view',
    'toggle_floor_lines',
    'toggle_anomaly_boxes',
    # Inspection and review tools.
    'close_anomaly',
    'mark_anomaly_valid',
    'mark_anomaly_ignored',
    'auto_detect_boundary',
    'manual_adjust_boundary',
    'confirm_wall_boundary',
    'auto_generate_floor_lines',
    'manual_adjust_floor_lines',
    'confirm_floor_range',
    'auto_align',
    'manual_align',
    'confirm_alignment',
    'view_anomaly_regions',
    'add_review_note',
    'save_review_result',
    'mark_wall_reviewed',
    # Results and report landing.
    'filter_all_walls',
    'filter_all_metrics',
    'filter_unreviewed',
    'sort_pass_rate',
    'open_report_preview',
    'summary_overall',
    'summary_flatness',
    'summary_verticality',
    'summary_report',
    # Report page.
    'report_back',
    'report_refresh',
    'report_edit',
    'report_export_pdf',
    'report_print',
    'toc_cover',
    'toc_project_info',
    'toc_site_photos',
    'toc_equipment',
    'toc_overall_summary',
    'toc_flatness',
    'toc_verticality',
    'toc_defect_notes',
    'toc_conclusion',
    'report_previous_page',
    'report_next_page',
    'report_zoom_out',
    'report_zoom_in',
    'report_fullscreen',
    'report_editor_close',
    'report_editor_cancel',
    'report_editor_save',
}


class MainWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        self.window.show()
        self._process_events()

    def tearDown(self):
        self.window.close()
        self._process_events()

    def _process_events(self):
        self.app.processEvents()
        self.app.processEvents()

    @staticmethod
    def _complete_project_data():
        return {
            field_name: PROJECT_INPUT.get(field_name, '')
            for _label, field_name in DIALOG_FIELDS
        }

    def _card_text(self, card, field_name):
        label = card.findChild(
            QLabel,
            f'project_{card.project_id}_{field_name}',
        )
        self.assertIsNotNone(label)
        return label.text()

    def test_home_starts_empty_without_project_cards(self):
        home = self.window.home_page

        self.assertEqual(PLACEHOLDER, '')
        self.assertEqual(self.window.current_page_name, 'home')
        self.assertEqual(home.project_count, 0)
        self.assertEqual(home.project_count_label.text(), '共 0 个项目')
        self.assertEqual(home.project_grid.count(), 0)
        self.assertEqual(home.project_cards, ())

    def test_added_projects_are_laid_out_in_two_columns(self):
        home = self.window.home_page
        for _index in range(3):
            home.add_project(self._complete_project_data())

        self.assertEqual(
            {
                home.project_grid.getItemPosition(index)[:2]
                for index in range(home.project_grid.count())
            },
            {(0, 0), (0, 1), (1, 0)},
        )

    def test_new_project_data_flows_to_card_workbench_and_report(self):
        self.window.open_new_project()
        self._process_events()
        dialog = self.window.new_project_dialog

        self.assertIsNotNone(dialog)
        self.assertEqual(set(dialog.fields), {name for _label, name in DIALOG_FIELDS})
        self.assertEqual(
            {editor.text() for editor in dialog.fields.values()},
            {''},
        )
        for field_name, value in PROJECT_INPUT.items():
            dialog.fields[field_name].setText(value)

        with redirect_stdout(io.StringIO()):
            dialog.create_button.click()
        self._process_events()

        home = self.window.home_page
        self.assertEqual(home.project_count, 1)
        self.assertEqual(home.project_count_label.text(), '共 1 个项目')
        self.assertEqual(self.window.current_page_name, 'workbench')
        self.assertIsNone(self.window.new_project_dialog)

        card = home.project_cards[0]
        expected_project_data = self._complete_project_data()
        self.assertEqual(card.project_data, expected_project_data)
        self.assertEqual(home.get_project_data(card.project_id), expected_project_data)
        self.assertEqual(self._card_text(card, 'title'), '测试项目')
        self.assertEqual(self._card_text(card, 'status'), '')
        self.assertEqual(self._card_text(card, 'location_value'), '东立面')
        self.assertEqual(self._card_text(card, 'scope_value'), '1层 至 10层')
        self.assertEqual(self._card_text(card, 'date_value'), '2026-07-21')
        for field_name in ('wall_count_value', 'pass_rate_value', 'updated_at_value'):
            self.assertEqual(self._card_text(card, field_name), '')

        workbench = self.window.workbench_page
        self.assertEqual(workbench.project_data, expected_project_data)
        self.assertEqual(workbench.project_name_label.text(), '测试项目')
        self.assertEqual(
            workbench.project_meta_label.text(),
            '检测范围 1层 至 10层  ·  检测标准 JGJ/T 132',
        )

        report = self.window.report_page
        self.assertEqual(report.report_header_number.text(), '报告编号：BG-001')
        self.assertEqual(report.report_paper_number.text(), '报告编号：BG-001')
        self.assertEqual(report.report_paper_organization.text(), '某检测单位')
        expected_paper = {
            'project_name': '测试项目',
            'developer': '某建设单位',
            'contractor': '',
            'inspection_unit': '某检测单位',
            'measurement_date': '2026-07-21',
            'report_date': '',
        }
        self.assertEqual(
            {
                field_name: label.text()
                for field_name, label in report.paper_metadata_values.items()
            },
            expected_paper,
        )
        expected_editor = {
            'project_name': '测试项目',
            'report_number': 'BG-001',
            'developer': '某建设单位',
            'contractor': '',
            'inspection_unit': '某检测单位',
            'supervisor': '',
            'inspection_date': '2026-07-21',
            'report_date': '',
        }
        self.assertEqual(
            {
                field_name: report.editor_fields[field_name].text()
                for field_name in expected_editor
            },
            expected_editor,
        )
        self.assertEqual(report.editor_fields['conclusion'].toPlainText(), '')

    def test_workbench_contains_all_major_regions_and_placeholder_viewport(self):
        self.window.show_workbench()
        self._process_events()
        page = self.window.workbench_page

        for object_name in (
            'workbenchHeader',
            'workbenchBody',
            'navRail',
            'workspaceContent',
            'wallListPanel',
            'viewportHost',
            'pointCloudViewport',
            'facade2dPanel',
            'rightInspector',
            'summaryBar',
        ):
            with self.subTest(region=object_name):
                self.assertIsNotNone(page.findChild(QWidget, object_name))

        viewport_host = page.findChild(QWidget, 'viewportHost')
        viewport = page.findChild(QWidget, 'pointCloudViewport')
        self.assertIs(viewport.parentWidget(), viewport_host)
        self.assertIn(
            PLACEHOLDER,
            [label.text() for label in viewport.findChildren(QLabel)],
        )

    def test_workbench_navigation_selects_data_results_and_report(self):
        page = self.window.workbench_page
        self.window.show_workbench()

        page.nav_buttons['data'].click()
        self.assertEqual(page.current_mode, 'data')
        self.assertIs(page.content_stack.currentWidget(), page.data_page)

        page.nav_buttons['walls'].click()
        self.assertEqual(page.current_mode, 'walls')
        self.assertIs(page.content_stack.currentWidget(), page.analysis_page)
        self.assertEqual(page.inspector_stack.currentIndex(), 0)

        page.nav_buttons['review'].click()
        self.assertEqual(page.current_mode, 'review')
        self.assertIs(page.content_stack.currentWidget(), page.analysis_page)
        self.assertEqual(page.inspector_stack.currentIndex(), 1)

        page.nav_buttons['results'].click()
        self.assertEqual(page.current_mode, 'results')
        self.assertIs(page.content_stack.currentWidget(), page.results_page)

        page.nav_buttons['report'].click()
        self.assertEqual(page.current_mode, 'report')
        self.assertIs(page.content_stack.currentWidget(), page.report_landing_page)

        page.findChild(QPushButton, 'open_report_preview').click()
        self.assertEqual(self.window.current_page_name, 'report')

    def test_unpopulated_tables_and_summary_values_are_empty(self):
        page = self.window.workbench_page
        imported_files = page.findChild(QTableWidget, 'importedFilesTable')
        results = page.findChild(QTableWidget, 'resultsTable')

        self.assertTrue(
            all(
                imported_files.item(row, column).text() == PLACEHOLDER
                for row in range(imported_files.rowCount())
                for column in range(4)
            )
        )
        self.assertTrue(
            all(
                results.item(row, column).text() == PLACEHOLDER
                for row in range(results.rowCount())
                for column in range(results.columnCount())
            )
        )
        for name in (
            'summary_overallValue',
            'summary_flatnessValue',
            'summary_verticalityValue',
            'summary_reportValue',
        ):
            self.assertEqual(page.findChild(QLabel, name).text(), PLACEHOLDER)

    def test_report_navigation_and_editor_drawer(self):
        self.window.show_report()
        self._process_events()
        report = self.window.report_page

        self.assertEqual(self.window.current_page_name, 'report')
        self.assertTrue(report.editor.isHidden())
        report.edit_button.click()
        self._process_events()
        self.assertFalse(report.editor.isHidden())
        self.assertEqual(
            {editor.text() for editor in report.editor_fields.values() if hasattr(editor, 'text')},
            {PLACEHOLDER},
        )
        self.assertEqual(
            report.editor_fields['conclusion'].toPlainText(), PLACEHOLDER
        )

        report.editor_cancel_button.click()
        self.assertTrue(report.editor.isHidden())
        report.back_button.click()
        self.assertEqual(self.window.current_page_name, 'workbench')

    def test_every_required_window_button_is_enabled_and_emits_clicked(self):
        missing = [
            name
            for name in sorted(REQUIRED_WINDOW_BUTTONS)
            if self.window.findChild(QPushButton, name) is None
        ]
        self.assertEqual(missing, [])

        with redirect_stdout(io.StringIO()):
            for object_name in sorted(REQUIRED_WINDOW_BUTTONS):
                with self.subTest(button=object_name):
                    button = self.window.findChild(QPushButton, object_name)
                    self.assertTrue(button.isEnabled())
                    spy = QSignalSpy(button.clicked)
                    button.click()
                    self.assertEqual(spy.count(), 1)

            # A newly created project card has the three reference actions.
            card = self.window.home_page.add_project(self._complete_project_data())
            for prefix in ('enter_project', 'export_project', 'delete_project'):
                object_name = f'{prefix}_{card.project_id}'
                button = card.findChild(QPushButton, object_name)
                self.assertIsNotNone(button)
                self.assertTrue(button.isEnabled())
                spy = QSignalSpy(button.clicked)
                button.click()
                self.assertEqual(spy.count(), 1)

    def test_dialog_action_buttons_are_enabled_and_emit_clicked(self):
        for object_name in (
            'close_new_project',
            'cancel_new_project',
            'create_project_and_enter',
        ):
            with self.subTest(button=object_name):
                dialog = NewProjectDialog(self.window)
                button = dialog.findChild(QPushButton, object_name)
                self.assertIsNotNone(button)
                self.assertTrue(button.isEnabled())
                spy = QSignalSpy(button.clicked)
                button.click()
                self.assertEqual(spy.count(), 1)
                dialog.deleteLater()
                self._process_events()

    def test_connected_business_button_reaches_service(self):
        button = self.window.findChild(QPushButton, 'connectDeviceButton')
        output = io.StringIO()
        with redirect_stdout(output):
            button.click()

        self.assertEqual(output.getvalue(), '')
        self.assertEqual(self.window.button_service.last_action, 'connect_device')
        self.assertIn('connect_device', self.window.button_service.action_history)


if __name__ == '__main__':
    unittest.main()
