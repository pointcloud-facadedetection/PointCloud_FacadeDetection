from types import SimpleNamespace

import cv2
import numpy as np
from PySide6.QtWidgets import QMessageBox

from services.report_export import ReportDataService


def test_project_dialog_requires_report_number_and_photo(qapp, tmp_path, monkeypatch):
    from ui.dialogs.project_create_dialog import ProjectCreateDialog

    dialog = ProjectCreateDialog()
    accepted = []
    dialog.accept = lambda: accepted.append(True)
    monkeypatch.setattr(QMessageBox, 'warning', lambda *_args: None)
    dialog.edt_name.setText('校验项目')
    dialog.spin_base_elevation.setValue(0.0)
    dialog.spin_base_floor_no.setValue(1)
    dialog.spin_standard_floor_height.setValue(3.0)
    dialog.spin_total_floor_count.setValue(10)
    dialog.edt_report_no.clear()
    dialog._on_accept()
    assert accepted == []
    assert dialog.tabs.currentIndex() == 2

    dialog.edt_report_no.setText('RJ-001')
    dialog._on_accept()
    assert accepted == []
    assert dialog.tabs.currentIndex() == 4

    photo = tmp_path / '现场.jpg'
    photo.write_bytes(b'photo')
    dialog._add_paths('photo_files', '现场照片上传', [str(photo)])
    dialog._on_accept()
    assert accepted == [True]
    dialog.close()


def test_report_validation_rejects_missing_and_black_photo(tmp_path):
    snapshot = {
        'project': {'report_no': 'RJ-001'},
        'facades': [{'wall_label': '墙面 1', 'images': []}],
    }
    assert any('缺少现场照片' in error
               for error in ReportDataService.validate_for_export(snapshot))

    black = tmp_path / 'black.png'
    cv2.imwrite(str(black), np.zeros((20, 20, 3), dtype=np.uint8))
    snapshot['facades'][0]['images'] = [{'key': 'photo', 'path': str(black)}]
    assert any('空白或全黑' in error
               for error in ReportDataService.validate_for_export(snapshot))

    valid = tmp_path / 'valid.png'
    cv2.imwrite(str(valid), np.arange(20 * 20 * 3, dtype=np.uint8).reshape(20, 20, 3))
    snapshot['facades'][0]['images'] = [{'key': 'photo', 'path': str(valid)}]
    assert ReportDataService.validate_for_export(snapshot) == []


def test_export_is_blocked_before_file_dialog(qapp, monkeypatch):
    from ui.pages.report_page import ReportPageMixin

    harness = ReportPageMixin()
    harness.current_project = SimpleNamespace(directory_path='unused')
    harness._report_preview_dirty = False
    harness._report_snapshot = {}
    monkeypatch.setattr(ReportDataService, 'validate_for_export',
                        lambda _snapshot: ['缺少现场照片'])
    called = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: called.append(args))
    monkeypatch.setattr(
        'ui.pages.report_page.QFileDialog.getSaveFileName',
        lambda *_args: (_ for _ in ()).throw(AssertionError('不得进入文件保存对话框')))
    harness._export_quality_report()
    assert called
