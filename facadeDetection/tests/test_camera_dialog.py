from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QDialog

from config.storage import Storage
from ui.dialogs.camera_dialog import CameraDialog


class FakeCapture:
    def __init__(self, opened=True):
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        return True, np.full((16, 24, 3), 127, dtype=np.uint8)

    def set(self, *_args):
        return True

    def release(self):
        self.released = True


def test_camera_capture_saves_photo_and_releases_device(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(Storage, 'DATA_DIR', tmp_path)
    capture = FakeCapture()
    monkeypatch.setattr('ui.dialogs.camera_dialog.cv2.VideoCapture',
                        lambda *_args: capture)
    dialog = CameraDialog()
    dialog._show_frame()
    dialog._capture_photo()
    path = Path(dialog.photo_path())
    assert path.is_file()
    assert path.parent == tmp_path / 'captures'
    assert path.read_bytes()
    dialog._capture_photo()
    assert dialog.photo_path() is None
    assert not dialog.btn_use.isEnabled()
    assert dialog._timer.isActive()
    dialog._capture_photo()
    assert dialog.photo_path() is not None
    dialog.accept()
    assert capture.released
    assert dialog._capture is None
    assert not dialog._timer.isActive()


def test_camera_unavailable_is_explained_without_a_photo(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(Storage, 'DATA_DIR', tmp_path)
    capture = FakeCapture(opened=False)
    monkeypatch.setattr('ui.dialogs.camera_dialog.cv2.VideoCapture',
                        lambda *_args: capture)
    dialog = CameraDialog()
    assert dialog.photo_path() is None
    assert not dialog.btn_capture.isEnabled()
    assert not dialog.btn_use.isEnabled()
    assert '无法打开本机相机' in dialog.status.text()
    dialog.close()


def test_capture_button_adds_photo_to_create_and_edit_forms(qapp, monkeypatch, tmp_path):
    from services.project_overview.project_overview_service import ProjectCard
    from ui.dialogs.project_create_dialog import ProjectCreateDialog
    photo = tmp_path / 'captured.jpg'
    photo.write_bytes(b'photo')

    class FakeDialog:
        def __init__(self, _parent):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        def photo_path(self):
            return str(photo)

    monkeypatch.setattr('ui.dialogs.camera_dialog.CameraDialog', FakeDialog)
    dialog = ProjectCreateDialog()
    assert dialog.btn_capture_photo.isEnabled()
    dialog._open_camera()
    assert dialog._resource_values('photo_files') == [str(photo)]
    assert dialog._resource_counts['photo_files'].text() == '已添加 1 项'
    dialog.close()

    project = ProjectCard(project_id='p', name='编辑项目', directory_path='')
    editor = ProjectCreateDialog(project=project)
    assert editor.btn_capture_photo.isEnabled()
    editor._open_camera()
    assert editor._resource_values('photo_files') == [str(photo)]
    editor.close()
