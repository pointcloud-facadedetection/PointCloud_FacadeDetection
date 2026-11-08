from __future__ import annotations

import os
import uuid
from pathlib import Path

import cv2
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from config.storage import Storage


class CameraDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._capture = None
        self._frame = None
        self._photo_path = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._show_frame)
        self.setWindowTitle('本机相机拍照')
        self.setModal(True)
        self.resize(760, 580)

        layout = QVBoxLayout(self)
        self.preview = QLabel('正在连接本机相机…')
        self.preview.setObjectName('cameraPreview')
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(640, 360)
        self.preview.setStyleSheet('background:#111827; color:#E2E8F0;')
        layout.addWidget(self.preview, 1)

        self.status = QLabel()
        self.status.setObjectName('cameraStatus')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.btn_capture = QPushButton('拍摄照片')
        self.btn_capture.setObjectName('btn_capture_photo')
        self.btn_capture.setProperty('buttonRole', 'primary')
        self.btn_capture.clicked.connect(self._capture_photo)
        buttons.addWidget(self.btn_capture)
        self.btn_use = QPushButton('使用这张照片')
        self.btn_use.setObjectName('btn_use_photo')
        self.btn_use.setProperty('buttonRole', 'primary')
        self.btn_use.setEnabled(False)
        self.btn_use.clicked.connect(self.accept)
        buttons.addWidget(self.btn_use)
        cancel = QPushButton('取消')
        cancel.setProperty('buttonRole', 'secondary')
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)
        self.btn_capture.setEnabled(False)
        self._open_camera()

    def _open_camera(self):
        try:
            backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
            capture = cv2.VideoCapture(0, backend)
            if not capture.isOpened():
                capture.release()
                capture = cv2.VideoCapture(0, cv2.CAP_ANY)
            if not capture.isOpened():
                capture.release()
                raise RuntimeError('无法打开本机相机，请检查设备连接、系统相机权限及是否被其他应用占用。')
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._capture = capture
            self.status.setText('相机已连接，请调整角度后拍摄。')
            self._timer.start(33)
        except Exception as error:
            self._capture = None
            self.preview.setText('无法使用本机相机')
            self.status.setText(str(error))
            self.btn_capture.setEnabled(False)
            self.btn_use.setEnabled(False)

    def _show_frame(self):
        if self._capture is None:
            return
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return
        self._frame = frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height,
                       channels * width, QImage.Format.Format_RGB888).copy()
        self.preview.setPixmap(QPixmap.fromImage(image).scaled(
            self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))
        self.btn_capture.setEnabled(True)

    def _capture_photo(self):
        if self._photo_path is not None:
            self._photo_path = None
            self.btn_use.setEnabled(False)
            self.btn_capture.setText('拍摄照片')
            self.status.setText('请调整角度后再次拍摄。')
            self._timer.start(33)
            return
        if self._frame is None:
            QMessageBox.information(self, '拍摄照片', '相机画面尚未就绪，请稍后重试。')
            return
        try:
            directory = Path(Storage.DATA_DIR) / 'captures'
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f'capture_{uuid.uuid4().hex}.jpg'
            if not cv2.imwrite(str(path), self._frame):
                raise OSError('照片写入失败，请检查应用数据目录的写权限。')
            self._photo_path = str(path)
            self._timer.stop()
            self.btn_capture.setText('重拍')
            self.status.setText(f'照片已保存：{path}\n点击“使用这张照片”加入项目。')
            self.btn_use.setEnabled(True)
        except Exception as error:
            QMessageBox.warning(self, '保存照片', str(error))

    def photo_path(self):
        return self._photo_path

    def accept(self):
        if not self._photo_path:
            QMessageBox.information(self, '使用照片', '请先拍摄一张照片。')
            return
        super().accept()

    def _stop_camera(self):
        self._timer.stop()
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def done(self, result):
        self._stop_camera()
        super().done(result)

    def closeEvent(self, event):
        self._stop_camera()
        super().closeEvent(event)
