"""独立的双图片 SuperPoint + LightGlue 匹配可视化工具。"""

from __future__ import annotations

# PySide6 与 OpenCV 的运行时生成成员无法被 Pylint 静态识别。
# pylint: disable=no-name-in-module,no-member

import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


SUPPORTED_IMAGES = "图片 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
MAX_PROCESSING_SIDE = 1600
MAX_KEYPOINTS = 4096
_LIGHTGLUE_ENGINE = None


def read_image(path: str) -> np.ndarray:
    """读取图片，兼容 Windows 中文路径。"""
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片：{path}")
    return image


def resize_for_processing(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, MAX_PROCESSING_SIDE / max(height, width))
    if scale == 1.0:
        return image
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def grabcut_foreground(image: np.ndarray, rectangle) -> tuple[np.ndarray, np.ndarray]:
    """使用矩形初始化 GrabCut，返回柔化背景后的图片和二值掩膜。"""
    x, y, width, height = (int(value) for value in rectangle)
    if width < 2 or height < 2:
        raise ValueError("框选区域太小，请重新框选目标建筑。")

    # GrabCut 要求初始化矩形不能超出图片边界。
    image_height, image_width = image.shape[:2]
    x = max(0, min(x, image_width - 2))
    y = max(0, min(y, image_height - 2))
    width = max(1, min(width, image_width - x - 1))
    height = max(1, min(height, image_height - y - 1))

    labels = np.zeros((image_height, image_width), dtype=np.uint8)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        image,
        labels,
        (x, y, width, height),
        background_model,
        foreground_model,
        5,
        cv2.GC_INIT_WITH_RECT,
    )
    mask = np.where(
        (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    if cv2.countNonZero(mask) == 0:
        raise ValueError("GrabCut 未找到前景，请扩大框选范围后重试。")

    # 柔化边界，避免硬切割边缘产生大量虚假的 SuperPoint 特征。
    alpha = cv2.GaussianBlur(mask, (0, 0), 2.0).astype(np.float32) / 255.0
    segmented = (
        image.astype(np.float32) * alpha[..., None]
    ).clip(0, 255).astype(np.uint8)
    return segmented, mask


def _get_lightglue_engine():
    """延迟加载模型，避免程序启动时占用显存。"""
    global _LIGHTGLUE_ENGINE  # pylint: disable=global-statement
    if _LIGHTGLUE_ENGINE is not None:
        return _LIGHTGLUE_ENGINE

    try:
        import torch
        from lightglue import LightGlue, SuperPoint
    except ImportError as error:
        raise RuntimeError(
            "缺少 LightGlue 依赖，请先执行：\n"
            "pip install -r test_photo_matching/requirements.txt"
        ) from error

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = SuperPoint(max_num_keypoints=MAX_KEYPOINTS).eval().to(device)
    matcher = (
        LightGlue(features="superpoint", filter_threshold=0.0).eval().to(device)
    )
    _LIGHTGLUE_ENGINE = (torch, extractor, matcher, device)
    return _LIGHTGLUE_ENGINE


def _image_tensor(image: np.ndarray, torch, device):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    contiguous = np.ascontiguousarray(rgb)
    return (
        torch.from_numpy(contiguous)
        .permute(2, 0, 1)
        .float()
        .div_(255.0)
        .to(device)
    )


def match_lightglue(image_a: np.ndarray, image_b: np.ndarray):
    """返回 OpenCV 风格关键点、匹配、置信度和推理设备。"""
    torch, extractor, matcher, device = _get_lightglue_engine()
    tensor_a = _image_tensor(image_a, torch, device)
    tensor_b = _image_tensor(image_b, torch, device)

    with torch.inference_mode():
        features_a = extractor.extract(tensor_a)
        features_b = extractor.extract(tensor_b)
        result = matcher({"image0": features_a, "image1": features_b})

    points_a = features_a["keypoints"][0].detach().cpu().numpy()
    points_b = features_b["keypoints"][0].detach().cpu().numpy()
    pairs = result["matches"][0].detach().cpu().numpy()
    scores = result["scores"][0].detach().cpu().numpy()

    keypoints_a = [
        cv2.KeyPoint(float(point[0]), float(point[1]), 5.0) for point in points_a
    ]
    keypoints_b = [
        cv2.KeyPoint(float(point[0]), float(point[1]), 5.0) for point in points_b
    ]
    matches = [
        cv2.DMatch(int(pair[0]), int(pair[1]), float(1.0 - score))
        for pair, score in zip(pairs, scores)
    ]
    return keypoints_a, keypoints_b, matches, scores, str(device)


def side_by_side(image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
    target_height = max(image_a.shape[0], image_b.shape[0])

    def pad(image):
        bottom = target_height - image.shape[0]
        return cv2.copyMakeBorder(
            image, 0, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(24, 27, 35)
        )

    return np.hstack((pad(image_a), pad(image_b)))


def render_result(
    image_a: np.ndarray,
    keypoints_a,
    image_b: np.ndarray,
    keypoints_b,
    matches,
    show_matches: bool,
    show_keypoints: bool,
) -> np.ndarray:
    if show_matches:
        canvas = side_by_side(image_a, image_b)
        offset_x = image_a.shape[1]
        colors = (
            (0, 255, 255),
            (0, 165, 255),
            (80, 220, 80),
            (255, 120, 40),
            (255, 80, 220),
            (80, 80, 255),
            (255, 220, 70),
            (80, 255, 200),
            (210, 120, 255),
            (255, 255, 255),
        )

        points = []
        for index, match in enumerate(matches):
            point_a = tuple(
                round(value) for value in keypoints_a[match.queryIdx].pt
            )
            raw_point_b = keypoints_b[match.trainIdx].pt
            point_b = (round(raw_point_b[0]) + offset_x, round(raw_point_b[1]))
            points.append((index + 1, point_a, point_b, colors[index % len(colors)]))

        # 先绘制带黑色描边的粗线，确保亮色或复杂背景上仍然清晰。
        for _number, point_a, point_b, color in points:
            cv2.line(canvas, point_a, point_b, (0, 0, 0), 7, cv2.LINE_AA)
            cv2.line(canvas, point_a, point_b, color, 3, cv2.LINE_AA)

        if show_keypoints:
            for number, point_a, point_b, color in points:
                for point in (point_a, point_b):
                    cv2.circle(canvas, point, 12, (0, 0, 0), -1, cv2.LINE_AA)
                    cv2.circle(canvas, point, 9, color, -1, cv2.LINE_AA)
                    label_point = (point[0] + 13, point[1] - 10)
                    cv2.putText(
                        canvas,
                        str(number),
                        label_point,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 0),
                        5,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        canvas,
                        str(number),
                        label_point,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
        return canvas

    if show_keypoints:
        flags = cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
        view_a = cv2.drawKeypoints(
            image_a, keypoints_a, None, color=(67, 211, 158), flags=flags
        )
        view_b = cv2.drawKeypoints(
            image_b, keypoints_b, None, color=(250, 183, 70), flags=flags
        )
        return side_by_side(view_a, view_b)

    return side_by_side(image_a, image_b)


class ImageCanvas(QLabel):
    def __init__(self):
        super().__init__("请先上传 Image A 和 Image B")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(760, 480)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "background: #181b23; color: #8f96a8; border: 1px solid #343949;"
        )
        self._pixmap = None

    def set_bgr_image(self, image: np.ndarray):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimage = QImage(
            rgb.data, width, height, channels * width, QImage.Format.Format_RGB888
        ).copy()
        self._pixmap = QPixmap.fromImage(qimage)
        self._fit_pixmap()

    def _fit_pixmap(self):
        if self._pixmap is None:
            return
        self.setPixmap(
            self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_pixmap()


class PhotoMatchingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LightGlue 图片匹配可视化")
        self.resize(1280, 820)

        self.images = [None, None]
        self.paths = [None, None]
        self.keypoints = [[], []]
        self.segmented_photo = None
        self.grabcut_mask = None
        self._all_matches = []
        self._match_scores = np.empty(0, dtype=np.float32)
        self._inference_device = ""
        self._match_signature = None
        self._sampled_matches = []

        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        title = QLabel("图片特征匹配测试")
        title.setObjectName("title")
        subtitle = QLabel("仅使用两张已导出的图片，不读取点云或项目数据")
        subtitle.setObjectName("subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        image_bar = QHBoxLayout()
        self.path_labels = []
        self.segment_button = None
        for index, name in enumerate(("Image A", "Image B")):
            card = QFrame()
            card.setObjectName("card")
            card_layout = QVBoxLayout(card)
            button = QPushButton(f"上传 {name}")
            button.clicked.connect(
                lambda _checked=False, i=index: self._choose_image(i)
            )
            path_label = QLabel("尚未选择图片")
            path_label.setObjectName("pathLabel")
            path_label.setWordWrap(True)
            self.path_labels.append(path_label)
            card_layout.addWidget(button)
            if index == 0:
                self.segment_button = QPushButton("框选目标建筑（GrabCut）")
                self.segment_button.setEnabled(False)
                self.segment_button.setToolTip(
                    "在弹出的窗口中拖动矩形，按 Enter 或 Space 确认，Esc 取消"
                )
                self.segment_button.clicked.connect(self._select_target_building)
                card_layout.addWidget(self.segment_button)
            card_layout.addWidget(path_label)
            image_bar.addWidget(card)
        layout.addLayout(image_bar)

        controls = QFrame()
        controls.setObjectName("card")
        controls_layout = QHBoxLayout(controls)

        controls_layout.addWidget(QLabel("特征算法"))
        self.algorithm = QComboBox()
        self.algorithm.addItem("SuperPoint + LightGlue")
        controls_layout.addWidget(self.algorithm)
        self.use_grabcut = QCheckBox("使用 GrabCut")
        self.use_grabcut.setEnabled(False)
        controls_layout.addWidget(self.use_grabcut)

        controls_layout.addSpacing(18)
        controls_layout.addWidget(QLabel("匹配置信度 ≥"))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(10)
        self.threshold_slider.setMinimumWidth(180)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(0.10)
        controls_layout.addWidget(self.threshold_slider)
        controls_layout.addWidget(self.threshold_spin)

        self.show_matches = QCheckBox("显示 matches")
        self.show_matches.setChecked(True)
        self.show_keypoints = QCheckBox("显示 keypoints")
        self.show_keypoints.setChecked(True)
        controls_layout.addSpacing(18)
        controls_layout.addWidget(self.show_matches)
        controls_layout.addWidget(self.show_keypoints)
        self.resample_button = QPushButton("随机换 10 个")
        controls_layout.addWidget(self.resample_button)
        controls_layout.addStretch()
        layout.addWidget(controls)

        self.canvas = ImageCanvas()
        layout.addWidget(self.canvas, 1)
        self.status = QLabel("等待上传两张图片")
        self.status.setObjectName("status")
        layout.addWidget(self.status)
        self.setCentralWidget(root)

        self.threshold_slider.valueChanged.connect(self._slider_changed)
        self.threshold_spin.valueChanged.connect(self._spin_changed)
        self.show_matches.toggled.connect(self._refresh)
        self.show_keypoints.toggled.connect(self._refresh)
        self.resample_button.clicked.connect(self._resample)
        self.use_grabcut.toggled.connect(self._grabcut_toggled)

    def _apply_style(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #101219;
                color: #e8eaf0;
                font-family: "Segoe UI", "Microsoft YaHei";
                font-size: 14px;
            }
            QLabel#title { font-size: 24px; font-weight: 700; }
            QLabel#subtitle, QLabel#pathLabel { color: #9299aa; }
            QLabel#status { color: #aeb5c5; padding: 2px; }
            QFrame#card {
                background: #1b1e27;
                border: 1px solid #303543;
                border-radius: 8px;
            }
            QPushButton {
                background: #3467eb;
                border: none;
                border-radius: 6px;
                padding: 9px 18px;
                font-weight: 600;
            }
            QPushButton:hover { background: #4778f2; }
            QComboBox, QDoubleSpinBox {
                background: #262a36;
                border: 1px solid #424858;
                border-radius: 5px;
                padding: 5px 9px;
            }
            QCheckBox { spacing: 7px; }
            """
        )

    def _choose_image(self, index: int):
        path, _ = QFileDialog.getOpenFileName(
            self, f"选择 Image {'A' if index == 0 else 'B'}", "", SUPPORTED_IMAGES
        )
        if not path:
            return
        try:
            image = resize_for_processing(read_image(path))
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "读取失败", str(error))
            return

        self.images[index] = image
        self.paths[index] = path
        self.keypoints = [[], []]
        self._all_matches = []
        self._match_scores = np.empty(0, dtype=np.float32)
        self._inference_device = ""
        self._match_signature = None
        if index == 0:
            self.segmented_photo = None
            self.grabcut_mask = None
            self.segment_button.setEnabled(True)
            self.segment_button.setText("框选目标建筑（GrabCut）")
            self.use_grabcut.blockSignals(True)
            self.use_grabcut.setChecked(False)
            self.use_grabcut.setEnabled(False)
            self.use_grabcut.blockSignals(False)
        self.path_labels[index].setText(
            f"{Path(path).name}\n"
            f"{image.shape[1]} × {image.shape[0]}，等待 LightGlue 处理"
        )

        if all(item is not None for item in self.images):
            self._run_matching()
            return
        self._refresh()

    def _select_target_building(self):
        if self.images[0] is None:
            return

        window_name = "Select target building - Enter/Space: OK, Esc: Cancel"
        self.status.setText("请在弹出窗口中框选目标建筑，然后按 Enter 确认")
        QApplication.processEvents()
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        image_height, image_width = self.images[0].shape[:2]
        scale = min(1.0, 1100 / image_width, 750 / image_height)
        cv2.resizeWindow(
            window_name,
            max(320, round(image_width * scale)),
            max(240, round(image_height * scale)),
        )
        try:
            rectangle = cv2.selectROI(
                window_name,
                self.images[0],
                showCrosshair=True,
                fromCenter=False,
            )
        finally:
            cv2.destroyWindow(window_name)
        if rectangle[2] == 0 or rectangle[3] == 0:
            self.status.setText("已取消框选")
            return

        self.status.setText("正在执行 GrabCut 分割……")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            self.segmented_photo, self.grabcut_mask = grabcut_foreground(
                self.images[0], rectangle
            )
        except (ValueError, cv2.error) as error:
            QMessageBox.critical(self, "GrabCut 分割失败", str(error))
            self.status.setText("GrabCut 分割失败")
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.segment_button.setText("重新框选目标建筑")
        self.use_grabcut.blockSignals(True)
        self.use_grabcut.setEnabled(True)
        self.use_grabcut.setChecked(True)
        self.use_grabcut.blockSignals(False)
        if self.images[1] is not None:
            self._run_matching()
        else:
            self.canvas.set_bgr_image(self.segmented_photo)
            self.status.setText("GrabCut 分割完成，请上传 Image B")

    def _grabcut_toggled(self, checked: bool):
        if checked and self.segmented_photo is None:
            return
        self._match_signature = None
        if all(item is not None for item in self.images):
            self._run_matching()
        else:
            self._refresh()

    def _run_matching(self):
        image_a = (
            self.segmented_photo
            if self.use_grabcut.isChecked() and self.segmented_photo is not None
            else self.images[0]
        )
        mode = "GrabCut + LightGlue" if image_a is self.segmented_photo else "LightGlue"
        self.status.setText(f"正在运行 {mode}，请稍候……")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            (
                self.keypoints[0],
                self.keypoints[1],
                self._all_matches,
                self._match_scores,
                self._inference_device,
            ) = match_lightglue(image_a, self.images[1])
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            self.status.setText("LightGlue 匹配失败")
            QMessageBox.critical(self, "LightGlue 匹配失败", str(error))
            return
        finally:
            QApplication.restoreOverrideCursor()

        self._match_signature = None
        for item_index in range(2):
            current = self.images[item_index]
            suffix = (
                "，已启用 GrabCut"
                if item_index == 0 and self.use_grabcut.isChecked()
                else ""
            )
            self.path_labels[item_index].setText(
                f"{Path(self.paths[item_index]).name}\n"
                f"{current.shape[1]} × {current.shape[0]}，"
                f"{len(self.keypoints[item_index])} 个 SuperPoint 关键点{suffix}"
            )
        self._refresh()

    def _slider_changed(self, value: int):
        self.threshold_spin.blockSignals(True)
        self.threshold_spin.setValue(value / 100)
        self.threshold_spin.blockSignals(False)
        self._refresh()

    def _spin_changed(self, value: float):
        self.threshold_slider.blockSignals(True)
        self.threshold_slider.setValue(round(value * 100))
        self.threshold_slider.blockSignals(False)
        self._refresh()

    def _resample(self):
        self._match_signature = None
        self._refresh()

    def _refresh(self):
        if any(image is None for image in self.images):
            loaded = sum(image is not None for image in self.images)
            self.status.setText(f"已上传 {loaded}/2 张图片")
            if loaded == 1:
                index = 0 if self.images[0] is not None else 1
                source = (
                    self.segmented_photo
                    if index == 0
                    and self.use_grabcut.isChecked()
                    and self.segmented_photo is not None
                    else self.images[index]
                )
                preview = cv2.drawKeypoints(
                    source,
                    self.keypoints[index],
                    None,
                    color=(67, 211, 158),
                )
                self.canvas.set_bgr_image(preview)
            return

        threshold = self.threshold_spin.value()
        matches = [
            match
            for match, score in zip(self._all_matches, self._match_scores)
            if float(score) >= threshold
        ]
        signature = (
            self.paths[0],
            self.paths[1],
            round(threshold, 2),
            self.use_grabcut.isChecked(),
        )
        if signature != self._match_signature:
            self._match_signature = signature
            sample_size = min(10, len(matches))
            self._sampled_matches = random.sample(matches, sample_size)
        rendered = render_result(
            (
                self.segmented_photo
                if self.use_grabcut.isChecked()
                and self.segmented_photo is not None
                else self.images[0]
            ),
            self.keypoints[0],
            self.images[1],
            self.keypoints[1],
            self._sampled_matches,
            self.show_matches.isChecked(),
            self.show_keypoints.isChecked(),
        )
        self.canvas.set_bgr_image(rendered)
        preprocessing = "GrabCut + " if self.use_grabcut.isChecked() else ""
        self.status.setText(
            f"{preprocessing}SuperPoint + LightGlue ({self._inference_device}) · "
            f"置信度 ≥ {threshold:.2f} · "
            f"A: {len(self.keypoints[0])} keypoints · "
            f"B: {len(self.keypoints[1])} keypoints · "
            f"共 {len(matches)} matches，随机显示 {len(self._sampled_matches)} 个"
        )


def main():
    app = QApplication(sys.argv)
    window = PhotoMatchingWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
