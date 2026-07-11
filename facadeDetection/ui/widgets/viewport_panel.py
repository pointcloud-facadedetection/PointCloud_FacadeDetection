"""Open3D viewport shell and visual overlays."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class ViewportPanel(QFrame):
    """Hosts the native Open3D surface and the prototype's overlay widgets."""

    tool_requested = Signal(str)

    def __init__(self, viewport_widget: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("viewportPanel")
        self.setMinimumSize(640, 480)
        self._viewport_widget = viewport_widget

        self.badge = self._build_badge()
        self.toolbar = self._build_toolbar()
        self.info_card = self._build_info_card()
        self.bottom_hint = self._build_bottom_hint()

        # A GLFW HWND always paints above non-native Qt siblings on Windows.
        # Keep the prototype controls around the native surface instead of
        # pretending they can be composited over it; this preserves every
        # control and keeps input reliable.
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)
        top = QHBoxLayout()
        top.setSpacing(10)
        top.addWidget(self.badge)
        top.addStretch(1)
        top.addWidget(self.toolbar)
        root.addLayout(top)

        center = QHBoxLayout()
        center.setSpacing(10)
        center.addWidget(viewport_widget, 1)
        center.addWidget(self.info_card, 0, Qt.AlignmentFlag.AlignTop)
        root.addLayout(center, 1)
        root.addWidget(self.bottom_hint)

    def _build_badge(self) -> QFrame:
        badge = QFrame(self)
        badge.setObjectName("viewportBadge")
        row = QHBoxLayout(badge)
        row.setContentsMargins(12, 7, 12, 7)
        row.setSpacing(7)
        self.ready_dot = QLabel("●")
        self.ready_dot.setObjectName("readyDot")
        self.viewport_title = QLabel("OPEN3D VIEWPORT · 正在初始化")
        self.viewport_title.setObjectName("viewportTitle")
        row.addWidget(self.ready_dot)
        row.addWidget(self.viewport_title)
        return badge

    def _build_toolbar(self) -> QFrame:
        toolbar = QFrame(self)
        toolbar.setObjectName("viewportToolbar")
        row = QHBoxLayout(toolbar)
        row.setContentsMargins(7, 6, 7, 6)
        row.setSpacing(5)
        definitions = (
            ("旋转", "rotate", "旋转视图", True),
            ("平移", "pan", "平移视图", True),
            ("缩放", "zoom", "缩放视图", True),
            ("网格", "grid", "显示或隐藏参考网格", True),
            ("坐标", "axes", "显示或隐藏坐标轴", True),
            ("截图", "screenshot", "保存当前视口截图", False),
            ("全屏", "fullscreen", "切换全屏", False),
        )
        self.tool_buttons: dict[str, QToolButton] = {}
        self.navigation_group = QButtonGroup(toolbar)
        self.navigation_group.setExclusive(True)
        for text, action, tooltip, checkable in definitions:
            button = QToolButton()
            button.setText(text)
            button.setToolTip(tooltip)
            button.setCheckable(checkable)
            button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            if action in {"rotate", "pan", "zoom"}:
                self.navigation_group.addButton(button)
            if action in {"rotate", "grid"}:
                button.setChecked(True)
            button.clicked.connect(lambda checked=False, name=action: self.tool_requested.emit(name))
            row.addWidget(button)
            self.tool_buttons[action] = button
        return toolbar

    def _build_info_card(self) -> QFrame:
        card = QFrame(self)
        card.setObjectName("sceneCard")
        card.setFixedWidth(255)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(13, 12, 13, 12)
        layout.setSpacing(8)
        title = QLabel("场景信息")
        title.setObjectName("sceneTitle")
        layout.addWidget(title)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(7)
        self.info_values: dict[str, QLabel] = {}
        for key, caption in (
            ("filename", "文件"),
            ("points", "点数"),
            ("bbox", "边界框"),
            ("center", "中心"),
            ("range", "场景范围"),
            ("facades", "检测立面"),
        ):
            key_label = QLabel(caption)
            key_label.setObjectName("sceneKey")
            value_label = QLabel("-")
            value_label.setObjectName("sceneValue")
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            form.addRow(key_label, value_label)
            self.info_values[key] = value_label
        layout.addLayout(form)
        return card

    def _build_bottom_hint(self) -> QFrame:
        frame = QFrame(self)
        frame.setObjectName("bottomHint")
        row = QHBoxLayout(frame)
        row.setContentsMargins(11, 6, 11, 6)
        self.fps_label = QLabel("Open3D")
        self.fps_label.setObjectName("viewportTitle")
        self.hint_label = QLabel("鼠标左键旋转  |  Shift + 左键平移  |  滚轮缩放")
        self.hint_label.setObjectName("viewportHint")
        row.addWidget(self.fps_label)
        row.addStretch(1)
        row.addWidget(self.hint_label)
        return frame

    def raise_overlays(self) -> None:
        """Compatibility hook; controls are laid out outside the native HWND."""
        self.badge.raise_()
        self.toolbar.raise_()
        self.info_card.raise_()
        self.bottom_hint.raise_()

    def set_viewport_ready(self, ready: bool, message: str) -> None:
        self.ready_dot.setStyleSheet(f"color: {'#48bb78' if ready else '#ed8936'};")
        state = "已就绪" if ready else "降级显示"
        self.viewport_title.setText(f"OPEN3D VIEWPORT · {state}")
        self.viewport_title.setToolTip(message)
        self.fps_label.setText("Open3D · READY" if ready else "Open3D · FALLBACK")

    def set_scene_info(
        self,
        *,
        filename: str,
        points: int,
        bbox: str,
        center: str,
        scene_range: str,
        facades: str = "-",
    ) -> None:
        values = {
            "filename": filename,
            "points": f"{points:,}",
            "bbox": bbox,
            "center": center,
            "range": scene_range,
            "facades": facades,
        }
        for key, value in values.items():
            self.info_values[key].setText(value)
            self.info_values[key].setToolTip(value)

    def clear_scene_info(self) -> None:
        for value in self.info_values.values():
            value.setText("-")

    def set_hint(self, text: str) -> None:
        self.hint_label.setText(text)
