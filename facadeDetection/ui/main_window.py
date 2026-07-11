"""Main PySide6 window for the first UI integration milestone."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import QHBoxLayout, QMainWindow, QMessageBox, QStatusBar, QWidget

try:  # Support both direct-file and package execution.
    from ..view3d.open3d_viewport import Open3DViewport
except ImportError:
    from view3d.open3d_viewport import Open3DViewport

from .theme import APP_STYLESHEET
from .widgets.control_panel import ControlPanel
from .widgets.viewport_panel import ViewportPanel


class MainWindow(QMainWindow):
    """Owns layout and controls; service and algorithm actions stay reserved."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PointCloud FacadeDetection · 点云立面检测系统")
        self.resize(1600, 900)
        self.setMinimumSize(1120, 700)
        self.setStyleSheet(APP_STYLESHEET)
        self._fullscreen = False

        self.viewport = Open3DViewport()
        self._setup_ui()
        self._connect_signals()

        # Owning the single-shot timer lets closeEvent cancel a pending start.
        self._startup_timer = QTimer(self)
        self._startup_timer.setSingleShot(True)
        self._startup_timer.timeout.connect(self.viewport.start)
        self._startup_timer.start(0)

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.control_panel = ControlPanel()
        self.viewport_panel = ViewportPanel(self.viewport.get_widget())
        layout.addWidget(self.control_panel)
        layout.addWidget(self.viewport_panel, 1)

        self._build_menu()
        status = QStatusBar()
        status.setSizeGripEnabled(False)
        self.setStatusBar(status)
        self._set_status("界面已加载，正在启动 Open3D 视口…")

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("文件")
        open_action = QAction("选择点云文件…", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.control_panel._choose_files)
        file_menu.addAction(open_action)

        screenshot_action = QAction("视口截图（接口预留）", self)
        screenshot_action.triggered.connect(lambda: self._handle_tool("screenshot"))
        file_menu.addAction(screenshot_action)
        file_menu.addSeparator()

        exit_action = QAction("退出", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        view_menu = self.menuBar().addMenu("视图")
        reset_action = QAction("重置视角", self)
        reset_action.setShortcut(QKeySequence("0"))
        reset_action.triggered.connect(lambda: self._handle_action("reset_view"))
        view_menu.addAction(reset_action)

        fullscreen_action = QAction("切换全屏", self)
        fullscreen_action.setShortcut(QKeySequence("F11"))
        fullscreen_action.triggered.connect(lambda: self._handle_tool("fullscreen"))
        view_menu.addAction(fullscreen_action)

        help_menu = self.menuBar().addMenu("帮助")
        about_action = QAction("关于本项目", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _connect_signals(self) -> None:
        self.control_panel.files_selected.connect(self._files_selected)
        self.control_panel.action_requested.connect(self._handle_action)
        self.control_panel.value_changed.connect(self._handle_value_change)
        self.control_panel.color_changed.connect(
            lambda color: self._set_status(f"已选择颜色 {color}；应用接口等待 services 接入", 4000)
        )
        self.viewport_panel.tool_requested.connect(self._handle_tool)
        self.viewport.signals.ready.connect(self._on_viewport_ready)

    def _on_viewport_ready(self, ready: bool, message: str) -> None:
        self.viewport_panel.set_viewport_ready(ready, message)
        self._set_status(message, 8000)

    def _files_selected(self, paths: list[str]) -> None:
        names = "、".join(Path(path).name for path in paths[:3])
        suffix = "…" if len(paths) > 3 else ""
        self._set_status(
            f"已选择 {len(paths)} 个文件（{names}{suffix}）；加载接口等待 view3d / services 接入",
            7000,
        )

    def _handle_value_change(self, key: str, value: float) -> None:
        labels = {"voxel_size": "体素大小", "point_size": "点大小", "min_area": "最小立面面积"}
        self._set_status(f"{labels.get(key, key)}设为 {value:g}；参数接口已预留", 2500)

    def _handle_action(self, action: str) -> None:
        if action == "reset_view":
            self.viewport.reset_view()
            self._set_status("Open3D 参考视角已重置", 3000)
            return

        labels = {
            "denoise": "去噪处理",
            "normals": "法向量计算",
            "bbox": "边界框显示",
            "clear": "清空场景",
            "detect_facades": "建筑立面检测",
            "reset_facades": "立面显示恢复",
            "quality_report": "质量评估报告",
            "selection_mode": "框选模式",
            "segment_selection": "框选区域分割",
            "reset_segments": "分割颜色恢复",
            "apply_color": "应用颜色",
            "reset_color": "原始颜色恢复",
            "set_target": "设置目标点云",
            "set_source": "设置源点云",
            "registration_mode": "手工配准模式",
            "exit_registration": "退出配准模式",
            "add_correspondence": "添加对应点",
            "undo_correspondence": "撤销对应点",
            "execute_registration": "粗配准",
            "icp_refine": "ICP 精配准",
            "save_registration": "保存配准结果",
        }
        self._set_status(f"“{labels.get(action, action)}”控件已预留，等待 services / 算法模块接入", 6000)

    def _handle_tool(self, tool: str) -> None:
        if tool == "fullscreen":
            self._fullscreen = not self._fullscreen
            self.showFullScreen() if self._fullscreen else self.showNormal()
            return
        label = self.viewport_panel.tool_buttons[tool].text()
        self._set_status(f"“{label}”视口操作接口已预留，等待 view3d 交互实现", 5000)

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "关于点云立面检测系统",
            "PointCloud FacadeDetection\n\n"
            "当前里程碑：PySide6 窗口、Open3D 嵌入式视口、UI 布局和控件占位。\n"
            "点云交互、算法、services 与数据库按团队分工后续接入。",
        )

    def _set_status(self, message: str, timeout: int = 0) -> None:
        self.statusBar().showMessage(message, timeout)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self._startup_timer.stop()
        self.viewport.shutdown()
        event.accept()
