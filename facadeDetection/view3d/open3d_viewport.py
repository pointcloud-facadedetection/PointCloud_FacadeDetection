import numpy as np
from PySide6.QtCore import QEvent, QObject, QTimer, Qt, QPoint, Slot, Signal
from PySide6.QtGui import QImage, QWindow
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .base_viewport import BaseViewport
from .camera import CameraController
from .geometry_factory import (
    make_bbox, make_normals, make_pair_lines, make_sphere,
)
from .interaction import ViewportInteractor
from .open3d_adapter import Open3DAdapter
from .roi_selection import ROISelectionController
from .pick_overlay import PointPickOverlay
from .scene import PointCloudScene
from .window_embed import NativeWindowFinder


class _ContainerEventBridge(QObject):
    """Qt 事件桥接器：统一事件路径，仅在未锁定时转发到 interactor。"""
    def __init__(self, interactor, overlay=None, parent=None):
        super().__init__(parent)
        self.interactor = interactor
        self._overlay = overlay

    def eventFilter(self, watched, event):
        try:
            # ROI 硬锁：让事件流向 Overlay（Overlay 已 grabMouse/grabKeyboard）
            if getattr(self.interactor, 'input_locked', False):
                return False
            et = event.type()
            handled = False
            if et == QEvent.MouseButtonPress:
                handled = bool(self.interactor.handle_mouse_press(event))
            elif et == QEvent.MouseMove:
                handled = bool(self.interactor.handle_mouse_move(event))
            elif et == QEvent.MouseButtonRelease:
                handled = bool(self.interactor.handle_mouse_release(event))
            elif et == QEvent.Wheel:
                handled = bool(self.interactor.handle_wheel(event))
            if self._overlay is not None:
                try:
                    self._overlay.update()
                except Exception:
                    pass
            # 若已处理则返回 True，避免事件继续传递到原生窗口
            return handled
        except Exception:
            return super().eventFilter(watched, event)
# 直接在 QWindow 上安装事件桥，确保原生窗口的鼠标事件能送达到 interactor。
class _QWindowEventBridge(QObject):
    def __init__(self, interactor, overlay_getter=None, parent=None):
        super().__init__(parent)
        self._interactor = interactor
        self._overlay_getter = overlay_getter  # callable -> overlay or None

    def _update_overlay(self):
        try:
            if callable(self._overlay_getter):
                ov = self._overlay_getter()
                if ov is not None:
                    ov.update()
        except Exception:
            pass

    def eventFilter(self, watched, event):
        try:
            # ROI 硬锁时：不要拦截，交由 Overlay（顶层 Tool 窗口）处理
            if getattr(self._interactor, 'input_locked', False):
                return False
            et = event.type()
            handled = False
            from PySide6.QtCore import QEvent as _QE
            if et == _QE.MouseButtonPress:
                handled = bool(self._interactor.handle_mouse_press(event))
            elif et == _QE.MouseMove:
                handled = bool(self._interactor.handle_mouse_move(event))
            elif et == _QE.MouseButtonRelease:
                handled = bool(self._interactor.handle_mouse_release(event))
            elif et == _QE.Wheel:
                handled = bool(self._interactor.handle_wheel(event))
            if handled:
                self._update_overlay()
                return True
        except Exception:
            pass
        return False


class _RenderQueue(QObject):
    color = Signal(str, object)
    points = Signal(str, object, object)

# TODO(Open3DViewport): [视口渲染刷新] 当前定时器持续驱动视口处理；应按视口可见性/焦点启停，结合 dirty 状态和渲染节流，避免 Windows 原生窗口抢占其他程序画面。
class Open3DViewport(BaseViewport):
    def __init__(self, parent=None):
        self._root = QWidget(parent)
        self._root.setObjectName('open3dViewport')
        self._root.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._layout = QVBoxLayout(self._root)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self._adapter = Open3DAdapter()
        self._scene = PointCloudScene(self._adapter)
        self._camera = CameraController(self._adapter)
        self._interactor = ViewportInteractor(self._adapter, self._camera, self._scene)
        self._window_finder = NativeWindowFinder()

        self._container = None
        self._qwindow = None
        self._fallback_label = None
        self._event_bridge = None
        self._window_title = "PointCloud FacadeDetection"
        self._pick_markers = []
        self._pick_lines = []
        self._pick_marker_xyz = np.zeros((0, 3), dtype=np.float64)
        self._init_success = False
        self._overlay = None
        self._roi_controller = None
        # 交互模式
        class InteractionMode:
            NAVIGATE = 'navigate'
            PICK = 'pick'
            ROI = 'roi'
        self.InteractionMode = InteractionMode
        self._mode = InteractionMode.NAVIGATE

        # ROI 选择模式状态
        self._roi_on_complete = None
        self._scene_view_initialized = False
        self._destroyed = False
        self._pending_point_updates = {}
        self._pending_color_updates = {}
        self._render_sequences = {}

        self.native = self
        self._render_queue = _RenderQueue(self._root)
        self._render_queue.color.connect(self._queue_cloud_color, Qt.QueuedConnection)
        self._render_queue.points.connect(self._queue_cloud_points, Qt.QueuedConnection)

        self._init_ui()

        self._timer = QTimer(self._root)
        self._timer.timeout.connect(self.process_events)
        # 该定时器会触发 GLFW 事件；由适配器决定是否实际提交帧
        self._timer.start(33)

    def queue_update_cloud_color(self, name, colors):
        """Thread-safe color update; Open3D is touched only by the Qt GUI thread."""
        self._render_queue.color.emit(name, colors)

    def queue_update_cloud_points(self, name, positions, colors=None):
        """Thread-safe geometry replacement for worker completion callbacks."""
        self._render_queue.points.emit(name, positions, colors)

    def invalidate_render_queue(self):
        """Invalidate queued worker updates before replacing the scene.

        A queued Qt signal can outlive the geometry that originated it.  The
        sequence bump makes such an update harmless even when it is delivered
        after a station switch.
        """
        self._pending_point_updates.clear()
        self._pending_color_updates.clear()
        self._render_sequences = {
            name: sequence + 1
            for name, sequence in self._render_sequences.items()
        }
        self._adapter.request_render('queue.invalidated')

    def _queue_cloud_color(self, name, colors):
        sequence = self._render_sequences.get(name, 0) + 1
        self._render_sequences[name] = sequence
        old = self._pending_color_updates.get(name)
        if old is not None:
            del old
        self._pending_color_updates[name] = (sequence, colors)
        self._adapter.request_render('color.queued')

    def _queue_cloud_points(self, name, positions, colors=None):
        # 仅保留最新的工作进程快照。
        sequence = self._render_sequences.get(name, 0) + 1
        self._render_sequences[name] = sequence
        old = self._pending_point_updates.get(name)
        if old is not None:
            del old
        self._pending_point_updates[name] = (sequence, positions, colors)
        self._adapter.request_render('points.queued')

    @property
    def _clouds(self):
        return self._scene.clouds

    @property
    def _point_data(self):
        return self._scene.point_data

    @property
    def _bbox_visible(self):
        return self._scene.bbox_visible

    @property
    def _active_name(self):
        return self._scene.active_name

    @property
    def _vis(self):
        return self._adapter.vis

    @property
    def point_pick_callback(self):
        return self._interactor.point_pick_callback

    @point_pick_callback.setter
    def point_pick_callback(self, callback):
        self._interactor.point_pick_callback = callback

    @property
    def selection_callback(self):
        return self._interactor.selection_callback

    @selection_callback.setter
    def selection_callback(self, callback):
        self._interactor.selection_callback = callback

    @property
    def pick_enabled(self):
        return self._interactor.pick_enabled

    @pick_enabled.setter
    def pick_enabled(self, value):
        self._interactor.pick_enabled = bool(value)

    @property
    def selection_enabled(self):
        return self._interactor.selection_enabled

    @selection_enabled.setter
    def selection_enabled(self, value):
        self._interactor.selection_enabled = bool(value)

    def _init_ui(self):
        layout = self._layout
        try:
            self._adapter.create_window(self._window_title, width=1280, height=960, visible=True)
            handle = self._window_finder.find(
                self._window_title,
                timeout=120.0,
                process_events=self.process_events,
            )
            if handle is None:
                raise RuntimeError("Cannot locate Open3D native window handle")

            self._qwindow = QWindow.fromWinId(int(handle))
            self._container = QWidget.createWindowContainer(self._qwindow, self._root)
            self._container.setFocusPolicy(Qt.StrongFocus)
            self._container.setMouseTracking(True)

            # PICK 使用独立顶层覆盖层；旧的内嵌 _Overlay 会与 GLFW 子窗口竞争输入，
            # 因此不再实例化。ROI 仍由 ROISelectionController 独立接管。
            self._overlay = None
            self._pick_overlay = PointPickOverlay(self, self._container)

            def _sync_overlay_size():
                try:
                    if self._pick_overlay is None or self._container is None:
                        return
                    if not self._container.isVisible():
                        return

                        # 获取容器在屏幕上的精确位置
                    container_global = self._container.mapToGlobal(QPoint(0, 0))
                    self._pick_overlay.setGeometry(
                            container_global.x(),
                            container_global.y(),
                            self._container.width(),
                            self._container.height()
                        )
                    self._pick_overlay.raise_()

                        # 如果处于激活状态，确保焦点
                    if getattr(self._pick_overlay, '_active', False):
                        self._pick_overlay.setFocus(Qt.OtherFocusReason)
                except RuntimeError:
                    pass

            original_resize = self._root.resizeEvent
            def _on_root_resize(event):
                if callable(original_resize):
                    original_resize(event)
                _sync_overlay_size()
            self._root.resizeEvent = _on_root_resize

            QTimer.singleShot(0, _sync_overlay_size)

            self._event_bridge = _ContainerEventBridge(
                self._interactor, overlay=None, parent=self._root
            )
            # 作为兜底安装在 container 上（主要通道由 QWindow 事件桥处理）
            self._container.installEventFilter(self._event_bridge)
            layout.addWidget(self._container)

            self._roi_controller = ROISelectionController(
                self, container_widget=self._container
            )

            # 在 QWindow 上安装事件桥，确保右键平移与拖拽事件送达 interactor
            try:
                def _get_overlay():
                    return self._roi_controller.overlay() if hasattr(self._roi_controller, 'overlay') else self._overlay
                self._qwindow_event_bridge = _QWindowEventBridge(self._interactor, overlay_getter=_get_overlay, parent=self._root)
                self._qwindow.installEventFilter(self._qwindow_event_bridge)
            except Exception:
                pass

            self._camera.viewport_widget = self._container
            # 向相机提供场景比例估计值，以确保投影/标记尺寸的一致性
            try:
                self._camera.set_scene_scale_provider(self._fallback_scene_scale)
            except Exception:
                pass
            self._adapter.configure_render_options()
            self._init_success = True
            # 原生 Open3D 启动时会自带一个摄像机；项目视图
            # 只有在加载完第一个云后才会初始化。

        except Exception as exc:
            import traceback
            error_msg = f"Open3D render initialization failed.\n\n{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            self._fallback_label = QLabel(error_msg)
            self._fallback_label.setAlignment(Qt.AlignCenter)
            self._fallback_label.setStyleSheet(
                "background:#111827;color:#fca5a5;font-size:12px;padding:24px;"
            )
            layout.addWidget(self._fallback_label)
            self._adapter.destroy()
            self._init_success = False

    def _initialize_scene_view(self):
        try:
            ctr = self._adapter.get_view_control()
            if ctr is None:
                return
            # 默认切换为正交投影：FoV=Config.ORTHO_FOV_DEG 触发 Open3D 正交模式
            try:
                current_fov = ctr.get_field_of_view()
                from config.settings import Config
                target = float(getattr(Config, 'ORTHO_FOV_DEG', 5.0))
                if current_fov > (target + 0.5):
                    ctr.change_field_of_view(target - current_fov)
            except Exception:
                pass
            # 对齐至便于展示立面的正投影正面视图
            center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            try:
                pts = []
                for data in self._scene.point_data.values():
                    p = data.get("pos")
                    if p is not None and len(p) > 0:
                        pts.append(np.asarray(p, dtype=np.float64))
                if pts:
                    allp = np.vstack(pts)
                    center = np.mean(allp, axis=0)
            except Exception:
                pass
            # Open3D 在此嵌入式视口中的起始视角为俯视图。将该视图旋转 90 度
            # 使其面向建筑物正面：Y 轴表示深度，Z轴仍保持屏幕向上方向。
            front = np.array([0.0, -1.0, 0.0], dtype=np.float64)
            up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

            ctr.set_lookat(center.astype(np.float64))
            ctr.set_front(front / (np.linalg.norm(front) + 1e-12))
            ctr.set_up(up / (np.linalg.norm(up) + 1e-12))
            # 合理的初始缩放比例
            ctr.set_zoom(0.6)
            self._scene_view_initialized = True
        except Exception:
            pass

    def reset_view(self):
        """恢复建筑立面默认正视图"""
        self._scene_view_initialized = False
        if self._scene.point_data:
            self._initialize_scene_view()

    # ---------------- 统一交互模式 ----------------
    def set_mode(self, mode: str):
        """切换交互模式：navigate/pick/roi，统一事件与锁定策略。"""
        try:
            mode = str(mode).lower()
        except Exception:
            mode = self.InteractionMode.NAVIGATE
        self._mode = mode
        try:
            if mode == self.InteractionMode.ROI:
                # 硬锁输入，仅 Overlay 接收事件
                setattr(self._interactor, 'input_locked', True)
                # 关闭点选/框选（由 Overlay 控制）
                self._interactor.pick_enabled = False
                self._interactor.selection_enabled = False
            elif mode == self.InteractionMode.PICK:
                setattr(self._interactor, 'input_locked', False)
                self._interactor.selection_enabled = False
                # pick_enabled 由 enter_pick_mode 设置细节
            else:  # NAVIGATE
                setattr(self._interactor, 'input_locked', False)
                self._interactor.pick_enabled = False
                self._interactor.selection_enabled = False
        except Exception:
            pass

    def process_events(self):
        if not self._init_success:
            return
        try:
            # TODO(process_events): [视口渲染刷新] 此处将队列处理、GLFW 事件轮询和渲染绑定在同一周期；应按需提交局部刷新，并在失焦/隐藏时暂停渲染、恢复时补帧。
            pending = self._pending_point_updates
            self._pending_point_updates = {}
            for name, (sequence, positions, colors) in pending.items():
                if sequence == self._render_sequences.get(name):
                    self.update_cloud_points(name, positions, colors)
            pending_colors = self._pending_color_updates
            self._pending_color_updates = {}
            for name, (sequence, colors) in pending_colors.items():
                if sequence == self._render_sequences.get(name):
                    self.update_cloud_color(name, colors)
            self._adapter.poll()
        except Exception:
            pass
    def close(self):
        self.destroy()

    def destroy(self):
        """Stop Qt delivery and release the embedded Open3D window once."""
        # TODO(destroy): [项目生命周期] 销毁前应由统一生命周期状态机协调后台任务、视口覆盖层/几何体、渲染队列和 Open3D 资源，确保关闭流程幂等且无残留引用。
        if self._destroyed:
            return
        self._destroyed = True
        self._init_success = False
        if hasattr(self, '_timer') and self._timer is not None:
            self._timer.stop()
        self.invalidate_render_queue()
        self._pending_point_updates.clear()
        self._pending_color_updates.clear()
        self._render_sequences.clear()
        try:
            if self._roi_controller is not None:
                self._roi_controller.cancel()
        except Exception:
            pass
        self._roi_on_complete = None
        for watched, bridge_name in (
            (self._qwindow, '_qwindow_event_bridge'),
            (self._container, '_event_bridge'),
        ):
            bridge = getattr(self, bridge_name, None)
            if watched is not None and bridge is not None:
                try:
                    watched.removeEventFilter(bridge)
                except Exception:
                    pass

        # QWindow::fromWinId 只是原生窗口的 Qt 包装。先解除嵌入关系并隐藏容器，
        # 避免 Qt 在 Visualizer 已销毁后继续向 GLFW 窗口投递
        # 重绘/焦点事件（Windows 下这是关闭警告的主要来源）。
        container = self._container
        qwindow = self._qwindow
        if self._pick_overlay is not None:
            try:
                self._pick_overlay.deactivate()
            except Exception:
                pass
            self._pick_overlay = None
        self._overlay = None
        self._qwindow = None
        self._container = None
        if container is not None:
            try:
                container.removeEventFilter(self._event_bridge)
                container.setParent(None)
                container.hide()
                container.deleteLater()
            except Exception:
                pass
        if qwindow is not None:
            try:
                qwindow.removeEventFilter(getattr(self, '_qwindow_event_bridge', None))
            except Exception:
                pass
            try:
                qwindow.setParent(None)
            except Exception:
                pass

        # destroy_window 必须在创建 Visualizer 的 GUI 线程中、Qt 事件源
        # 停止之后调用；adapter.destroy() 已做幂等保护。
        self._adapter.destroy()
        # fromWinId 返回的包装对象不能在 Visualizer 销毁后继续存活到
        # QApplication 退出，否则 Qt 平台插件可能再次访问 GLFW 句柄。
        if qwindow is not None:
            try:
                qwindow.deleteLater()
            except Exception:
                pass

    def get_widget(self):
        return self._root

    def project_points(self, points):
        """将 3D 点投影为屏幕坐标（逻辑像素）。返回 (screen, valid) 或 None。"""
        try:
            return self._camera.project_points(points)
        except Exception:
            return None

    def get_picked_point(self):
        return self._interactor.last_picked_point

    def apply_scan_pose_view(self, eye, lookat, up, zoom=0.6):
        """按扫描仪位姿设置视口相机。"""
        if not self._init_success:
            return False
        try:
            self._camera.set_look_at(
                np.asarray(lookat, dtype=np.float64),
                np.asarray(eye, dtype=np.float64),
                np.asarray(up, dtype=np.float64),
            )
            control = self._adapter.get_view_control()
            if control is not None and zoom is not None:
                control.set_zoom(float(zoom))
            if self._overlay is not None:
                self._overlay.update()
            return True
        except Exception:
            return False

    def focus_on_plane(self, center, normal, extent=None, zoom=0.5):
        """沿平面法向正对目标中心，并尽量保持当前观察侧。"""
        if not self._init_success:
            return False
        try:
            center = np.asarray(center, dtype=np.float64).reshape(3)
            normal = np.asarray(normal, dtype=np.float64).reshape(3)
            norm = float(np.linalg.norm(normal))
            if not np.isfinite(center).all() or norm < 1e-12:
                return False
            normal /= norm

            current_front, _current_up, _right = self._camera.get_camera_basis()
            if (
                current_front is not None
                and float(np.dot(normal, current_front)) < 0.0
            ):
                normal *= -1.0

            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if abs(float(np.dot(normal, world_up))) > 0.95:
                world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            target_extent = max(float(extent or 0.0), 1.0)
            eye = center + normal * max(2.0, target_extent * 2.0)
            return self.apply_scan_pose_view(
                eye,
                center,
                world_up,
                zoom=zoom,
            )
        except (TypeError, ValueError):
            return False

    # ------------------------------------------------------------------
    # ROI 视觉辅助：2D 红框 + 3D 包围盒
    # ------------------------------------------------------------------
    def clear_roi_visuals(self):
        self._interactor.clear_selection_rect()
        self._adapter.remove_geometry("__roi_selection_bbox")
        if self._pick_overlay is not None:
            self._pick_overlay.update()

    def show_roi_bbox(self, min_bound, max_bound, color=(1.0, 1.0, 1.0)):
        self._adapter.remove_geometry("__roi_selection_bbox")
        self._adapter.add_geometry("__roi_selection_bbox", make_bbox(min_bound, max_bound, color=color), reset_bounding_box=False)
        if self._pick_overlay is not None:
            self._pick_overlay.update()

    def clear_facade_boundary(self):
        self._adapter.remove_geometry("__facade_boundary")
        if self._overlay is not None:
            self._overlay.update()

    def show_facade_boundary(self, corners, color=(1.0, 0.0, 0.0)):
        from view3d.geometry_factory import make_closed_polyline

        self._adapter.remove_geometry("__facade_boundary")
        geometry = make_closed_polyline(corners, color=color)
        if geometry is None:
            return
        self._adapter.add_geometry(
            "__facade_boundary",
            geometry,
            reset_bounding_box=False,
        )
        try:
            self._adapter.vis.get_render_option().line_width = 3.0
        except Exception:
            pass
        if self._overlay is not None:
            self._overlay.update()

    # ------------------------------------------------------------------
    # 高级 ROI 框选模式（建筑外立面检测专用）
    # ------------------------------------------------------------------
    def enter_roi_selection(self, cloud_name=None, on_complete=None):
        """全新 ROI 框选：由 ROISelectionController 接管，硬性锁定视口交互。"""
        try:
            self.clear_roi_visuals()
        except Exception:
            pass
        try:
            # store callback for forwarding
            self._roi_on_complete = on_complete
            if self._roi_controller is not None:
                # 统一切换模式到 ROI
                self.set_mode(self.InteractionMode.ROI)
                self._roi_controller.start(cloud_name, self._handle_roi_complete)
        except Exception:
            pass

    def exit_roi_selection(self):
        try:
            if self._roi_controller is not None:
                self._roi_controller.cancel()
        except Exception:
            pass
        self._roi_on_complete = None
        try:
            # 恢复导航模式
            self.set_mode(self.InteractionMode.NAVIGATE)
        except Exception:
            pass

    def _handle_roi_complete(self, min_bound, max_bound, indices, p1=None, p2=None):
        """如果提供了外部回调函数，则将ROI选择结果转发至该回调函数。"""
        cb = self._roi_on_complete
        if callable(cb):
            try:
                cb(min_bound, max_bound, indices, p1, p2)
            except Exception:
                pass

    def add_cloud(self, name, positions, colors=None, point_size=0.3):
        had_clouds = bool(self._scene.point_data)
        self._scene.add_cloud(name, positions, colors, point_size=point_size, reset_view=not had_clouds)
        if not had_clouds and not self._scene_view_initialized:
            self._initialize_scene_view()

    def update_cloud_color(self, name, colors):
        if name not in self._scene.point_data:
            return
        self._scene.update_cloud_color(name, colors)

    def present_scene(self):
        """Apply queued updates and present a frame immediately."""
        # TODO(present_scene): [视口渲染刷新] 当前接口强制立即渲染；应改为可合并的按需/局部刷新，并受焦点管控与帧率上限约束。
        try:
            pending = self._pending_point_updates
            self._pending_point_updates = {}
            for name, (sequence, positions, colors) in pending.items():
                if sequence == self._render_sequences.get(name):
                    self.update_cloud_points(name, positions, colors)
            pending_colors = self._pending_color_updates
            self._pending_color_updates = {}
            for name, (sequence, colors) in pending_colors.items():
                if sequence == self._render_sequences.get(name):
                    self.update_cloud_color(name, colors)
            adapter = self._adapter
            adapter._last_render_time = 0.0
            adapter.request_render('present')
            vis = adapter.vis
            if vis is not None and not adapter._destroyed:
                vis.poll_events()
                vis.update_renderer()
                adapter._render_pending = False
                adapter._scene_dirty = False
        except Exception as exc:
            print(f'[PCFD] viewport.present_failed error={exc!r}', flush=True)

    def update_cloud_points(self, name, positions, colors=None):
        if name not in self._scene.point_data:
            return
        self._scene.update_cloud_points(name, positions, colors)

    def replace_cloud_snapshot(self, name, positions, colors=None, metadata=None):
        """GUI-thread atomic point/colour/metadata replacement."""
        return self._scene.replace_cloud_snapshot(name, positions, colors, metadata)

    def commit_processing_snapshot(self, name, positions, colors=None,
                                   metadata=None, reset_view=False):
        """Commit the latest processing snapshot once, on the GUI thread."""
        sequence = self._render_sequences.get(name, 0) + 1
        self._render_sequences[name] = sequence
        self._pending_point_updates.pop(name, None)
        self._pending_color_updates.pop(name, None)
        count = self._scene.commit_cloud_snapshot(
            name, positions, colors, metadata, reset_view=reset_view)
        self._adapter.request_render('snapshot.committed')
        return count

    def remove_cloud(self, name):
        self._scene.remove_cloud(name)
        if not self._scene.point_data:
            # A station switch clears the old scene before adding the new
            # proxy. The next cloud must receive the canonical front view.
            self._scene_view_initialized = False
            self.clear_pick_markers()

    def clear(self):
        # TODO(clear): [项目生命周期] 清空场景时应与项目状态切换绑定，统一释放全部点云数组、Open3D 几何体及视口辅助对象，避免资源泄漏。
        if getattr(self._adapter, '_destroyed', False):
            self._scene.clear()
            return
        self.clear_pick_markers()
        self.clear_facade_boundary()
        self._scene.clear()
        self._scene_view_initialized = False

    def set_point_size(self, name, size):
        self._scene.set_point_size(name, size)

    def set_all_point_size(self, size):
        size = max(Open3DAdapter.MIN_POINT_SIZE,
                   min(float(size), Open3DAdapter.MAX_POINT_SIZE))
        for name in self._scene.get_cloud_names():
            self._scene.set_point_size(name, size)

    def toggle_bbox(self, name, min_bound, max_bound):
        bbox_name = f"{name}__bbox"
        visible = self._scene.bbox_visible.get(name, False)
        if visible:
            self._adapter.remove_geometry(bbox_name)
            self._scene.bbox_visible[name] = False
            return False
        self._adapter.add_geometry(bbox_name, make_bbox(min_bound, max_bound), reset_bounding_box=False)
        self._scene.bbox_visible[name] = True
        return True

    def toggle_normals(self, name, normals, length=0.5, max_lines=8000):
        normal_name = f"{name}__normals"
        if normal_name in self._scene.normal_names:
            self._adapter.remove_geometry(normal_name)
            self._scene.normal_names.remove(normal_name)
            return False
        data = self._scene.get_cloud_data(name)
        if data is None:
            return False
        lines = make_normals(data["pos"], normals, length=length, max_lines=max_lines)
        if lines is None:
            return False
        self._adapter.add_geometry(normal_name, lines, reset_bounding_box=False)
        self._scene.normal_names.add(normal_name)
        return True

    def get_cloud_names(self):
        return self._scene.get_cloud_names()

    def get_cloud_data(self, name):
        return self._scene.get_cloud_data(name)

    def get_active_cloud_name(self):
        return self._scene.active_name

    def set_selection_enabled(self, enabled, cloud_name=None):
        self._interactor.set_selection_enabled(enabled, cloud_name=cloud_name)

    def set_pick_enabled(self, enabled, radius=14, cloud_name=None):
        self._interactor.set_pick_enabled(enabled, radius=radius, cloud_name=cloud_name)

    def enter_pick_mode(self, cloud_name=None, pick_radius=8, callback=None):
        try:
            if self._roi_controller is not None:
                self._roi_controller.cancel()
        except Exception:
            pass
        if not self._camera.ensure_pinhole_projection():
            print(
                '[Pick] 无法切换到可拾取的透视相机模式',
                flush=True,
            )
            return
        self._interactor.invalidate_pick_projection_cache()
        self.set_selection_enabled(False)
        self.set_pick_enabled(True, radius=pick_radius, cloud_name=cloud_name)
        try:
            self.set_mode(self.InteractionMode.PICK)
        except Exception:
            pass
        if callback is not None:
            self.point_pick_callback = callback
        if self._pick_overlay is not None:
            self._pick_overlay.activate()

    def _handle_pick_screen(self, pos):
        """Handle a click from the top-level PICK overlay."""
        picked = self._interactor.pick_at_screen(pos, cloud_name=self._interactor.pick_cloud)
        if picked is None:
            return
        self._interactor.last_picked_point = np.asarray(picked['point'], dtype=float)
        callback = self._interactor.point_pick_callback
        if callable(callback):
            callback(picked)
        if self._pick_overlay is not None:
            self._pick_overlay.update()

    def pick_at_screen(self, pos, cloud_name=None):
        """公开的视图点选入口，算法层和业务层不接触 interactor。"""
        return self._interactor.pick_at_screen(pos, cloud_name=cloud_name)

    def enter_registration_pick_mode(self, source_cloud, target_cloud,
                                     callback, pick_radius=10):
        """
        进入注册配准选点模式。

        视图层职责：
        - 管理选点状态（源/目标点列表）
        - 渲染选点标记
        - 将选点事件通过回调通知业务层

        算法层职责（外部）：
        - 接收点对后计算变换矩阵
        """
        self.clear_pick_markers()

        # 选点状态机 —— 纯视图状态，无业务逻辑
        self._registration_pick_state = {
            'source_cloud': source_cloud,
                'target_cloud': target_cloud,
            'source_points': [],
            'target_points': [],
            'next_is_source': True,  # 下一次点击收集源点还是目标点
        }

        # 设置交互模式为 PICK
        self.set_mode(self.InteractionMode.PICK)
        self._interactor.pick_enabled = True
        self._interactor.pick_cloud = source_cloud  # 第一次先选源站点
        self._interactor.point_pick_callback = self._on_registration_pick

        # 存储外部回调（业务层传入）
        self._registration_callback = callback

        # 激活覆盖层
        if self._pick_overlay is not None:
            self._pick_overlay.activate()

        # 渲染初始状态（空标记）
        self.update_pick_markers()

    def _on_registration_pick(self, picked):
        """
        选点回调 —— 视图层内部处理。
        职责：记录点坐标、切换源/目标、更新视觉标记、通知业务层。
        """
        state = getattr(self, '_registration_pick_state', None)
        if state is None:
            return

        cloud_name = picked.get('cloud_name')
        point = np.asarray(picked['point'], dtype=np.float64)

        # 验证点属于正确的站点
        expected = state['source_cloud'] if state['next_is_source'] else state['target_cloud']
        if cloud_name != expected:
            return

        # 记录点坐标
        if state['next_is_source']:
            state['source_points'].append(point)
            state['next_is_source'] = False
            # 切换下一次点击的目标站点
            self._interactor.pick_cloud = state['target_cloud']
        else:
            state['target_points'].append(point)
            state['next_is_source'] = True
            # 切换回源站点
            self._interactor.pick_cloud = state['source_cloud']

        # 更新视觉标记
        self.update_pick_markers(
            state['source_points'],
            state['target_points']
        )

        # 通知业务层回调
        callback = getattr(self, '_registration_callback', None)
        if callable(callback):
            try:
                callback(picked, state['next_is_source'])
            except Exception:
                pass

    def registration_pick_points(self):
        """返回当前已选的源/目标点对 —— 供业务层读取。"""
        state = getattr(self, '_registration_pick_state', None) or {}
        return (list(state.get('source_points', [])),
                list(state.get('target_points', [])))

    def handle_pick_screen(self, pos):
        """处理来自覆盖层的屏幕点击，唯一的点选入口。"""
        picked = self._interactor.pick_at_screen(pos, cloud_name=self._interactor.pick_cloud)
        if picked is None:
            return

        # 更新最后选中的点（供外部查询）
        self._interactor.last_picked_point = np.asarray(picked['point'], dtype=float)

        # 触发选点回调
        callback = self._interactor.point_pick_callback
        if callable(callback):
            try:
                callback(picked)
            except Exception:
                pass

        # 刷新覆盖层显示
        if self._pick_overlay is not None:
            self._pick_overlay.update()

    def exit_pick_mode(self):
        self.set_pick_enabled(False)
        self.point_pick_callback = None
        if self._pick_overlay is not None:
            self._pick_overlay.deactivate()
        try:
            self.set_mode(self.InteractionMode.NAVIGATE)
        except Exception:
            pass

    def update_pick_markers(self, src_points=None, tgt_points=None):
        self.clear_pick_markers()
        src = np.asarray(src_points if src_points is not None else [], dtype=np.float64).reshape(-1, 3)
        tgt = np.asarray(tgt_points if tgt_points is not None else [], dtype=np.float64).reshape(-1, 3)
        self._pick_marker_xyz = src.copy() if len(src) else np.zeros((0, 3), dtype=np.float64)

        r_src = self._marker_radius_for_points(src) if len(src) else None
        r_tgt = self._marker_radius_for_points(tgt) if len(tgt) else None

        for i, point in enumerate(src):
            name = f"__pick_src_{i}"
            try:
                self._adapter.add_geometry(name, make_sphere(point, [1.0, 0.2, 0.2], r_src or 0.08), reset_bounding_box=False)
                self._pick_markers.append(name)
            except Exception:
                pass

        for i, point in enumerate(tgt):
            name = f"__pick_tgt_{i}"
            try:
                self._adapter.add_geometry(name, make_sphere(point, [0.2, 1.0, 0.2], r_tgt or 0.08), reset_bounding_box=False)
                self._pick_markers.append(name)
            except Exception:
                pass

        lines = make_pair_lines(src, tgt)
        if lines is not None:
            name = "__pick_pair_lines"
            try:
                self._adapter.add_geometry(name, lines, reset_bounding_box=False)
                self._pick_lines.append(name)
            except Exception:
                pass
        try:
            self._adapter.poll()
        except Exception:
            pass
    def _refresh_pick_overlay_labels(self):
        if self._pick_overlay is not None:
            self._pick_overlay.update()

    def clear_pick_markers(self):
        for name in self._pick_markers + self._pick_lines:
            self._adapter.remove_geometry(name)
        self._pick_markers = []
        self._pick_lines = []
        self._pick_marker_xyz = np.zeros((0, 3), dtype=np.float64)
        if self._pick_overlay is not None:
            self._pick_overlay.update()

    def save_screenshot(self, path):
        image = self._adapter.capture_screen()
        if image is None:
            return
        arr = (np.asarray(image) * 255).clip(0, 255).astype(np.uint8)
        height, width, _ = arr.shape
        qimg = QImage(arr.data, width, height, arr.strides[0], QImage.Format_RGB888).copy()
        qimg.save(path)

    def capture_match_view(self, cloud_name=None):
        """捕获自动匹配所需的当前点云彩色图、深度图和相机参数。"""
        if not self._camera.ensure_pinhole_projection():
            raise RuntimeError('当前正交视图无法转换为匹配所需的透视相机')
        self._interactor.invalidate_pick_projection_cache()
        captured = self._adapter.capture_color_depth_camera(point_size=5.0)
        if captured is None:
            raise RuntimeError('无法捕获当前点云视图')
        color, depth, camera = captured
        rgb = (np.asarray(color) * 255.0).clip(0, 255).astype(np.uint8)
        depth_array = np.asarray(depth, dtype=np.float64).copy()
        if rgb.ndim != 3 or rgb.shape[2] < 3 or depth_array.ndim != 2:
            raise RuntimeError('Open3D 返回的视图图像格式无效')
        camera_matrix = np.asarray(
            camera.intrinsic.intrinsic_matrix,
            dtype=np.float64,
        ).copy()

        # 原生 Open3D 帧缓冲可能仍保持创建窗口时的尺寸。按嵌入式三维显示框
        # 的宽高比居中裁剪，确保保存文件不包含显示框之外的缓冲区域。
        view_width = max(1, int(self._container.width()))
        view_height = max(1, int(self._container.height()))
        target_aspect = view_width / view_height
        source_height, source_width = rgb.shape[:2]
        source_aspect = source_width / max(source_height, 1)
        if abs(source_aspect - target_aspect) > 1e-3:
            if source_aspect > target_aspect:
                crop_width = max(1, int(round(source_height * target_aspect)))
                x0 = max(0, (source_width - crop_width) // 2)
                rgb = rgb[:, x0:x0 + crop_width]
                depth_array = depth_array[:, x0:x0 + crop_width]
                camera_matrix[0, 2] -= x0
            else:
                crop_height = max(1, int(round(source_width / target_aspect)))
                y0 = max(0, (source_height - crop_height) // 2)
                rgb = rgb[y0:y0 + crop_height]
                depth_array = depth_array[y0:y0 + crop_height]
                camera_matrix[1, 2] -= y0
        result = {
            'view_bgr': np.ascontiguousarray(rgb[:, :, :3][:, :, ::-1]),
            'depth_image': depth_array,
            'camera_matrix': camera_matrix,
            'extrinsic': np.asarray(camera.extrinsic, dtype=np.float64).copy(),
        }
        data = self.get_cloud_data(cloud_name) if cloud_name else None
        if data is not None and data.get('pos') is not None:
            # 保留对当前显示点云的只读引用，算法据此建立像素→点索引图。
            result['cloud_points'] = np.asarray(data['pos'])
        return result

    def _marker_radius_for_points(self, pts: np.ndarray | None, pixel_radius: float = 6.0):
        try:
            if pts is None or len(pts) == 0:
                return max(self._fallback_scene_scale() * 0.002, 0.05)
            ctr = self._adapter.get_view_control()
            if ctr is None:
                return 0.08
            if not self._camera.is_orthographic():
                try:
                    params = ctr.convert_to_pinhole_camera_parameters()
                    fx = float(params.intrinsic.intrinsic_matrix[0, 0])
                    screen, valid = self._camera.project_points(pts)
                    if screen is None:
                        return 0.08
                    z = screen[:, 2][valid]
                    if len(z) == 0:
                        return 0.08
                    z_avg = float(np.median(z))
                    world_per_pixel = z_avg / max(fx, 1e-6)
                    r = world_per_pixel * float(pixel_radius)
                    return float(max(0.02, min(r, 0.5)))
                except Exception:
                    pass

            try:
                if hasattr(self._container, "devicePixelRatioF"):
                    dpr = float(self._container.devicePixelRatioF())
                elif hasattr(self._container, "devicePixelRatio"):
                    dpr = float(self._container.devicePixelRatio())
                else:
                    dpr = 1.0
            except Exception:
                dpr = 1.0
            try:
                zoom = float(ctr.get_zoom())
            except Exception:
                zoom = 0.6
            scene_scale = self._fallback_scene_scale()
            scene_factor = max(0.2, min(scene_scale / 10.0, 3.0))
            from config.settings import Config
            pan_base = float(getattr(Config, 'PAN_BASE_SPEED', 0.06))
            zoom_factor = 0.6 / max(zoom, 0.02)
            world_per_pixel = pan_base * zoom_factor * scene_factor / max(1.0, dpr)
            world_per_pixel = max(world_per_pixel, 1e-6)
            r = world_per_pixel * float(pixel_radius)
            return float(max(0.02, min(r, 0.5)))
        except Exception:
            return 0.08

    def _fallback_scene_scale(self):
        extents = []
        for data in self._scene.point_data.values():
            pos = data["pos"]
            if len(pos):
                extents.append(np.max(pos, axis=0) - np.min(pos, axis=0))
        if not extents:
            return 1.0
        max_dim = float(np.max(np.vstack(extents)))
        return max_dim if max_dim > 0 else 1.0
