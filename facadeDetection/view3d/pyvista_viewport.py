"""基于 PyVista/VTK 的三维视口（BaseViewport 对齐实现）。

与 Open3DViewport 同一接口：外部代码经 BaseViewport 与既有扩展方法访问，
不感知渲染后端差异。相对 legacy Open3D 的收益：
- Qt 原生 widget，无 GLFW 原生窗口嵌入 hack（无白窗、无线程归属约束）
- 线宽/透明度/材质生效（legacy 管线被无视的 RenderOption 问题不存在）
- 按需渲染，无需 33ms 轮询节拍

注意：open3d 库本身仍用于 PLY IO 与几何算法（与视口无关），本文件只替换
渲染窗口部分。
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QWidget

from .base_viewport import BaseViewport
from .lod import normalize_colors

try:
    from config.settings import Config
    BACKGROUND = str(getattr(Config, 'VIEWPORT_BACKGROUND', '#FFFFFF'))
except Exception:
    BACKGROUND = '#FFFFFF'

# 地面网格 fragment shader（移植自 StudioViewport.tsx 的 GLSL，Z-up 适配）。
# 拆两段：gridLine 辅助函数放声明区（GLSL 不允许函数内定义函数），
# 网格计算语句放 //VTK::Light::Impl（该挂钩位于 fragment main 函数体内）。
# 与原版差异：固定网格（不跟随相机）、alpha 混向白底输出不透明色规避透明深度问题。
_GRID_FRAGMENT_DECL = """
in vec3 gridWorldPos;

float gridLine(vec2 p, float scale, float linePx) {
  vec2 c = p / scale;
  vec2 fw = fwidth(c);
  vec2 d = abs(fract(c - 0.5) - 0.5);
  vec2 halfW = linePx * 0.5 * fw;
  vec2 line = 1.0 - smoothstep(halfW, halfW + 0.75 * fw, d);
  return max(line.x, line.y);
}
"""

_GRID_FRAGMENT_IMPL = """
  vec2 p = gridWorldPos.xy;
  float dist = distance(p, vec2(GRID_CX, GRID_CY));

  float minor = gridLine(p, 0.1, 0.05)
      * (1.0 - smoothstep(GRID_FADE_NEAR * 2.0, GRID_FADE_FAR * 2.0, dist));
  float major = gridLine(p, 1.0, 0.05)
      * (1.0 - smoothstep(GRID_FADE_NEAR * 10.0, GRID_FADE_FAR * 10.0, dist));

  vec3 gridColor = mix(vec3(0.45, 0.45, 0.45), vec3(0.0, 0.0, 0.0), major);
  float alpha = max(minor, major);
  if (alpha <= 0.001) discard;
  gl_FragData[0] = vec4(gridColor, alpha);
"""


def _grid_fragment_for(center):
    return (_GRID_FRAGMENT_IMPL
            .replace('GRID_CX', f'{float(center[0]):.6f}')
            .replace('GRID_CY', f'{float(center[1]):.6f}')
            .replace('GRID_FADE_NEAR', '8.0')
            .replace('GRID_FADE_FAR', '15.0'))


class _RenderQueue(QObject):
    """跨线程渲染请求编组：worker 只发信号，VTK 对象只在 GUI 线程触碰。"""

    color = Signal(str, object)
    points = Signal(str, object, object)


class _RenderCounterShim:
    """main_window 首帧等待读取 viewport._adapter._frames_rendered 的兼容垫片。"""

    def __init__(self):
        self._frames_rendered = 0


class _GestureFilter(QObject):
    """把 Qt 捏合手势转发给视口的缩放处理。"""

    def __init__(self, viewport, parent=None):
        super().__init__(parent)
        self._viewport = viewport

    def eventFilter(self, watched, event):
        try:
            from PySide6.QtCore import QEvent, Qt
            if event.type() == QEvent.Gesture:
                pinch = event.gesture(Qt.PinchGesture)
                if pinch is not None:
                    factor = float(pinch.scaleFactor())
                    if factor > 0:
                        self._viewport._zoom_by_factor(factor)
                    return True
        except Exception:
            pass
        return False


class PyVistaViewport(BaseViewport):
    """Qt 原生 PyVista 视口，接口与 Open3DViewport 对齐。"""

    def __init__(self, parent=None):
        from pyvistaqt import QtInteractor

        self._widget = QtInteractor(parent)
        self._widget.setObjectName('pyvistaViewport')
        self._plotter = self._widget  # QtInteractor 本身即 Plotter API
        self._plotter.set_background(BACKGROUND)
        # 坐标轴指示（X红/Y绿/Z蓝），替代 Web 版 AxesHelper
        try:
            self._plotter.add_axes(line_width=2)
        except Exception:
            pass

        # 云数据模型与 view3d.scene.PointCloudScene 对齐
        self.point_data: dict[str, dict] = {}
        self._polys: dict[str, pv.PolyData] = {}
        self._actors: dict[str, object] = {}
        self._bbox_actors: dict[str, object] = {}
        self._hidden_rows: dict[str, np.ndarray] = {}
        self._active_name = None

        self._render_queue = _RenderQueue(self._widget)
        self._render_queue.color.connect(
            self.update_cloud_color, Qt.QueuedConnection)
        self._render_queue.points.connect(
            self.update_cloud_points, Qt.QueuedConnection)
        self._adapter = _RenderCounterShim()  # 首帧等待兼容垫片
        self._render_enabled = True
        self._scene_view_initialized = False
        self._grid_actor = None
        self._grid_border_actor = None
        self._pick_callback = None
        self._last_picked_point = None

        # Z-up 转台交互（Tekla 式）：左键旋转、右键平移、滚轮缩放
        self._drag_left = None
        self._drag_right = None
        self._install_zup_interaction()

        # 交互模式与拾取状态
        self._mode = self.InteractionMode.NAVIGATE
        self._pick_cloud = None
        self._pick_radius = 8.0
        self._pick_press = None
        self._registration_pick_state = None
        self._registration_callback = None
        self._pick_marker_actors = []
        self._measure_actors = []
        self._ruler_actors = []
        self._heatmap_quad_actors = []
        self._overlay_renderer = None
        self._plane_pick_model = None
        self._plane_pick_callback = None

        # ROI 框选：复用 roi_selection.py 的覆盖层控制器（_interactor 指向自身）
        self._roi_on_complete = None
        self._roi_bbox_actor = None
        self._interactor = self
        try:
            from .roi_selection import ROISelectionController
            self._roi_controller = ROISelectionController(
                self, container_widget=self._widget)
        except Exception:
            self._roi_controller = None

        # 渲染服务算建筑包围盒需要 viewport._camera.project_points()，
        # 签名与本类同名方法一致，直接以自身充当相机垫片。
        self._camera = self

        # 触控手势：捏合缩放（Surface 触屏）
        try:
            from PySide6.QtCore import Qt as _Qt
            self._gesture_filter = _GestureFilter(self)
            self._widget.grabGesture(_Qt.PinchGesture)
            self._widget.installEventFilter(self._gesture_filter)
        except Exception:
            pass

        # VTK 按需渲染：渲染成功后递增帧计数
        try:
            self._widget.render_window.AddObserver(
                'EndEvent', self._on_render_end)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Z-up 转台交互（左键旋转 / 右键平移 / 滚轮缩放）
    # ------------------------------------------------------------------
    _ROTATE_SPEED_DEG = 0.3   # 每像素旋转角速度（方位/俯仰）
    _ELEV_LIMIT_DEG = 85.0    # 俯仰钳制：留出余量，避免视线与全局Z平行
                               # （平行时VTK每帧刷view-up重置告警）
    _WHEEL_FACTOR = 1.15      # 滚轮缩放倍率

    def _install_zup_interaction(self):
        """以 vtkInteractorStyleUser 接管全部鼠标事件，实现 Z 轴锁定转台。

        旋转模型与 Tekla/BIM 一致：水平拖绕注视点的全局 Z 轴转方位，
        垂直拖改变俯仰并钳制在 ±89°，视线 up 永远锁定全局 Z，绝不滚转。
        """
        try:
            import vtk
            style = vtk.vtkInteractorStyleUser()
            style.AddObserver('LeftButtonPressEvent',
                              lambda o, e: self._on_press('left', o))
            style.AddObserver('RightButtonPressEvent',
                              lambda o, e: self._on_press('right', o))
            style.AddObserver('LeftButtonReleaseEvent',
                              lambda o, e: self._on_release('left', o))
            style.AddObserver('RightButtonReleaseEvent',
                              lambda o, e: self._on_release('right'))
            style.AddObserver('MouseMoveEvent',
                              lambda o, e: self._on_mouse_move(o))
            style.AddObserver('MouseWheelForwardEvent',
                              lambda o, e: self._on_wheel(1))
            style.AddObserver('MouseWheelBackwardEvent',
                              lambda o, e: self._on_wheel(-1))
            # pyvista 的 RenderWindowInteractor 是包装器，底层 vtk 对象在 .interactor
            vtk_iren = getattr(self._plotter.iren, 'interactor',
                               self._plotter.iren)
            vtk_iren.SetInteractorStyle(style)
            self._zup_style = style  # 防 GC
        except Exception as exc:
            print(f'[PCFD] viewport.zup_interaction_failed: {exc!r}',
                  flush=True)

    @staticmethod
    def _pointer_xy(style_obj):
        """从样式对象拿底层 interactor 的事件坐标（包装器不保证有此方法）。"""
        try:
            iren = style_obj.GetInteractor()
            return tuple(iren.GetEventPosition())
        except Exception:
            return None

    def _on_press(self, button, style_obj):
        xy = self._pointer_xy(style_obj)
        if xy is None:
            return
        if button == 'left':
            if self._mode == self.InteractionMode.PICK:
                # 拾取模式：左键只做点选，不进入旋转
                self._pick_press = xy
                print(f'[PCFD] viewport.pick_press {xy}', flush=True)
                return
            self._drag_left = xy
        else:
            self._drag_right = xy

    def _on_release(self, button, style_obj=None):
        if button == 'left':
            if self._mode == self.InteractionMode.PICK:
                press, self._pick_press = self._pick_press, None
                if press is not None and style_obj is not None:
                    xy = self._pointer_xy(style_obj)
                    if (xy is not None
                            and abs(xy[0] - press[0]) <= 6
                            and abs(xy[1] - press[1]) <= 6):
                        from PySide6.QtCore import QPoint
                        print(f'[PCFD] viewport.pick_release {xy}',
                              flush=True)
                        self.handle_pick_screen(QPoint(*xy))
                return
            self._drag_left = None
        else:
            self._drag_right = None

    def _on_mouse_move(self, style_obj):
        xy = self._pointer_xy(style_obj)
        if xy is None:
            return
        if self._drag_left is not None:
            dx, dy = xy[0] - self._drag_left[0], xy[1] - self._drag_left[1]
            self._drag_left = xy
            if dx or dy:
                self._turntable_rotate(dx, dy)
        elif self._drag_right is not None:
            dx, dy = xy[0] - self._drag_right[0], xy[1] - self._drag_right[1]
            self._drag_right = xy
            if dx or dy:
                self._camera_pan(dx, dy)

    def _turntable_rotate(self, dx, dy):
        """Z 轴锁定转台：dx→绕全局Z的方位角，dy→俯仰角（钳制）。"""
        try:
            camera = self._plotter.camera
            pos = np.asarray(camera.position, dtype=np.float64)
            fp = np.asarray(camera.focal_point, dtype=np.float64)
            off = pos - fp
            radius = float(np.linalg.norm(off))
            if radius < 1e-9:
                return
            az = np.arctan2(off[1], off[0])
            el = np.arcsin(np.clip(off[2] / radius, -1.0, 1.0))
            az -= np.radians(dx * self._ROTATE_SPEED_DEG)
            el = np.clip(el + np.radians(dy * self._ROTATE_SPEED_DEG),
                         -np.radians(self._ELEV_LIMIT_DEG),
                         np.radians(self._ELEV_LIMIT_DEG))
            new_off = radius * np.array(
                [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                 np.sin(el)])
            camera.position = tuple(fp + new_off)
            camera.focal_point = tuple(fp)
            # up 锁回全局 Z：永远不出现滚转
            camera.up = (0.0, 0.0, 1.0)
            self._render()
        except Exception:
            pass

    def _camera_pan(self, dx, dy):
        """按注视点深度的世界/像素比平移（VTK 经典配方）。"""
        try:
            renderer = self._plotter.renderer
            camera = self._plotter.camera
            fp = np.asarray(camera.focal_point, dtype=np.float64)
            pos = np.asarray(camera.position, dtype=np.float64)
            renderer.SetWorldPoint(fp[0], fp[1], fp[2], 1.0)
            renderer.WorldToDisplay()
            d = renderer.GetDisplayPoint()
            renderer.SetDisplayPoint(d[0] - dx, d[1] - dy, d[2])
            renderer.DisplayToWorld()
            w = renderer.GetWorldPoint()
            delta = np.array([w[0] / w[3], w[1] / w[3], w[2] / w[3]]) - fp
            camera.focal_point = tuple(fp + delta)
            camera.position = tuple(pos + delta)
            self._render()
        except Exception:
            pass

    def _on_wheel(self, direction):
        """滚轮缩放：正交改 parallel scale，透视走 Dolly。"""
        factor = (self._WHEEL_FACTOR if direction > 0
                  else 1.0 / self._WHEEL_FACTOR)
        self._zoom_by_factor(factor)

    def _zoom_by_factor(self, factor):
        """按倍率缩放（滚轮/捏合手势共用）：正交改 parallel scale，透视 Dolly。"""
        try:
            camera = self._plotter.camera
            try:
                parallel = bool(camera.parallel_projection)
            except Exception:
                parallel = False
            if parallel:
                scale = float(camera.parallel_scale) / factor
                camera.parallel_scale = max(1e-6, scale)
            else:
                camera.Dolly(factor)
            self._render()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 基础装配
    # ------------------------------------------------------------------
    def _on_render_end(self, *_args):
        self._adapter._frames_rendered += 1

    @property
    def frames_rendered(self):
        return self._adapter._frames_rendered

    def _render(self):
        if self._render_enabled:
            try:
                self._plotter.update()
            except Exception:
                pass

    def get_widget(self):
        return self._widget

    def widget(self):
        return self._widget

    def set_render_enabled(self, enabled):
        """页面切换门控：VTK 按需渲染，这里只记录标志并立即补一帧。"""
        self._render_enabled = bool(enabled)
        if enabled:
            self._render()

    def has_clouds(self) -> bool:
        return bool(self.point_data)

    def get_active_cloud_name(self):
        return self._active_name

    # ------------------------------------------------------------------
    # 云管理（语义对齐 PointCloudScene）
    # ------------------------------------------------------------------
    def add_cloud(self, name, positions, colors=None, point_size=0.75):
        positions = np.ascontiguousarray(
            np.asarray(positions, dtype=np.float32).reshape(-1, 3))
        has_source_colors = (
            colors is not None
            and np.asarray(colors).size == len(positions) * 3
            and np.all(np.isfinite(np.asarray(colors))))
        colors = np.ascontiguousarray(
            normalize_colors(colors, len(positions)).astype(np.float32))
        first_cloud = not self.point_data
        self.point_data[name] = {
            'pos': positions,
            'color': colors,
            'size': max(0.01, min(float(point_size), 5.0)),
            'has_source_colors': has_source_colors,
            'color_source': 'ply_rgb' if has_source_colors else 'fallback_gray',
        }
        self._hidden_rows.pop(name, None)
        self._active_name = name
        self._upload_cloud(name, reset_view=first_cloud)

    def _visible_view(self, name):
        """(pos, color) 的可见视图：set_hidden_rows 摘除的行不进入渲染。

        point_data 始终保存全量数组（拾取/配色口径不变），掩码只在上传
        PolyData 时应用。
        """
        data = self.point_data[name]
        rows = self._hidden_rows.get(name)
        if rows is None:
            return data['pos'], data['color']
        keep = np.ones(len(data['pos']), dtype=bool)
        valid = rows[(rows >= 0) & (rows < len(data['pos']))]
        keep[valid] = False
        return data['pos'][keep], data['color'][keep]

    def set_hidden_rows(self, name, rows=None):
        """把指定行从显示中摘除（复核页选中立面：点云让位给热力贴图）。

        rows=None 或空数组恢复完整显示。
        """
        if name not in self.point_data:
            return
        if rows is None or len(rows) == 0:
            self._hidden_rows.pop(name, None)
        else:
            self._hidden_rows[name] = np.asarray(rows, dtype=np.int64)
        self._upload_cloud(name)

    def _upload_cloud(self, name, reset_view=False):
        data = self.point_data[name]
        pos_vis, col_vis = self._visible_view(name)
        poly = self._polys.get(name)
        if poly is None or poly.n_points != len(pos_vis):
            # 点数变化（含隐藏行切换）：PolyData 不能就地改尺寸，重建并换掉 actor
            poly = pv.PolyData(pos_vis)
            self._polys[name] = poly
            old = self._actors.pop(name, None)
            if old is not None:
                self._plotter.remove_actor(old)
            self._actors[name] = None
        else:
            poly.points[:] = pos_vis
        poly.point_data['colors'] = (col_vis * 255).astype(np.uint8)
        actor = self._actors.get(name)
        if actor is None:
            actor = self._plotter.add_points(
                poly, scalars='colors', rgb=True,
                point_size=max(1, int(round(data['size'] * 4))),
                render_points_as_spheres=False, name=f'cloud::{name}')
            self._actors[name] = actor
        else:
            poly.Modified()
        if reset_view:
            self._initialize_scene_view()
        self._render()

    def _initialize_scene_view(self):
        """初始相机与 Open3D 视口对齐：正视立面、Z 轴向上、正交投影。"""
        try:
            # 与 Open3DViewport._initialize_scene_view 同向：
            # front=[0,-1,0]（面向建筑正面），up=[0,0,1]（Z 轴屏幕向上）
            self._plotter.view_vector((0.0, -1.0, 0.0),
                                      viewup=(0.0, 0.0, 1.0))
            self._plotter.reset_camera()
            # 与 Config.ORTHO_FOV_DEG≈5° 的 Open3D 正交模式等效
            try:
                self._plotter.enable_parallel_projection()
            except Exception:
                pass
            self._plotter.reset_camera()  # 正交下重取构图
            try:
                # reset 后再强制一次，防止被构图重置回透视
                self._plotter.enable_parallel_projection()
            except Exception:
                pass
            # 地面网格（StudioViewport.tsx shader 移植）：
            # 取消下一行注释即可恢复显示
            self._add_ground_grid()
            self._scene_view_initialized = True
        except Exception:
            pass

    def _add_ground_grid(self):
        """场景下方铺地面网格（StudioViewport.tsx 的 shader 移植）。"""
        try:
            import vtk
            if self._grid_actor is not None:
                return
            pts = [d['pos'] for d in self.point_data.values()
                   if len(d.get('pos', []))]
            if not pts:
                return
            allp = np.vstack(pts)
            center = allp.mean(axis=0)
            span = float(np.ptp(allp, axis=0).max())
            grid_z = float(allp[:, 2].min()) - max(0.05, span * 0.01)
            size = max(120.0, span * 1.5)  # 覆盖点云地面足迹
            plane = pv.Plane(center=(float(center[0]), float(center[1]),
                                     grid_z),
                             direction=(0, 0, 1),
                             i_size=size, j_size=size,
                             i_resolution=1, j_resolution=1)
            actor = self._plotter.add_mesh(plane, name='ground_grid')
            # 半透明：alpha 低处透出白底，而非混色
            try:
                actor.prop.opacity = 0.99
            except Exception:
                pass
            # 外围边框：与网格同尺寸的外圈线框
            try:
                edges = plane.extract_feature_edges()
                self._grid_border_actor = self._plotter.add_mesh(
                    edges, color=(0.25, 0.25, 0.25), line_width=2,
                    name='ground_grid_border')
            except Exception:
                self._grid_border_actor = None
            # VTK 9.7：自定义 varying 走 ValuePass 挂钩传递 vertex→fragment
            prop = actor.GetShaderProperty()
            prop.AddVertexShaderReplacement(
                '//VTK::ValuePass::Dec', True,
                'out vec3 gridWorldPos;', False)
            prop.AddVertexShaderReplacement(
                '//VTK::ValuePass::Impl', True,
                'gridWorldPos = vertexMC.xyz;', False)
            prop.AddFragmentShaderReplacement(
                '//VTK::ValuePass::Dec', True,
                _GRID_FRAGMENT_DECL, False)
            prop.AddFragmentShaderReplacement(
                '//VTK::Light::Impl', True,
                _grid_fragment_for(center), False)
            self._grid_actor = actor
        except Exception as exc:
            print(f'[PCFD] viewport.ground_grid_failed: {exc!r}', flush=True)

    def reset_view(self):
        """恢复建筑立面默认正视图（与 Open3D 视口一致）。"""
        self._scene_view_initialized = False
        if self.point_data:
            self._initialize_scene_view()

    def update_cloud_color(self, name, colors):
        data = self.point_data.get(name)
        if data is None:
            return
        data['color'] = np.ascontiguousarray(
            normalize_colors(colors, len(data['pos'])).astype(np.float32))
        data.pop('display_proxy_lookup', None)
        _, col_vis = self._visible_view(name)
        poly = self._polys.get(name)
        if poly is None or poly.n_points != len(col_vis):
            # 点数已变化（换数据集/去噪/隐藏行切换后）：整朵重传，避免数组长度不匹配
            self._upload_cloud(name)
            return
        poly.point_data['colors'][:] = (col_vis * 255).astype(np.uint8)
        poly.Modified()
        self._render()

    def update_cloud_points(self, name, positions, colors=None):
        data = self.point_data.get(name)
        if data is None:
            return
        new_pos = np.ascontiguousarray(
            np.asarray(positions, dtype=np.float32).reshape(-1, 3))
        if len(new_pos) != len(data['pos']):
            # 点集重排后旧隐藏行号失效，恢复完整显示
            self._hidden_rows.pop(name, None)
        data['pos'] = new_pos
        if colors is None:
            colors = data.get('color')
        data['color'] = np.ascontiguousarray(
            normalize_colors(colors, len(data['pos'])).astype(np.float32))
        data.pop('display_proxy_lookup', None)
        self._active_name = name
        self._upload_cloud(name)

    def replace_cloud_snapshot(self, name, positions, colors=None, metadata=None):
        if name not in self.point_data:
            return False
        self.update_cloud_points(name, positions, colors)
        if metadata:
            self.point_data[name].update(metadata)
            self.point_data[name].pop('display_proxy_lookup', None)
        return True

    def commit_cloud_snapshot(self, name, positions, colors=None, metadata=None,
                              point_size=0.3, reset_view=False):
        """Create or atomically replace a cloud owned by the GUI thread."""
        if name not in self.point_data:
            self.add_cloud(name, positions, colors, point_size=point_size)
            data = self.point_data.get(name)
            if data is not None and metadata:
                data.update(metadata)
            return len(data.get('pos', [])) if data is not None else 0
        self.replace_cloud_snapshot(name, positions, colors, metadata)
        return len(self.point_data[name]['pos'])

    def commit_processing_snapshot(self, name, positions, colors=None,
                                   metadata=None, reset_view=False):
        return self.commit_cloud_snapshot(
            name, positions, colors, metadata, reset_view=reset_view)

    def remove_cloud(self, name):
        actor = self._actors.pop(name, None)
        if actor is not None:
            self._plotter.remove_actor(actor)
        bbox = self._bbox_actors.pop(name, None)
        if bbox is not None:
            self._plotter.remove_actor(bbox)
        self._polys.pop(name, None)
        self.point_data.pop(name, None)
        self._hidden_rows.pop(name, None)
        if self._active_name == name:
            self._active_name = next(iter(self.point_data), None)
        self._render()

    def clear(self):
        for name in list(self._actors):
            self._plotter.remove_actor(self._actors[name])
        for name in list(self._bbox_actors):
            self._plotter.remove_actor(self._bbox_actors[name])
        for grid in (self._grid_actor, self._grid_border_actor):
            if grid is not None:
                self._plotter.remove_actor(grid)
        self._grid_actor = None
        self._grid_border_actor = None
        self._actors.clear()
        self._bbox_actors.clear()
        self._polys.clear()
        self.point_data.clear()
        self._hidden_rows.clear()
        self._active_name = None
        self._render()

    def get_cloud_names(self) -> list[str]:
        return list(self.point_data.keys())

    def get_cloud_data(self, name):
        data = self.point_data.get(name)
        if data is not None and 'display_proxy_lookup' not in data:
            displayed = np.asarray(data.get('proxy_ids', []), dtype=np.int64)
            if len(displayed) == len(data.get('pos', [])):
                data['display_proxy_lookup'] = {
                    int(v): i for i, v in enumerate(displayed)}
        return data

    def set_point_size(self, name, size):
        data = self.point_data.get(name)
        actor = self._actors.get(name)
        if data is not None and actor is not None:
            value = max(1, int(round(float(size) * 4)))
            data['size'] = float(size)
            actor.prop.point_size = value
            self._render()

    def set_all_point_size(self, size):
        for name in self.get_cloud_names():
            self.set_point_size(name, size)

    # ------------------------------------------------------------------
    # 包围盒 / 法线（VTK 线宽生效，不再受 legacy 1px 限制）
    # ------------------------------------------------------------------
    def toggle_bbox(self, name, min_bound, max_bound):
        if name in self._bbox_actors:
            self._plotter.remove_actor(self._bbox_actors.pop(name))
            self._render()
            return False
        bounds = [min_bound[0], max_bound[0],
                  min_bound[1], max_bound[1],
                  min_bound[2], max_bound[2]]
        actor = self._plotter.add_mesh(
            pv.Box(bounds=bounds), style='wireframe',
            color=(0.3, 0.8, 1.0), line_width=3,
            name=f'bbox::{name}')
        self._bbox_actors[name] = actor
        self._render()
        return True

    def toggle_normals(self, name, normals, length=0.5, max_lines=8000):
        key = f'{name}__normals'
        if key in self._bbox_actors:
            self._plotter.remove_actor(self._bbox_actors.pop(key))
            self._render()
            return False
        data = self.point_data.get(name)
        if data is None:
            return False
        pos = data['pos']
        normals = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
        step = max(1, len(pos) // max_lines)
        sel = np.arange(0, len(pos), step)
        cloud = pv.PolyData(pos[sel])
        cloud.point_data['normals'] = normals[sel]
        glyph = cloud.glyph(orient='normals', scale=False, factor=float(length))
        actor = self._plotter.add_mesh(glyph, color=(0.9, 0.9, 0.4), name=key)
        self._bbox_actors[key] = actor
        self._render()
        return True

    # ------------------------------------------------------------------
    # 跨线程入口（worker 完成回调走 QueuedConnection 回 GUI）
    # ------------------------------------------------------------------
    def queue_update_cloud_color(self, name, colors):
        self._render_queue.color.emit(name, colors)

    def queue_update_cloud_points(self, name, positions, colors=None):
        self._render_queue.points.emit(name, positions, colors)

    def invalidate_render_queue(self):
        pass  # VTK 无延迟队列，语义空实现

    # ------------------------------------------------------------------
    # 投影 / 拾取 / ROI
    # ------------------------------------------------------------------
    class InteractionMode:
        NAVIGATE = 'navigate'
        PICK = 'pick'
        ROI = 'roi'

    def set_mode(self, mode: str):
        """交互模式切换：ROI/PICK 由覆盖层或拾取态接管输入。"""
        self._mode = str(mode).lower()

    def project_points(self, points):
        """3D 点投影为屏幕坐标（逻辑像素，与 Qt 事件坐标一致）。

        向量化实现：相机合成投影矩阵一次乘完（不逐点调 vtk，160 万点毫秒级）。
        VTK display 坐标原点在左下、且为物理像素：换算 y 翻转并除以 dpr。
        """
        try:
            renderer = self._plotter.renderer
            camera = renderer.GetActiveCamera()
            # VTK display coordinates are framebuffer (physical) pixels while
            # Qt mouse events are logical pixels.  Use the renderer size for
            # the matrix and convert exactly once at the API boundary.
            fb_w, fb_h = (int(v) for v in renderer.GetSize())
            fb_w = max(1, fb_w); fb_h = max(1, fb_h)
            dpr = max(1.0, float(self._widget.devicePixelRatioF()))
            width = fb_w / dpr
            height = fb_h / dpr
            aspect = fb_w / fb_h
            vtk_m = camera.GetCompositeProjectionTransformMatrix(
                aspect, -1.0, 1.0)
            m = np.array([[vtk_m.GetElement(i, j) for j in range(4)]
                          for i in range(4)], dtype=np.float64)
            pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
            hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
            clip = hom @ m.T
            w = clip[:, 3]
            with np.errstate(divide='ignore', invalid='ignore'):
                ndc = clip[:, :3] / np.where(np.abs(w) < 1e-12, np.nan, w)[:, None]
            # VTK display：左下原点、物理像素 → 转 Qt 左上原点、逻辑像素
            sx = (ndc[:, 0] + 1.0) * 0.5 * fb_w / dpr
            sy = (ndc[:, 1] + 1.0) * 0.5 * fb_h / dpr
            screen = np.stack([sx, height - sy], axis=1)
            valid = np.isfinite(ndc).all(axis=1) & (np.abs(ndc[:, 2]) <= 1.0)
            return screen, valid
        except Exception as exc:
            print(f'[PCFD] viewport.project_points_failed: {exc!r}',
                  flush=True)
            return None

    # ---- 屏幕拾取（投影最近点，全云通用，无 VTK picker 依赖）----
    def pick_at_screen(self, pos, cloud_name=None):
        """屏幕坐标 → 最近云点。返回 {'cloud_name','point','index'} 或 None。"""
        name = cloud_name or self._active_name
        data = self.point_data.get(name)
        if data is None:
            return None
        projected = self.project_points(data['pos'])
        if projected is None:
            return None
        screen, valid = projected
        if not np.any(valid):
            return None
        xy = np.array([pos.x(), pos.y()], dtype=np.float64)
        dist = np.where(valid,
                        np.hypot(screen[:, 0] - xy[0], screen[:, 1] - xy[1]),
                        np.inf)
        row = int(np.argmin(dist))
        if not np.isfinite(dist[row]) or dist[row] > self._pick_radius:
            return None
        point = np.asarray(data['pos'][row], dtype=np.float64)
        self._last_picked_point = point
        return {'cloud_name': name, 'point': point, 'index': row}

    def handle_pick_screen(self, pos):
        """屏幕点击的唯一拾取入口（与 Open3D 版同语义）。"""
        if self._plane_pick_model is not None:
            point = self.ray_plane_pick(pos, self._plane_pick_model)
            print(f'[PCFD] viewport.plane_pick -> {point}', flush=True)
            if point is None:
                return
            callback = self._plane_pick_callback
            if callable(callback):
                try:
                    callback({'cloud_name': self._active_name,
                              'point': point, 'index': -1})
                except Exception:
                    pass
            return
        picked = self.pick_at_screen(pos, cloud_name=self._pick_cloud)
        print(f'[PCFD] viewport.cloud_pick -> {picked is not None}',
              flush=True)
        if picked is None:
            return
        callback = self._pick_callback
        if callable(callback):
            try:
                callback(picked)
            except Exception:
                pass

    # ---- 平面射线拾取（复核页测距/靠尺：屏幕点 → 相机射线 → 立面平面交点）----
    def ray_plane_pick(self, pos, plane_model):
        """Qt 屏幕坐标 → 相机射线与 plane_model（ax+by+cz+d=0）的交点。

        display 坐标换算与 project_points 同一约定：display 为物理像素、
        左下原点，Qt 事件为逻辑像素、左上原点（y 翻转 + dpr）。
        正交投影下射线起点在近平面、方向恒定，天然正确。
        """
        try:
            renderer = self._plotter.renderer
            fb_w, fb_h = (int(v) for v in renderer.GetSize())
            fb_w = max(1, fb_w); fb_h = max(1, fb_h)
            dpr = max(1.0, float(self._widget.devicePixelRatioF()))
            x = float(pos.x()) * dpr
            y = fb_h - float(pos.y()) * dpr
            endpoints = []
            for depth in (0.0, 1.0):
                renderer.SetDisplayPoint(x, y, depth)
                renderer.DisplayToWorld()
                w = renderer.GetWorldPoint()
                if abs(w[3]) < 1e-12:
                    return None
                endpoints.append(
                    np.array([w[0] / w[3], w[1] / w[3], w[2] / w[3]],
                             dtype=np.float64))
            direction = endpoints[1] - endpoints[0]
            if np.linalg.norm(direction) < 1e-12:
                return None
            from algorithms.facade.interactive_ruler import ray_plane_intersect
            return ray_plane_intersect(endpoints[0], direction, plane_model)
        except Exception as exc:
            print(f'[PCFD] viewport.ray_plane_pick_failed: {exc!r}',
                  flush=True)
            return None

    def enter_plane_pick_mode(self, plane_model, callback=None):
        """平面拾取：复用 PICK 模式的 6px 点击判定，回调直接给平面交点。"""
        print('[PCFD] viewport.enter_plane_pick_mode', flush=True)
        self._mode = self.InteractionMode.PICK
        self._plane_pick_model = np.asarray(plane_model, dtype=np.float64)
        self._plane_pick_callback = callback
        self._pick_callback = None

    def set_selection_enabled(self, enabled, cloud_name=None):
        pass  # 框选由 ROI 模式接管

    def set_pick_enabled(self, enabled, radius=14, cloud_name=None):
        if enabled:
            self._pick_radius = float(radius)
            if cloud_name:
                self._pick_cloud = cloud_name
        else:
            self._pick_callback = None

    def enter_pick_mode(self, cloud_name=None, pick_radius=8, callback=None):
        print('[PCFD] viewport.enter_pick_mode（会顶掉平面拾取）', flush=True)
        self._mode = self.InteractionMode.PICK
        self._pick_cloud = cloud_name or self._active_name
        self._pick_radius = float(pick_radius)
        self._pick_callback = callback
        self._plane_pick_model = None
        self._plane_pick_callback = None

    def exit_pick_mode(self):
        print('[PCFD] viewport.exit_pick_mode', flush=True)
        self._mode = self.InteractionMode.NAVIGATE
        self._pick_callback = None
        self._plane_pick_model = None
        self._plane_pick_callback = None
        self._registration_pick_state = None
        self.clear_pick_markers()

    def get_picked_point(self) -> np.ndarray | None:
        return self._last_picked_point

    # ---- 配准人工选点（源/目标交替，契约与 Open3D 版一致）----
    def enter_registration_pick_mode(self, source_cloud, target_cloud,
                                     callback, pick_radius=10):
        self._registration_pick_state = {
            'source_cloud': source_cloud,
            'target_cloud': target_cloud,
            'source_points': [],
            'target_points': [],
            'next_is_source': True,
        }
        self._mode = self.InteractionMode.PICK
        self._pick_radius = float(pick_radius)
        self._pick_cloud = source_cloud   # 第一次先选源站点
        self._pick_callback = self._on_registration_pick
        self._registration_callback = callback
        self.update_pick_markers()

    def _on_registration_pick(self, picked):
        state = getattr(self, '_registration_pick_state', None)
        if state is None:
            return
        cloud_name = picked.get('cloud_name')
        point = np.asarray(picked['point'], dtype=np.float64)
        expected = (state['source_cloud'] if state['next_is_source']
                    else state['target_cloud'])
        if cloud_name != expected:
            return
        if state['next_is_source']:
            state['source_points'].append(point)
            state['next_is_source'] = False
            self._pick_cloud = state['target_cloud']
        else:
            state['target_points'].append(point)
            state['next_is_source'] = True
            self._pick_cloud = state['source_cloud']
        self.update_pick_markers(state['source_points'],
                                 state['target_points'])
        callback = getattr(self, '_registration_callback', None)
        if callable(callback):
            try:
                callback(picked, state['next_is_source'])
            except Exception:
                pass

    def registration_pick_points(self):
        state = getattr(self, '_registration_pick_state', None) or {}
        return (list(state.get('source_points', [])),
                list(state.get('target_points', [])))

    def update_pick_markers(self, src_points=None, tgt_points=None):
        """配准选点标记：红=源，绿=目标。屏幕空间点精灵（约 12px），
        不随相机缩放；覆盖层渲染，永压点云之上。"""
        self.clear_pick_markers()
        for points, color in ((src_points or [], (1.0, 0.3, 0.3)),
                              (tgt_points or [], (0.3, 0.9, 0.5))):
            for p in points:
                try:
                    actor = self._point_marker_actor(p, color, 12,
                                                     'pick_marker')
                    self._pick_marker_actors.append(actor)
                except Exception:
                    pass
        self._render()

    def clear_pick_markers(self):
        for actor in self._pick_marker_actors:
            self._remove_visual_actor(actor)
        self._pick_marker_actors.clear()
        self._render()

    # ---- 覆盖层渲染管线（layer-1）：工具可视化独立管线，永压点云之上 ----
    def _get_overlay_renderer(self):
        """layer-1 覆盖层：共享主相机（自动同步），每帧清深度缓冲，
        本层内容（热力贴图/边框/测距/靠尺/标记点）不被点云遮挡。"""
        if self._overlay_renderer is None:
            import vtk
            rw = self._plotter.render_window
            main = self._plotter.renderer
            if rw.GetNumberOfLayers() < 2:
                rw.SetNumberOfLayers(2)
            ren = vtk.vtkRenderer()
            ren.SetLayer(1)
            ren.InteractiveOff()
            ren.SetActiveCamera(main.GetActiveCamera())
            ren.PreserveDepthBufferOff()
            rw.AddRenderer(ren)
            self._overlay_renderer = ren
        return self._overlay_renderer

    def _to_overlay(self, actor):
        """把 add_mesh/add_points 建好的 actor 挪进覆盖层管线。"""
        try:
            ren = self._get_overlay_renderer()
            self._plotter.renderer.RemoveActor(actor)
            ren.AddActor(actor)
        except Exception:
            pass
        return actor

    def _remove_visual_actor(self, actor):
        """移除可能在主层或覆盖层的可视化 actor。"""
        try:
            self._plotter.remove_actor(actor)
        except Exception:
            pass
        ren = self._overlay_renderer
        if ren is not None:
            try:
                ren.RemoveActor(actor)
            except Exception:
                pass

    def _point_marker_actor(self, point, color, size_px, name):
        """屏幕空间点标记：点精灵（非三维球体），尺寸只随分辨率像素，
        不随相机缩放变化；进覆盖层。"""
        poly = pv.PolyData(np.asarray(point, dtype=np.float64).reshape(1, 3))
        actor = self._plotter.add_points(
            poly, color=color, point_size=int(size_px),
            render_points_as_spheres=False, lighting=False, name=name)
        return self._to_overlay(actor)

    # ---- 复核页工具可视化（测距连线 / 交互靠尺），不入 BaseViewport 契约 ----
    def show_measure_line(self, p1, p2, color=(0.12, 0.39, 0.85)):
        """两点测距连线 + 端点标记（屏幕空间点精灵，约 12px）。"""
        self.clear_measure_visuals()
        try:
            actor = self._plotter.add_mesh(
                pv.Line(tuple(p1), tuple(p2)), color=color,
                line_width=3, lighting=False, name='measure_line')
            self._measure_actors.append(self._to_overlay(actor))
            for p in (p1, p2):
                actor = self._point_marker_actor(p, color, 12,
                                                 'measure_point')
                self._measure_actors.append(actor)
        except Exception:
            pass
        self._render()

    def clear_measure_visuals(self):
        for actor in self._measure_actors:
            self._remove_visual_actor(actor)
        self._measure_actors.clear()
        self._render()

    def show_ruler_overlay(self, render, exceeded=False):
        """交互靠尺渲染规格：corners(4,3) 尺面 + defect_segments 着色分段。

        分段按超限红 / 合格绿着色；尺面超限整体转红色调。
        """
        self.clear_ruler_overlay()
        try:
            corners = np.asarray(render.get('corners'), dtype=np.float64)
            if corners.shape == (4, 3):
                plate_color = ((0.85, 0.25, 0.22) if exceeded
                               else (0.30, 0.55, 0.95))
                plate = pv.PolyData(corners,
                                    faces=np.array([4, 0, 1, 2, 3]))
                actor = self._plotter.add_mesh(
                    plate, color=plate_color, opacity=0.35,
                    lighting=False, name='ruler_plate')
                try:
                    actor.ForceTranslucentOn()
                except Exception:
                    pass
                self._ruler_actors.append(self._to_overlay(actor))
                loop = np.vstack([corners, corners[0]])
                actor = self._plotter.add_mesh(
                    pv.lines_from_points(loop), color=plate_color,
                    line_width=3, lighting=False, name='ruler_outline')
                self._ruler_actors.append(self._to_overlay(actor))
            for seg in render.get('defect_segments') or []:
                color = ((0.86, 0.15, 0.15) if seg.get('exceed')
                         else (0.13, 0.65, 0.35))
                actor = self._plotter.add_mesh(
                    pv.Line(tuple(seg['start']), tuple(seg['end'])),
                    color=color, line_width=6, lighting=False,
                    name='ruler_defect')
                self._ruler_actors.append(self._to_overlay(actor))
            depression = render.get('depression_xyz')
            if depression is not None:
                actor = self._point_marker_actor(
                    depression, (0.95, 0.45, 0.10), 16,
                    'ruler_depression')
                self._ruler_actors.append(actor)
        except Exception:
            pass
        self._render()

    def clear_ruler_overlay(self):
        for actor in self._ruler_actors:
            self._remove_visual_actor(actor)
        self._ruler_actors.clear()
        self._render()

    def clear_tool_visuals(self):
        self.clear_measure_visuals()
        self.clear_ruler_overlay()
        self.clear_heatmap_quad()
        self.clear_pick_markers()

    # ---- 热力图贴图 quad（复核页选中立面呈现）----
    def show_heatmap_quad(self, image, origin, u_axis, v_axis, uv_bounds,
                          normal=None, lift=None):
        """把透明热力 PNG（RGBA）贴到立面四边面上（逐像素 alpha 镂空）。

        image 为检测时导出的 facade_NNN_{mode}_transparent.png（RGBA uint8，
        行 0 = v 最大）；uv_bounds=(u0,v0,u1,v1) 必须与 PNG 内容严格对应
        （见 interactive_ruler.transparent_png_uv_rect）。透明像素 RGB 为黑，
        故 actor 必须进透明通道（ForceTranslucentOn），否则镂空区变黑块。
        lift 缺省自适应：max(5cm, 跨度×0.1%)，保证贴图浮在残留几何之上。
        """
        self.clear_heatmap_quad()
        try:
            rgba = np.asarray(image)
            if rgba.ndim != 3 or rgba.shape[2] != 4:
                return
            rgba = np.ascontiguousarray(rgba.astype(np.uint8))
            origin = np.asarray(origin, dtype=np.float64).reshape(3)
            u_axis = np.asarray(u_axis, dtype=np.float64).reshape(3)
            v_axis = np.asarray(v_axis, dtype=np.float64).reshape(3)
            u0, v0, u1, v1 = np.asarray(uv_bounds,
                                        dtype=np.float64).reshape(4)
            if normal is None:
                n = np.cross(u_axis, v_axis)
                norm = np.linalg.norm(n)
                n = n / norm if norm > 1e-12 else np.zeros(3)
            else:
                n = np.asarray(normal, dtype=np.float64).reshape(3)
                norm = np.linalg.norm(n)
                n = n / norm if norm > 1e-12 else np.zeros(3)
            if lift is None:
                lift = max(0.05, 0.001 * max(u1 - u0, v1 - v0))
            offset = n * float(lift)
            corners = np.array([
                origin + u0 * u_axis + v0 * v_axis,
                origin + u1 * u_axis + v0 * v_axis,
                origin + u1 * u_axis + v1 * v_axis,
                origin + u0 * u_axis + v1 * v_axis,
            ]) + offset
            quad = pv.PolyData(corners, faces=np.array([4, 0, 1, 2, 3]))
            # pyvista 0.49 无 t_coords 便捷属性，走底层 VTK 接口；
            # PNG 行序：行 0 = v 最大（图像顶部），VTK t=1 为纹理顶部。
            # 因此 v 小的物理底部对应纹理底部 t=0，v 大的物理顶部对应 t=1。
            from vtk.util.numpy_support import numpy_to_vtk
            tcoords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]],
                               dtype=np.float32)
            quad.GetPointData().SetTCoords(numpy_to_vtk(tcoords))
            texture = pv.Texture(rgba)
            actor = self._plotter.add_mesh(quad, texture=texture,
                                           lighting=False,
                                           name='heatmap_quad')
            try:
                # 逐像素 alpha 混合：镂空区不写深度、不遮点云
                actor.ForceTranslucentOn()
            except Exception:
                pass
            self._heatmap_quad_actors.append(self._to_overlay(actor))
            self._add_heatmap_frame(corners)
        except Exception as exc:
            print(f'[PCFD] viewport.heatmap_quad_failed: {exc!r}', flush=True)
        self._render()

    def _add_heatmap_frame(self, corners):
        """四边面边缘高亮：外圈宽线低透明做发光，内圈亮线收边。"""
        loop = np.vstack([corners, corners[0]])
        lines = pv.lines_from_points(loop)
        glow = self._plotter.add_mesh(lines, color='#38BDF8', opacity=0.30,
                                      line_width=12, lighting=False,
                                      name='heatmap_frame_glow')
        try:
            glow.ForceTranslucentOn()
        except Exception:
            pass
        self._heatmap_quad_actors.append(self._to_overlay(glow))
        core = self._plotter.add_mesh(lines, color='#0284C7',
                                      line_width=4, lighting=False,
                                      name='heatmap_frame_core')
        self._heatmap_quad_actors.append(self._to_overlay(core))

    def show_heatmap_frame(self, corners_3d, lift=None):
        """仅画四边面边缘高亮（选中立面无热力 PNG 时的选中反馈）。"""
        self.clear_heatmap_quad()
        try:
            corners = np.asarray(corners_3d, dtype=np.float64).reshape(-1, 3)
            if len(corners) < 3:
                return
            if lift is None:
                span = float(np.linalg.norm(
                    corners.max(axis=0) - corners.min(axis=0)))
                lift = max(0.05, 0.001 * span)
            n = np.cross(corners[1] - corners[0],
                         corners[2] - corners[0])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                corners = corners + (n / norm) * float(lift)
            self._add_heatmap_frame(corners)
        except Exception as exc:
            print(f'[PCFD] viewport.heatmap_frame_failed: {exc!r}', flush=True)
        self._render()

    def clear_heatmap_quad(self):
        if self._heatmap_quad_actors:
            for actor in self._heatmap_quad_actors:
                self._remove_visual_actor(actor)
            self._heatmap_quad_actors = []
            self._render()

    # ---- 相机：快照/恢复 + 正视平面（复核页贴图呈现配套）----
    def snapshot_camera(self):
        try:
            camera = self._plotter.camera
            return {
                'position': tuple(camera.position),
                'focal_point': tuple(camera.focal_point),
                'up': tuple(camera.up),
                'parallel_scale': float(camera.parallel_scale),
                'parallel_projection': bool(camera.parallel_projection),
            }
        except Exception:
            return None

    def restore_camera(self, state):
        if not state:
            return
        try:
            camera = self._plotter.camera
            camera.position = tuple(state['position'])
            camera.focal_point = tuple(state['focal_point'])
            camera.up = tuple(state['up'])
            if state.get('parallel_projection'):
                self._plotter.enable_parallel_projection()
            else:
                self._plotter.disable_parallel_projection()
            camera.parallel_scale = float(state['parallel_scale'])
            self._render()
        except Exception:
            pass

    def focus_plane(self, normal, center, up, span):
        """正视立面：视线沿 -normal、正交投影、按跨度取景。

        相机位置显式放在 center + normal * 原视距处（不走 view_vector，
        避免其内部重取构图改动 focal_point）。span 为 (u 跨度, v 跨度)
        （米）。相机的上方向恒为世界上方向 +Z（z-up 约定），仅当平面
        法向接近 ±Z（水平面，up 退化）时才回退到传入的 up。
        parallel_scale 按屏幕上/侧向跨度留 10% 边距。
        """
        try:
            camera = self._plotter.camera
            normal = np.asarray(normal, dtype=np.float64)
            normal = normal / np.linalg.norm(normal)
            center = np.asarray(center, dtype=np.float64)
            up = np.asarray(up, dtype=np.float64)
            u_span, v_span = (float(s) for s in span)
            fb_w, fb_h = (int(v) for v in self._plotter.renderer.GetSize())
            aspect = max(fb_w, 1) / max(fb_h, 1)
            # z-up：相机上方向恒为 +Z；法向几乎平行 Z 时退化，回退传入 up
            if abs(float(normal[2])) < 0.99:
                up = np.array([0.0, 0.0, 1.0])
            elif np.linalg.norm(up) < 1e-12:
                up = np.array([0.0, 1.0, 0.0])
            # z-up 下屏幕上方向 = v 轴在垂直 normal 平面内的投影方向
            screen_up_span = v_span
            screen_side_span = u_span
            distance = float(np.linalg.norm(
                np.asarray(camera.position, dtype=np.float64)
                - np.asarray(camera.focal_point, dtype=np.float64)))
            if distance < 1e-6:
                distance = max(u_span, v_span) * 3.0
            camera.focal_point = tuple(center)
            camera.position = tuple(center + normal * distance)
            camera.up = tuple(up)
            self._plotter.enable_parallel_projection()
            camera.parallel_scale = max(screen_up_span * 0.5,
                                        screen_side_span * 0.5 / aspect) * 1.1
            self._render()
        except Exception as exc:
            print(f'[PCFD] viewport.focus_plane_failed: {exc!r}', flush=True)

    # ---- ROI 框选（复用 roi_selection.py 的覆盖层与控制器）----
    def enter_roi_selection(self, cloud_name=None, on_complete=None):
        """ROI 框选：ROISelectionController 接管，覆盖层锁定输入。"""
        try:
            self.clear_roi_visuals()
        except Exception:
            pass
        self._roi_on_complete = on_complete
        if self._roi_controller is not None:
            self.set_mode(self.InteractionMode.ROI)
            self._roi_controller.start(cloud_name, self._handle_roi_complete)

    def exit_roi_selection(self):
        try:
            if self._roi_controller is not None:
                self._roi_controller.cancel()
        except Exception:
            pass
        self._roi_on_complete = None
        self.set_mode(self.InteractionMode.NAVIGATE)

    def _handle_roi_complete(self, min_bound, max_bound, indices, p1=None, p2=None):
        cb = self._roi_on_complete
        if callable(cb):
            try:
                cb(min_bound, max_bound, indices, p1, p2)
            except Exception:
                pass

    def select_indices_in_rect(self, cloud_name, start, end):
        """屏幕矩形选框 → 框内点的全局行索引（ROISelectionController 调用）。"""
        name = cloud_name or self._active_name
        data = self.point_data.get(name)
        if data is None:
            return np.array([], dtype=np.int64)
        projected = self.project_points(data['pos'])
        if projected is None:
            return np.array([], dtype=np.int64)
        screen, valid = projected
        x1, x2 = sorted((start.x(), end.x()))
        y1, y2 = sorted((start.y(), end.y()))
        mask = (valid
                & (screen[:, 0] >= x1) & (screen[:, 0] <= x2)
                & (screen[:, 1] >= y1) & (screen[:, 1] <= y2))
        return np.flatnonzero(mask).astype(np.int64)

    def build_roi_obb(self, cloud_name, indices, screen_rect=None,
                      pad_px=2.0, depth_pad=0.0):
        """Build the view-aligned ROI box for the exact screen selection.

        The selected indices are authoritative.  The box is deliberately not
        a world-axis AABB: its axes are the current camera right/up/front
        basis, so rotating the camera cannot turn a screen selection into a
        misleading, oversized XYZ box.
        """
        data = self.point_data.get(cloud_name or self._active_name)
        if data is None or indices is None:
            return None
        pts = np.asarray(data.get('pos'), dtype=np.float64)
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        idx = idx[(idx >= 0) & (idx < len(pts))]
        if len(idx) == 0:
            return None
        cam = self._plotter.renderer.GetActiveCamera()
        pos = np.asarray(cam.GetPosition(), dtype=np.float64)
        focal = np.asarray(cam.GetFocalPoint(), dtype=np.float64)
        front = focal - pos; front /= max(np.linalg.norm(front), 1e-12)
        up = np.asarray(cam.GetViewUp(), dtype=np.float64)
        up -= front * np.dot(up, front); up /= max(np.linalg.norm(up), 1e-12)
        right = np.cross(front, up); right /= max(np.linalg.norm(right), 1e-12)
        up = np.cross(right, front); up /= max(np.linalg.norm(up), 1e-12)
        axes = np.column_stack((right, up, front))
        local = (pts[idx] - focal) @ axes
        lo, hi = local.min(axis=0), local.max(axis=0)
        if screen_rect is not None and pad_px > 0:
            # Convert pixel padding using local projected extent.  This is
            # intentionally small and never has a fixed 0.5 m world floor.
            x1, y1, x2, y2 = map(float, (screen_rect[0].x(), screen_rect[0].y(),
                                         screen_rect[1].x(), screen_rect[1].y()))
            span_px = max(abs(x2-x1), abs(y2-y1), 1.0)
            span_local = max(float(np.ptp(local[:, :2], axis=0).max()), 1e-6)
            pad = span_local * float(pad_px) / span_px
            lo[:2] -= pad; hi[:2] += pad
        lo[2] -= float(depth_pad); hi[2] += float(depth_pad)
        center_local = (lo + hi) * 0.5
        center = focal + center_local @ axes.T
        half = (hi - lo) * 0.5
        corners = np.asarray([center + np.array([sx*half[0], sy*half[1], sz*half[2]]) @ axes.T
                              for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        return {'center': center, 'axes': axes, 'half_extents': half,
                'corners': corners, 'indices': idx}

    def clear_roi_visuals(self):
        self._remove_roi_bbox()

    def show_roi_bbox(self, min_bound, max_bound, color=(1.0, 1.0, 1.0)):
        self._remove_roi_bbox()
        bounds = [min_bound[0], max_bound[0],
                  min_bound[1], max_bound[1],
                  min_bound[2], max_bound[2]]
        self._roi_bbox_actor = self._plotter.add_mesh(
            pv.Box(bounds=bounds), style='wireframe',
            color=color, line_width=2, name='roi_bbox')
        self._render()

    def show_roi_obb(self, center, axes, half_extents,
                     color=(1.0, 0.2, 0.2)):
        """Render an explicit oriented wire box (never pv.Box AABB)."""
        self._remove_roi_bbox()
        c = np.asarray(center, dtype=float); a = np.asarray(axes, dtype=float)
        h = np.asarray(half_extents, dtype=float)
        corners = np.asarray([c + np.array([sx*h[0], sy*h[1], sz*h[2]]) @ a.T
                              for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        edges = []
        for i in range(8):
            for bit in (0, 1, 2):
                j = i ^ (1 << bit)
                if i < j: edges.append((i, j))
        lines = np.asarray([[2, i, j] for i, j in edges], dtype=np.int64).ravel()
        mesh = pv.PolyData(corners); mesh.lines = lines
        self._roi_bbox_actor = self._plotter.add_mesh(mesh, color=color,
                                                       line_width=2, name='roi_bbox')
        self._render()

    def _remove_roi_bbox(self):
        if self._roi_bbox_actor is not None:
            try:
                self._plotter.remove_actor(self._roi_bbox_actor)
            except Exception:
                pass
            self._roi_bbox_actor = None
            self._render()

    # ------------------------------------------------------------------
    # 截图 / 生命周期
    # ------------------------------------------------------------------
    def save_screenshot(self, path: str):
        self._plotter.screenshot(str(path))

    def show_image(self, *args, **kwargs):
        pass

    def add_point_cloud(self, name=None, points=None, colors=None,
                        positions=None, point_size=0.75, **_ignored):
        self.add_cloud(name, points if points is not None else positions,
                       colors, point_size=point_size)

    def add_cloud_numpy(self, name=None, points=None, colors=None,
                        positions=None, point_size=0.75, **_ignored):
        self.add_cloud(name, points if points is not None else positions,
                       colors, point_size=point_size)

    def destroy(self):
        self.clear()
        try:
            self._widget.close()
        except Exception:
            pass

    def close(self):
        self.destroy()
