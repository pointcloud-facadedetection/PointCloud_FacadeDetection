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

BACKGROUND = '#111827'


class _RenderQueue(QObject):
    """跨线程渲染请求编组：worker 只发信号，VTK 对象只在 GUI 线程触碰。"""

    color = Signal(str, object)
    points = Signal(str, object, object)


class _RenderCounterShim:
    """main_window 首帧等待读取 viewport._adapter._frames_rendered 的兼容垫片。"""

    def __init__(self):
        self._frames_rendered = 0


class PyVistaViewport(BaseViewport):
    """Qt 原生 PyVista 视口，接口与 Open3DViewport 对齐。"""

    def __init__(self, parent=None):
        from pyvistaqt import QtInteractor

        self._widget = QtInteractor(parent)
        self._widget.setObjectName('pyvistaViewport')
        self._plotter = self._widget  # QtInteractor 本身即 Plotter API
        self._plotter.set_background(BACKGROUND)

        # 云数据模型与 view3d.scene.PointCloudScene 对齐
        self.point_data: dict[str, dict] = {}
        self._polys: dict[str, pv.PolyData] = {}
        self._actors: dict[str, object] = {}
        self._bbox_actors: dict[str, object] = {}
        self._active_name = None

        self._render_queue = _RenderQueue(self._widget)
        self._render_queue.color.connect(
            self.update_cloud_color, Qt.QueuedConnection)
        self._render_queue.points.connect(
            self.update_cloud_points, Qt.QueuedConnection)
        self._adapter = _RenderCounterShim()  # 首帧等待兼容垫片
        self._render_enabled = True
        self._pick_callback = None
        self._last_picked_point = None

        # VTK 按需渲染：渲染成功后递增帧计数
        try:
            self._widget.render_window.AddObserver(
                'EndEvent', self._on_render_end)
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
    def add_cloud(self, name, positions, colors=None, point_size=0.5):
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
        self._active_name = name
        self._upload_cloud(name, reset_view=first_cloud)

    def _upload_cloud(self, name, reset_view=False):
        data = self.point_data[name]
        poly = self._polys.get(name)
        if poly is None:
            poly = pv.PolyData(data['pos'])
            self._polys[name] = poly
        else:
            poly.points[:] = data['pos']
        poly.point_data['colors'] = (data['color'] * 255).astype(np.uint8)
        actor = self._actors.get(name)
        if actor is None:
            actor = self._plotter.add_points(
                poly, scalars='colors', rgb=True,
                point_size=max(1, int(round(data['size'] * 3))),
                render_points_as_spheres=False, name=f'cloud::{name}')
            self._actors[name] = actor
        else:
            poly.Modified()
        if reset_view:
            self._plotter.reset_camera()
        self._render()

    def update_cloud_color(self, name, colors):
        data = self.point_data.get(name)
        if data is None:
            return
        data['color'] = np.ascontiguousarray(
            normalize_colors(colors, len(data['pos'])).astype(np.float32))
        data.pop('display_proxy_lookup', None)
        poly = self._polys.get(name)
        if poly is not None:
            poly.point_data['colors'][:] = (data['color'] * 255).astype(np.uint8)
            poly.Modified()
        self._render()

    def update_cloud_points(self, name, positions, colors=None):
        data = self.point_data.get(name)
        if data is None:
            return
        data['pos'] = np.ascontiguousarray(
            np.asarray(positions, dtype=np.float32).reshape(-1, 3))
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
        if self._active_name == name:
            self._active_name = next(iter(self.point_data), None)
        self._render()

    def clear(self):
        for name in list(self._actors):
            self._plotter.remove_actor(self._actors[name])
        for name in list(self._bbox_actors):
            self._plotter.remove_actor(self._bbox_actors[name])
        self._actors.clear()
        self._bbox_actors.clear()
        self._polys.clear()
        self.point_data.clear()
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
            value = max(1, int(round(float(size) * 3)))
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
    # 投影 / 拾取
    # ------------------------------------------------------------------
    def project_points(self, points):
        """3D 点投影为屏幕坐标（逻辑像素）。返回 (screen, valid) 或 None。"""
        try:
            import vtk
            renderer = self._plotter.renderer
            height = self._widget.height()
            pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
            screen = np.zeros((len(pts), 2), dtype=np.float64)
            valid = np.zeros(len(pts), dtype=bool)
            coord = vtk.vtkCoordinate()
            coord.SetCoordinateSystemToWorld()
            for i, (x, y, z) in enumerate(pts):
                coord.SetValue(float(x), float(y), float(z))
                d = coord.GetComputedDisplayValue(renderer)
                screen[i] = (d[0], height - d[1])   # VTK display 原点在左下
                valid[i] = 0.0 < d[2] < 1.0
            return screen, valid
        except Exception:
            return None

    def set_selection_enabled(self, enabled, cloud_name=None):
        pass  # ROI 交互后续迁移，先空实现保证接口可用

    def set_pick_enabled(self, enabled, radius=14, cloud_name=None):
        if not enabled:
            self._pick_callback = None

    def enter_pick_mode(self, cloud_name=None, pick_radius=8, callback=None):
        self._pick_callback = callback
        try:
            self._plotter.enable_point_picking(
                callback=self._on_picked, show_message=False,
                use_picker='point')
        except Exception:
            pass

    def exit_pick_mode(self):
        self._pick_callback = None
        try:
            self._plotter.disable_picking()
        except Exception:
            pass

    def _on_picked(self, picked, *_args):
        if picked is None:
            return
        point = getattr(picked, 'points', None)
        if point is not None and len(point):
            self._last_picked_point = np.asarray(point[0], dtype=np.float64)
        if self._pick_callback is not None:
            try:
                self._pick_callback(self._last_picked_point)
            except Exception:
                pass

    def handle_pick_screen(self, pos):
        pass  # Qt 侧拾取走 VTK picker，屏幕坐标入口暂不需要

    def pick_at_screen(self, pos, cloud_name=None):
        return self._last_picked_point

    def get_picked_point(self) -> np.ndarray | None:
        return self._last_picked_point

    def registration_pick_points(self):
        return [], []

    def update_pick_markers(self, src_points=None, tgt_points=None):
        pass

    def clear_pick_markers(self):
        pass

    # ------------------------------------------------------------------
    # ROI（占位：视觉层后续迁移，接口先对齐）
    # ------------------------------------------------------------------
    def enter_roi_selection(self, *args, **kwargs):
        pass

    def exit_roi_selection(self, *args, **kwargs):
        pass

    def clear_roi_visuals(self):
        pass

    def show_roi_bbox(self, *args, **kwargs):
        pass

    # ------------------------------------------------------------------
    # 截图 / 生命周期
    # ------------------------------------------------------------------
    def save_screenshot(self, path: str):
        self._plotter.screenshot(str(path))

    def show_image(self, *args, **kwargs):
        pass

    def add_point_cloud(self, name=None, points=None, colors=None,
                        positions=None, point_size=0.5, **_ignored):
        self.add_cloud(name, points if points is not None else positions,
                       colors, point_size=point_size)

    def add_cloud_numpy(self, name=None, points=None, colors=None,
                        positions=None, point_size=0.5, **_ignored):
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
