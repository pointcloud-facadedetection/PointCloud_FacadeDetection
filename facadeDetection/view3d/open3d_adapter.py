import numpy as np
import open3d as o3d
import threading
import time


class Open3DAdapter:
    MIN_POINT_SIZE = 0.01
    MAX_POINT_SIZE = 1.0
    MIN_POINT_PIXEL_SIZE = 0.5
    MAX_POINT_PIXEL_SIZE = 5.0

    def __init__(self):
        self.vis = None
        self.geometries = {}
        self._owner_thread_id = None
        self._destroyed = False
        # GLFW must continue to receive events, but rendering is demand-driven.
        self._scene_dirty = False
        self._render_pending = False
        self._interaction_active = False
        self._render_enabled = True
        self._last_render_time = 0.0
        self._last_event_poll_time = 0.0
        self._event_poll_interval = 1.0 / 30.0
        self._idle_render_interval = 0.20
        self._interaction_render_interval = 1.0 / 30.0

    def _assert_owner(self):
        """Visualizer/GLFW is single-threaded; fail early instead of racing WGL."""
        if self._owner_thread_id is not None and threading.get_ident() != self._owner_thread_id:
            raise RuntimeError('Open3D visualization must run on the GUI thread')
        if self._destroyed:
            return False
        return self.vis is not None

    def create_window(self, title, width=1280, height=960, visible=True):
        if self._owner_thread_id is not None and threading.get_ident() != self._owner_thread_id:
            raise RuntimeError('Open3D window must be created on the GUI thread')
        self._owner_thread_id = threading.get_ident()
        self._destroyed = False
        self.vis = o3d.visualization.Visualizer()
        ok = self.vis.create_window(
            window_name=title,
            width=width,
            height=height,
            visible=visible,
        )
        if not ok:
            raise RuntimeError("Open3D Visualizer create_window returned False")
        self.request_render('window.created')
        return self.vis

    def request_render(self, reason='unknown'):
        if not self._destroyed and self.vis is not None:
            self._scene_dirty = True
            self._render_pending = True

    def begin_interaction(self):
        if not self._destroyed:
            self._interaction_active = True
            self.request_render('interaction.begin')

    def end_interaction(self):
        if not self._destroyed:
            self._interaction_active = False
            self.request_render('interaction.end')
            # The final camera state must be presented without waiting for the
            # idle cadence after a drag/wheel gesture.
            self._last_render_time = 0.0

    def set_render_enabled(self, enabled):
        self._render_enabled = bool(enabled)
        if self._render_enabled:
            self.request_render('render.enabled')

    def configure_render_options(self):
        if self.vis is None:
            return
        opt = self.vis.get_render_option()
        # Match the Corporate Clean viewport token (#111827).
        opt.background_color = np.array([17 / 255, 24 / 255, 39 / 255])
        opt.point_size = self.MIN_POINT_PIXEL_SIZE
        opt.show_coordinate_frame = True
        self.request_render('render.options')

    def add_geometry(self, name, geometry, reset_bounding_box=False):
        if not self._assert_owner():
            return
        old = self.geometries.get(name)
        if old is not None:
            try:
                self.vis.remove_geometry(old, reset_bounding_box=False)
            except Exception:
                pass
            self.geometries.pop(name, None)
            del old
        self.geometries[name] = geometry
        self.vis.add_geometry(geometry, reset_bounding_box=reset_bounding_box)
        self.request_render('geometry.add')

    def update_geometry(self, geometry):
        """Upload changed attributes without removing/re-adding geometry."""
        if self._assert_owner():
            self.vis.update_geometry(geometry)
            self.request_render('geometry.update')

    def remove_geometry(self, name):
        if not self._assert_owner():
            return
        geom = self.geometries.pop(name, None)
        if geom is not None:
            try:
                self.vis.remove_geometry(geom, reset_bounding_box=False)
            except Exception:
                pass
            self.request_render('geometry.remove')

    def clear(self):
        if not self._assert_owner():
            return
        try:
            self.vis.clear_geometries()
        except Exception:
            for name in list(self.geometries.keys()):
                self.remove_geometry(name)
        self.geometries.clear()
        self.request_render('scene.clear')

    def set_point_size(self, size):
        if self._assert_owner():
            # 将归一化的值映射到 Open3D 的像素范围
            value = max(self.MIN_POINT_SIZE, min(float(size), self.MAX_POINT_SIZE))
            ratio = ((value - self.MIN_POINT_SIZE) /
                     (self.MAX_POINT_SIZE - self.MIN_POINT_SIZE))
            pixel_size = (self.MIN_POINT_PIXEL_SIZE + ratio *
                          (self.MAX_POINT_PIXEL_SIZE - self.MIN_POINT_PIXEL_SIZE))
            self.vis.get_render_option().point_size = pixel_size
            self.request_render('point_size')

    # TODO(poll): [视口渲染刷新]
    def poll(self):
        if not self._assert_owner():
            return False
        now = time.monotonic()
        if now - self._last_event_poll_time < self._event_poll_interval:
            return False
        self._last_event_poll_time = now
        self.vis.poll_events()
        if not self._render_enabled or not self._render_pending:
            return False
        interval = (self._interaction_render_interval
                    if self._interaction_active else self._idle_render_interval)
        if now - self._last_render_time < interval:
            return False
        self.vis.update_renderer()
        self._last_render_time = now
        self._scene_dirty = False
        self._render_pending = False
        return True

    def get_view_control(self):
        if not self._assert_owner():
            return None
        return self.vis.get_view_control()

    def capture_screen(self):
        if not self._assert_owner():
            return None
        self.vis.update_renderer()
        self._last_render_time = time.monotonic()
        self._scene_dirty = False
        self._render_pending = False
        return self.vis.capture_screen_float_buffer(do_render=False)

    def capture_color_depth_camera(self, point_size=5.0):
        """在同一渲染帧捕获彩色图、相机深度和针孔相机参数。"""
        if not self._assert_owner():
            return None
        render_option = self.vis.get_render_option()
        old_point_size = float(render_option.point_size)
        try:
            render_option.point_size = float(point_size)
            self.vis.poll_events()
            self.vis.update_renderer()
            color = self.vis.capture_screen_float_buffer(do_render=True)
            depth = self.vis.capture_depth_float_buffer(do_render=False)
            camera = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
            return color, depth, camera
        finally:
            render_option.point_size = old_point_size
            self.vis.update_renderer()

    def destroy(self):
        """Destroy the native window once, while GLFW is still available.
        策略：
        1. 先标记销毁状态，阻止后续操作
        2. 清理几何体引用（不触发 Open3D 调用）
        3. 使用 try/except 包裹 destroy_window，吞掉 GLFW 错误
        4. 延迟释放 vis 引用，避免半销毁状态访问
        """
        if self._destroyed:
            return
    
        # 线程安全检查
        if self._owner_thread_id is not None and threading.get_ident() != self._owner_thread_id:
            raise RuntimeError('Open3D window must be destroyed on the GUI thread')
    
        # Step 1: 标记销毁状态
        self._destroyed = True
        self._render_enabled = False
        self._render_pending = False
        self._scene_dirty = False
    
        # Step 2: 保存并清空引用（先于 Open3D 调用）
        vis = self.vis
        self.vis = None
        self.geometries.clear()
    
        # Step 3: 安全销毁窗口
        if vis is not None:
            try:
                # 先清除所有几何体，减少 destroy_window 的工作量
                try:
                    vis.clear_geometries()
                except Exception:
                    pass
            
                # 最终销毁
                vis.destroy_window()
            except Exception as e:
                # 吞掉所有 GLFW/Open3D 关闭阶段的错误
                print(f'[Open3DAdapter] destroy_window warning (safe to ignore): {e}',
                      flush=True)
            finally:
                # 确保引用释放
                del vis
