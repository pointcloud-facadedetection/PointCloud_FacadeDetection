import numpy as np
from config.settings import Config


class CameraController:
    def __init__(self, adapter, viewport_widget=None):
        self.adapter = adapter
        self.viewport_widget = viewport_widget
        self._state = None          # 保留，供 get/set_state 使用
        self._scene_scale_provider = None

    def get_state(self):
        ctr = self.adapter.get_view_control()
        if ctr is None:
            return self._state
        try:
            if self.is_orthographic():
                state = {
                    'mode': 'ortho',
                    'lookat': np.array(self._safe_get(ctr, 'get_lookat', [0, 0, 0]), dtype=float).tolist(),
                    'front': np.array(self._safe_get(ctr, 'get_front', [0, 0, -1]), dtype=float).tolist(),
                    'up': np.array(self._safe_get(ctr, 'get_up', [0, 1, 0]), dtype=float).tolist(),
                    'zoom': float(self._safe_get(ctr, 'get_zoom', 0.6)),
                    'fov': float(self._safe_get(ctr, 'get_field_of_view', getattr(Config, 'ORTHO_FOV_DEG', 5.0))),
                }
                self._state = state
                return self._state
        except Exception:
            pass
        try:
            self._state = ctr.convert_to_pinhole_camera_parameters()
        except Exception:
            # 如果转换失败，保留之前的状态
            pass
        return self._state

    def set_state(self, state):
        self._state = state
        ctr = self.adapter.get_view_control()
        if ctr is None or state is None:
            return
        # 在不进行针孔转换的情况下处理正交法状态词典
        try:
            if isinstance(state, dict) and state.get('mode') == 'ortho':
                lookat = np.asarray(state.get('lookat', [0, 0, 0]), dtype=float)
                front = np.asarray(state.get('front', [0, 0, -1]), dtype=float)
                up = np.asarray(state.get('up', [0, 1, 0]), dtype=float)
                zoom = float(state.get('zoom', 0.6))
                fov = float(state.get('fov', getattr(Config, 'ORTHO_FOV_DEG', 5.0)))
                try:
                    ctr.set_lookat(lookat)
                    ctr.set_front(front / (np.linalg.norm(front) + 1e-12))
                    ctr.set_up(up / (np.linalg.norm(up) + 1e-12))
                    current_fov = ctr.get_field_of_view()
                    step = fov - current_fov
                    if abs(step) > 1e-6:
                        ctr.change_field_of_view(step)
                    ctr.set_zoom(zoom)
                except Exception:
                    pass
                return
        except Exception:
            pass

        try:
            ctr.convert_from_pinhole_camera_parameters(state, allow_arbitrary=True)
        except TypeError:
            try:
                ctr.convert_from_pinhole_camera_parameters(state)
            except Exception:
                pass
        except Exception:
            pass

    def set_look_at(self, center, eye, up):
        ctr = self.adapter.get_view_control()
        if ctr is None:
            return
        try:
            ctr.set_lookat(np.asarray(center, dtype=float))
            ctr.set_front(np.asarray(center, dtype=float) - np.asarray(eye, dtype=float))
            ctr.set_up(np.asarray(up, dtype=float))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 正交投影控制 
    # ------------------------------------------------------------------
    def set_orthographic(self, enabled=True):
        """切换正交/透视投影。Open3D 中 FoV=5° 时自动进入正交模式。"""
        ctr = self.adapter.get_view_control()
        if ctr is None:
            return
        try:
            current_fov = ctr.get_field_of_view()
            target_fov = float(getattr(Config, 'ORTHO_FOV_DEG', 5.0)) if enabled else 60.0
            step = target_fov - current_fov
            if abs(step) > 1e-6:
                ctr.change_field_of_view(step)
        except Exception:
            pass

    def is_orthographic(self):
        ctr = self.adapter.get_view_control()
        if ctr is None:
            return False
        try:
            return ctr.get_field_of_view() <= 5.5
        except Exception:
            return False

    def project_points(self, points):
        """
        将 3D 点投影到视口屏幕坐标系（逻辑像素，左上角为原点）。
        返回 (screen_coords, valid_mask)，其中 screen_coords 形状为 (N, 3)，
        前两维为 x, y（与 Qt 事件坐标一致），第三维为深度值。
        """
        ctr = self.adapter.get_view_control()
        if ctr is None or self.viewport_widget is None:
            return None

        try:
            if not self.is_orthographic():
                params = ctr.convert_to_pinhole_camera_parameters()
                intrinsic = params.intrinsic.intrinsic_matrix
                extrinsic = params.extrinsic
                pts = np.asarray(points, dtype=np.float64)
                hom = np.c_[pts, np.ones(len(pts))]
                cam = (extrinsic @ hom.T).T
                z = cam[:, 2]
                valid = z > 1e-9
                uvw = (intrinsic @ cam[:, :3].T).T
                screen = np.zeros((len(pts), 3), dtype=np.float64)
                screen[valid, 0] = uvw[valid, 0] / z[valid]
                screen[valid, 1] = uvw[valid, 1] / z[valid]
                screen[valid, 2] = z[valid]
                # convert to logical pixels
                dpr = self._device_pixel_ratio()
                if dpr and dpr != 1.0:
                    screen[:, 0] /= dpr
                    screen[:, 1] /= dpr
                return screen, valid
        except Exception:
            # 转而采用正交路径
            pass

        try:
            w, h, dpr = self._viewport_metrics()
            lookat = np.asarray(self._safe_get(ctr, 'get_lookat', [0, 0, 0]), dtype=float)
            front = np.asarray(self._safe_get(ctr, 'get_front', [0, 0, -1]), dtype=float)
            up = np.asarray(self._safe_get(ctr, 'get_up', [0, 1, 0]), dtype=float)
            # 归一化
            front = front / (np.linalg.norm(front) + 1e-12)
            up = up - front * float(np.dot(front, up))
            up = up / (np.linalg.norm(up) + 1e-12)
            right = np.cross(front, up)
            right = right / (np.linalg.norm(right) + 1e-12)

            try:
                zoom = float(self._safe_get(ctr, 'get_zoom', 0.6))
            except Exception:
                zoom = 0.6
            scene_scale = self._estimate_scene_scale()
            scene_factor = max(0.2, min(scene_scale / 10.0, 3.0))
            pan_base = float(getattr(Config, 'PAN_BASE_SPEED', 0.06))
            zoom_factor = 0.6 / max(zoom, 0.02)
            world_per_pixel = pan_base * zoom_factor * scene_factor / max(1.0, dpr)
            world_per_pixel = max(world_per_pixel, 1e-6)

            pts = np.asarray(points, dtype=np.float64)
            d = pts - lookat.reshape(1, 3)
            u = d @ right.reshape(3,)
            v = d @ up.reshape(3,)
            z = d @ front.reshape(3,)
            screen = np.zeros((len(pts), 3), dtype=np.float64)
            screen[:, 0] = (w * 0.5) + (u / world_per_pixel)
            screen[:, 1] = (h * 0.5) - (v / world_per_pixel)
            screen[:, 2] = z
            valid = np.isfinite(screen[:, 0]) & np.isfinite(screen[:, 1])
            return screen, valid
        except Exception:
            return None

    # ---------------- 内部辅助函数 ----------------
    def _safe_get(self, ctr, attr, default):
        try:
            fn = getattr(ctr, attr)
            return fn() if callable(fn) else default
        except Exception:
            return default

    def _viewport_metrics(self):
        w = h = 1
        dpr = 1.0
        vw = self.viewport_widget
        try:
            if vw is not None:
                w = max(1, int(vw.width()))
                h = max(1, int(vw.height()))
                if hasattr(vw, "devicePixelRatioF"):
                    dpr = float(vw.devicePixelRatioF())
                elif hasattr(vw, "devicePixelRatio"):
                    dpr = float(vw.devicePixelRatio())
        except Exception:
            pass
        return w, h, dpr

    def _device_pixel_ratio(self):
        try:
            if hasattr(self.viewport_widget, "devicePixelRatioF"):
                return float(self.viewport_widget.devicePixelRatioF())
            elif hasattr(self.viewport_widget, "devicePixelRatio"):
                return float(self.viewport_widget.devicePixelRatio())
            elif hasattr(self.viewport_widget, "windowHandle") and self.viewport_widget.windowHandle() is not None:
                wh = self.viewport_widget.windowHandle()
                if hasattr(wh, "devicePixelRatio"):
                    return float(wh.devicePixelRatio())
        except Exception:
            pass
        return 1.0

    def _estimate_scene_scale(self):
        try:
            if callable(self._scene_scale_provider):
                s = float(self._scene_scale_provider())
                if s and s > 0:
                    return s
        except Exception:
            pass
        return 1.0

    def set_scene_scale_provider(self, provider):
        self._scene_scale_provider = provider