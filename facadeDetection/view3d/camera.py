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
            center = np.asarray(center, dtype=float).reshape(3)
            eye = np.asarray(eye, dtype=float).reshape(3)
            # Open3D ViewControl 的 ``front`` 指向观察点到相机的位置，
            # 即 camera_position = lookat + front * distance。它与通常
            # 相机“视线方向”(center - eye) 的定义正好相反。
            front = eye - center
            if np.linalg.norm(front) < 1e-12:
                return
            front = front / (np.linalg.norm(front) + 1e-12)

            up = np.asarray(up, dtype=float).reshape(3)
            up = up - front * float(np.dot(front, up))
            if np.linalg.norm(up) < 1e-12:
                candidates = (
                    np.array([0.0, 0.0, 1.0], dtype=float),
                    np.array([0.0, 1.0, 0.0], dtype=float),
                    np.array([1.0, 0.0, 0.0], dtype=float),
                )
                up = min(
                    candidates,
                    key=lambda axis: abs(float(np.dot(axis, front))),
                )
                up = up - front * float(np.dot(front, up))
            up = up / (np.linalg.norm(up) + 1e-12)

            ctr.set_lookat(center)
            ctr.set_front(front)
            ctr.set_up(up)
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

    def ensure_pinhole_projection(self, field_of_view: float = 35.0) -> bool:
        """确保当前视图可导出针孔参数，供精确拾取和深度反投影使用。"""
        ctr = self.adapter.get_view_control()
        if ctr is None:
            return False
        try:
            current = float(ctr.get_field_of_view())
            target = max(10.0, min(float(field_of_view), 90.0))
            if current <= 5.5:
                ctr.change_field_of_view(target - current)
                self.adapter.poll()
            ctr.convert_to_pinhole_camera_parameters()
            return True
        except Exception:
            return False

    # 屏幕坐标反投影到平面3D点（ROI BBox核心）
    def unproject_to_plane(self, screen_x, screen_y, plane_model, lookat=None):
        """
        将屏幕坐标反投影到指定平面的3D点（正交投影专用）。

        正交投影下，屏幕坐标 (sx, sy) 对应的3D射线为：
            X(t) = lookat + dx * right + dy * up + t * front
        其中 dx = (sx - w/2) * world_per_pixel, dy = -(sy - h/2) * world_per_pixel
        
        与平面 dot(n, X) + d = 0 相交，解得 t：
            t = -(dot(n, lookat) + d + dot(n, dx*right + dy*up)) / dot(n, front)

        Args:
            screen_x, screen_y: 屏幕坐标（逻辑像素，左上角为原点）
            plane_model: [nx, ny, nz, d] 平面方程 dot(n, X) + d = 0
            lookat: 可选，相机焦点。默认从 view control 获取。

        Returns:
            np.ndarray: 3D点坐标 (3,)，若射线与平面平行则返回 None
        """
        ctr = self.adapter.get_view_control()
        if ctr is None or self.viewport_widget is None:
            return None

        if not self.is_orthographic():
            return None

        try:
            w, h, dpr = self._viewport_metrics()
            if lookat is None:
                lookat = np.asarray(self._safe_get(ctr, 'get_lookat', [0, 0, 0]), dtype=float)
            front = np.asarray(self._safe_get(ctr, 'get_front', [0, 0, -1]), dtype=float)
            up = np.asarray(self._safe_get(ctr, 'get_up', [0, 1, 0]), dtype=float)

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

            dx = (screen_x - w * 0.5) * world_per_pixel
            dy = -(screen_y - h * 0.5) * world_per_pixel

            ray_origin = lookat + dx * right + dy * up

            n = np.asarray(plane_model[:3], dtype=float)
            n = n / (np.linalg.norm(n) + 1e-12)
            d_plane = float(plane_model[3])

            denom = float(np.dot(n, front))
            if abs(denom) < 1e-12:
                return None

            t = -(np.dot(n, ray_origin) + d_plane) / denom
            point = ray_origin + t * front
            return point.astype(np.float64)
        except Exception:
            return None

    def get_camera_basis(self):
        """
        获取当前相机坐标系的三个正交基向量。
        
        Returns:
            tuple: (front, up, right) 均为归一化 np.ndarray(3,)
            若获取失败则返回 (None, None, None)
        """
        ctr = self.adapter.get_view_control()
        if ctr is None:
            return None, None, None
        try:
            front = np.asarray(self._safe_get(ctr, 'get_front', [0, 0, -1]), dtype=float)
            up = np.asarray(self._safe_get(ctr, 'get_up', [0, 1, 0]), dtype=float)
            
            front = front / (np.linalg.norm(front) + 1e-12)
            up = up - front * float(np.dot(front, up))
            up = up / (np.linalg.norm(up) + 1e-12)
            right = np.cross(front, up)
            right = right / (np.linalg.norm(right) + 1e-12)
            return front, up, right
        except Exception:
            return None, None, None
        
    def get_world_per_pixel(self):
        """
        基于Open3D正交投影参数计算真实的world_per_pixel。
        """
        ctr = self.adapter.get_view_control()
        if ctr is None:
            return None
        try:
            w, h, dpr = self._viewport_metrics()
            h = max(h, 1)
            
            # 获取Open3D真实的zoom和场景范围
            zoom = float(self._safe_get(ctr, 'get_zoom', 0.6))
            
            # 场景范围：从view_control的bounding box获取（最准确）
            scene_scale = self._estimate_scene_scale()
            scene_scale = max(scene_scale, 1.0)
            
            # Open3D正交投影核心公式:
            try:
                # 尝试从view_control获取bounding box
                bbox = ctr.get_bounding_box()
                if bbox is not None:
                    o3d_extent = float(bbox.get_max_extent())
                    if o3d_extent > 0:
                        scene_scale = o3d_extent
            except Exception:
                pass
            
            wpp = 2.0 * zoom * scene_scale / (h * dpr)
            wpp = max(wpp, 1e-9)
            
            return float(wpp)
        except Exception:
            return None
        
    def project_points(self, points):
        """
        使用 Open3D 当前渲染相机将 3D 点投影到 Qt 视口坐标。

        透视视图走针孔参数；正交视图不能导出针孔矩阵，改用 lookat / zoom。
        """
        ctr = self.adapter.get_view_control()
        if ctr is None or self.viewport_widget is None:
            return None

        try:
            pts = np.asarray(points, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[1] != 3:
                return None
            if self.is_orthographic():
                return self._project_points_orthographic(pts)
            params = ctr.convert_to_pinhole_camera_parameters()
            intrinsic = np.asarray(params.intrinsic.intrinsic_matrix, dtype=np.float64)
            extrinsic = np.asarray(params.extrinsic, dtype=np.float64)
            rotation = extrinsic[:3, :3]
            translation = extrinsic[:3, 3]
            cam = pts @ rotation.T + translation
            z = cam[:, 2]

            screen = np.zeros((len(pts), 3), dtype=np.float64)
            valid = np.isfinite(cam).all(axis=1) & (z > 1e-9)
            screen[valid, 0] = (
                intrinsic[0, 0] * cam[valid, 0] / z[valid]
                + intrinsic[0, 2]
            )
            screen[valid, 1] = (
                intrinsic[1, 1] * cam[valid, 1] / z[valid]
                + intrinsic[1, 2]
            )
            screen[:, 2] = z

            # Open3D 相机内参采用帧缓冲物理像素，Qt 事件采用逻辑像素。
            dpr = self._device_pixel_ratio()
            if dpr > 0 and dpr != 1.0:
                screen[:, :2] /= dpr
            return screen, valid
        except Exception:
            try:
                return self._project_points_orthographic(
                    np.asarray(points, dtype=np.float64)
                )
            except Exception:
                return None

    def _project_points_orthographic(self, pts):
        """把世界点投到正交视图的逻辑像素坐标。"""
        ctr = self.adapter.get_view_control()
        front, up, right = self.get_camera_basis()
        if ctr is None or front is None:
            return None
        lookat = np.asarray(self._safe_get(ctr, 'get_lookat', [0, 0, 0]), dtype=float)
        world_per_pixel = self.get_world_per_pixel()
        if world_per_pixel is None or world_per_pixel <= 0:
            return None
        width, height, dpr = self._viewport_metrics()
        relative = pts - lookat
        screen = np.zeros((len(pts), 3), dtype=np.float64)
        screen[:, 0] = (
            width * dpr * 0.5 + (relative @ right) / world_per_pixel
        ) / max(dpr, 1e-6)
        screen[:, 1] = (
            height * dpr * 0.5 - (relative @ up) / world_per_pixel
        ) / max(dpr, 1e-6)
        screen[:, 2] = relative @ front
        valid = np.isfinite(screen).all(axis=1)
        return screen, valid

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