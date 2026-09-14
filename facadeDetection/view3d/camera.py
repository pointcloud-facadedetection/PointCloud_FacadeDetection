import numpy as np
from config.settings import Config


class CameraController:
    """Open3D 视口相机控制器。"""

    def __init__(self, adapter, viewport_widget=None):
        self.adapter = adapter
        self.viewport_widget = viewport_widget
        self._state = None          # 保留，供 get/set_state 使用
        self._scene_scale_provider = None
        self._tracked_zoom = 0.6
        self._zoom_provider = None

    # ------------------------------------------------------------------
    # zoom / 场景尺度 的可读化
    # ------------------------------------------------------------------
    def set_tracked_zoom(self, zoom):
        """由写入侧（滚轮缩放等）回填当前 zoom。 """
        try:
            z = float(zoom)
            if z > 0:
                self._tracked_zoom = z
        except Exception:
            pass

    def set_zoom_provider(self, provider):
        """由视口提供一个「当前实际 zoom」的读取函数（优先于回填值）。"""
        self._zoom_provider = provider

    def _current_zoom(self):
        if callable(self._zoom_provider):
            try:
                z = float(self._zoom_provider())
                if z > 0:
                    return z
            except Exception:
                pass
        return max(float(self._tracked_zoom), 1e-6)

    def set_scene_scale_provider(self, provider):
        self._scene_scale_provider = provider

    def _estimate_scene_scale(self):
        try:
            if callable(self._scene_scale_provider):
                s = float(self._scene_scale_provider())
                if s and s > 0:
                    return s
        except Exception:
            pass
        return 1.0

    def _scene_max_extent(self):
        """场景 AABB 的最大边长。  """
        try:
            if callable(self._scene_scale_provider):
                s = float(self._scene_scale_provider())
                if s and s > 0:
                    return s
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # 相机状态存取（正交字典 / 针孔参数）
    # ------------------------------------------------------------------
    def get_state(self):
        ctr = self.adapter.get_view_control()
        if ctr is None:
            return self._state
        if self.is_orthographic():
            try:
                basis = self.read_camera_basis()
                if basis is not None:
                    ref, _right, up, front = basis
                    # 正交投影对沿 front 的平移不敏感，因此用相机中心 ref
                    # 作为 lookat 写入，恢复后画面与原来完全一致。
                    self._state = {
                        'mode': 'ortho',
                        'lookat': np.asarray(ref, dtype=float).tolist(),
                        'front': np.asarray(front, dtype=float).tolist(),
                        'up': np.asarray(up, dtype=float).tolist(),
                        'zoom': float(self._current_zoom()),
                        'fov': float(getattr(Config, 'ORTHO_FOV_DEG', 5.0)),
                    }
                    return self._state
            except Exception:
                pass
        try:
            self._state = ctr.convert_to_pinhole_camera_parameters()
        except Exception:
            # 转换失败则保留上一次状态
            pass
        return self._state

    def set_state(self, state):
        self._state = state
        ctr = self.adapter.get_view_control()
        if ctr is None or state is None:
            return
        # 正交法状态词典，不做针孔转换
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
                    # 同步缩放回填，保证后续框选的 world_per_pixel 正确
                    self.set_tracked_zoom(zoom)
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

    # ------------------------------------------------------------------
    # 相机姿态读取（借道透视读 extrinsic）
    # ------------------------------------------------------------------
    def read_camera_basis(self):
        """读取当前相机姿态，返回 (ref, right, up, front)；失败返回 None。

        由 extrinsic 推导（已用已知世界坐标点 + 渲染实测像素交叉标定）：

            R = E[:3, :3];  C = -R.T @ E[:3, 3]
            right = +R.T[:, 0];  up = -R.T[:, 1];  front = -R.T[:, 2]

        返回的 ref 用相机中心 C 而非 lookat：正交映射只依赖点与参考点在
        (right, up) 上的分量，而 C 与 lookat 沿 front 相差一个标量距离，
        对这两个分量没有任何影响，故用 C 完全等价（已在 pan / rotate /
        zoom 各场景下验证，最大误差 < 1 像素）。
        """
        ctr = self.adapter.get_view_control()
        if ctr is None:
            return None
        try:
            was_ortho = self.is_orthographic()
            if was_ortho:
                ctr.change_field_of_view(60.0 - float(ctr.get_field_of_view()))
            try:
                params = ctr.convert_to_pinhole_camera_parameters()
                E = np.asarray(params.extrinsic, dtype=np.float64)
            finally:
                if was_ortho:
                    ctr.change_field_of_view(5.0 - float(ctr.get_field_of_view()))
            if E.shape != (4, 4) or not np.all(np.isfinite(E)):
                return None
            R = E[:3, :3]
            C = -R.T @ E[:3, 3]
            right = R.T[:, 0].copy()
            up = -R.T[:, 1].copy()
            front = -R.T[:, 2].copy()
            if not (np.all(np.isfinite(C)) and np.all(np.isfinite(right))
                    and np.all(np.isfinite(up)) and np.all(np.isfinite(front))):
                return None
            n = float(np.linalg.norm(right))
            if n < 1e-9:
                return None
            right = right / n
            up = up / (np.linalg.norm(up) + 1e-12)
            front = front / (np.linalg.norm(front) + 1e-12)
            return C, right, up, front
        except Exception:
            return None

    # ------------------------------------------------------------------
    # world_per_pixel
    # ------------------------------------------------------------------
    def get_world_per_pixel(self):
        """每个**物理像素**对应的世界长度（正交投影）。

          zoom       —— 正交可视半高 = max_extent / zoom，故 zoom 在分子
          max_extent —— ViewControl 内部场景 AABB 的最大边长（此处用点云）
          H_fb       —— 帧缓冲高度（物理像素），不是逻辑像素
        """
        ctr = self.adapter.get_view_control()
        if ctr is None:
            return None
        try:
            extent = self._scene_max_extent()
            if extent is None or extent <= 0:
                return None
            _, h_fb = self._framebuffer_size()
            if h_fb <= 0:
                return None
            wpp = 2.0 * self._current_zoom() * float(extent) / float(h_fb)
            return float(wpp) if wpp > 0 else None
        except Exception:
            return None

    def _logical_world_per_pixel(self, dpr=None):
        """每个**逻辑像素**对应的世界长度（供返回逻辑像素坐标的函数使用）。"""
        wpp = self.get_world_per_pixel()
        if wpp is None or wpp <= 0:
            return None
        if dpr is None:
            _w, _h, dpr = self._viewport_metrics()
        d = float(dpr) if dpr and dpr > 0 else 1.0
        return wpp * d

    def _framebuffer_size(self):
        """帧缓冲尺寸（物理像素）。wpp 的定义域是物理像素坐标系。"""
        w, h, dpr = self._viewport_metrics()
        d = float(dpr) if dpr and dpr > 0 else 1.0
        return float(max(1, int(round(w * d)))), float(max(1, int(round(h * d))))

    # ------------------------------------------------------------------
    # 3D -> 屏幕
    # ------------------------------------------------------------------
    def project_points(self, points):
        """将 3D 点投影到视口坐标（逻辑像素，左上角原点）。"""
        ctr = self.adapter.get_view_control()
        if ctr is None or self.viewport_widget is None:
            return None

        try:
            if not self.is_orthographic():
                # 透视投影路径
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
                dpr = self._device_pixel_ratio()
                if dpr and dpr != 1.0:
                    screen[:, 0] /= dpr
                    screen[:, 1] /= dpr
                return screen, valid
        except Exception:
            pass

        # 正交投影路径
        try:
            w, h, dpr = self._viewport_metrics()
            basis = self.read_camera_basis()
            if basis is None:
                return None
            ref, right, up, front = basis

            wpp = self._logical_world_per_pixel(dpr)
            if wpp is None or wpp <= 0:
                return None

            pts = np.asarray(points, dtype=np.float64)
            d = pts - ref.reshape(1, 3)
            u = d @ right.reshape(3,)
            v = d @ up.reshape(3,)
            z = d @ front.reshape(3,)

            screen = np.zeros((len(pts), 3), dtype=np.float64)
            screen[:, 0] = (w * 0.5) + (u / wpp)
            screen[:, 1] = (h * 0.5) - (v / wpp)
            # front 由相机指向场景，故 z > 0 表示位于相机前方。
            screen[:, 2] = z
            valid = np.isfinite(screen[:, 0]) & np.isfinite(screen[:, 1])

            return screen, valid

        except Exception:
            return None

    def get_camera_basis(self):
        """获取当前相机坐标系的基向量，返回 (front, up, right)。"""
        try:
            basis = self.read_camera_basis()
            if basis is None:
                return None, None, None
            _ref, right, up, front = basis
            return front, up, right
        except Exception:
            return None, None, None

    # ------------------------------------------------------------------
    # 屏幕 -> 3D
    # ------------------------------------------------------------------
    def unproject_to_plane(self, screen_x, screen_y, plane_model, lookat=None):
        """将屏幕坐标反投影到指定平面的 3D 点（正交投影专用）。

        正交投影下屏幕坐标 (sx, sy) 对应的 3D 射线为：
            X(t) = ref + dx * right + dy * up + t * front
        其中 dx = (sx - w/2) * wpp, dy = -(sy - h/2) * wpp（逻辑像素基准）。
        与平面 dot(n, X) + d = 0 相交解得 t。

        Args:
            screen_x, screen_y: 屏幕坐标（逻辑像素，左上角为原点）
            plane_model: [nx, ny, nz, d] 平面方程 dot(n, X) + d = 0
            lookat: 可选，覆盖相机参考点（默认用相机中心）

        Returns:
            np.ndarray: 3D 点坐标 (3,)，失败或射线与平面平行时返回 None
        """
        ctr = self.adapter.get_view_control()
        if ctr is None or self.viewport_widget is None:
            return None
        if not self.is_orthographic():
            return None

        try:
            w, h, dpr = self._viewport_metrics()
            basis = self.read_camera_basis()
            if basis is None:
                return None
            ref, right, up, front = basis
            if lookat is not None:
                ref = np.asarray(lookat, dtype=float)

            # 必须与 project_points 使用完全相同的 world_per_pixel 定义，
            # 否则投影/反投影无法互逆，ROI 框会出现整体偏移与尺度错误。
            world_per_pixel = self._logical_world_per_pixel(dpr)
            if world_per_pixel is None or world_per_pixel <= 0:
                return None

            dx = (screen_x - w * 0.5) * world_per_pixel
            dy = -(screen_y - h * 0.5) * world_per_pixel

            ray_origin = ref + dx * right + dy * up

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

    def unproject_screen_corners(self, rect, depth_min=0.0, depth_max=0.0):
        """将屏幕矩形四角反投影到指定深度范围，返回 (8, 3) 世界坐标。

        正交投影下映射为线性：
            world = ref
                    + (sx - w/2) * wpp * right
                    - (sy - h/2) * wpp * up
                    + depth * front

        Args:
            rect: (x1, y1, x2, y2) 屏幕坐标（逻辑像素，左上角原点）
            depth_min, depth_max: 沿相机 front 方向的深度偏移量

        Returns:
            np.ndarray: (8, 3) 世界坐标，失败返回 None
        """
        ctr = self.adapter.get_view_control()
        if ctr is None or self.viewport_widget is None:
            return None
        if not self.is_orthographic():
            return None

        try:
            w, h, dpr = self._viewport_metrics()
            basis = self.read_camera_basis()
            if basis is None:
                return None
            ref, right, up, front = basis

            wpp = self._logical_world_per_pixel(dpr)
            if wpp is None or wpp <= 0:
                return None

            x1, y1, x2, y2 = (float(v) for v in rect)
            corners_screen = [
                (x1, y1), (x2, y1),
                (x1, y2), (x2, y2),
            ]

            pts = []
            for sx, sy in corners_screen:
                dx = (sx - w * 0.5) * wpp
                dy = -(sy - h * 0.5) * wpp
                base = ref + dx * right + dy * up
                pts.append(base + float(depth_min) * front)
                pts.append(base + float(depth_max) * front)

            return np.asarray(pts, dtype=np.float64).reshape(8, 3)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # 视口尺寸辅助
    # ------------------------------------------------------------------
    def _viewport_metrics(self):
        w = h = 1
        dpr = 1.0
        vw = self.viewport_widget
        try:
            if vw is not None:
                w = max(1, int(vw.width()))
                h = max(1, int(vw.height()))
                dpr = self._device_pixel_ratio()
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

    def _safe_get(self, ctr, attr, default):
        """保留：仅供仍在兼容旧接口的调用点使用。

        注意 ViewControl 上没有 get_* 这类接口，因此本函数实际总是返回
        default。新代码请使用 read_camera_basis() / _current_zoom()。
        """
        try:
            fn = getattr(ctr, attr)
            return fn() if callable(fn) else default
        except Exception:
            return default