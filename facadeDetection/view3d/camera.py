import numpy as np


class CameraController:
    def __init__(self, adapter, viewport_widget=None):
        self.adapter = adapter
        self.viewport_widget = viewport_widget
        self._state = None

    def auto_range(self):
        try:
            self.adapter.reset_view_point()
        except Exception:
            pass

    def reset_view(self):
        self.auto_range()

    def get_state(self):
        ctr = self.adapter.get_view_control()
        if ctr is None:
            return self._state
        try:
            self._state = ctr.convert_to_pinhole_camera_parameters()
        except Exception:
            pass
        return self._state

    def set_state(self, state):
        self._state = state
        ctr = self.adapter.get_view_control()
        if ctr is None or state is None:
            return
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

    def project_points(self, points):
        ctr = self.adapter.get_view_control()
        if ctr is None or self.viewport_widget is None:
            return None

        try:
            params = ctr.convert_to_pinhole_camera_parameters()
        except Exception:
            return None

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

        # 对齐 Qt 事件坐标（DPI 缩放差异）
        try:
            dpr = 1.0
            if hasattr(self.viewport_widget, "devicePixelRatioF"):
                dpr = float(self.viewport_widget.devicePixelRatioF())
            elif hasattr(self.viewport_widget, "devicePixelRatio"):
                dpr = float(self.viewport_widget.devicePixelRatio())
            elif hasattr(self.viewport_widget, "windowHandle") and self.viewport_widget.windowHandle() is not None:
                wh = self.viewport_widget.windowHandle()
                if hasattr(wh, "devicePixelRatio"):
                    dpr = float(wh.devicePixelRatio())
            if dpr and dpr != 1.0:
                screen[:, 0] /= dpr
                screen[:, 1] /= dpr
        except Exception:
            pass

        return screen, valid