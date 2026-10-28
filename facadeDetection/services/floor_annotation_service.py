"""楼层标高到照片 2D 的统一投影标注。"""
from __future__ import annotations

import json
import cv2
import numpy as np


class FloorAnnotationService:
    @staticmethod
    def annotate(image_bgr, building_info, uv_bounds, uv_to_photo,
                 plane_frame, grid_step=0.0):
        """在已完成 UV→照片 H 的图像上绘制楼层线和标签。

        plane_frame=(origin,u_axis,v_axis,plane_model)，楼层标高采用全局 Z。
        只绘制与当前热力图 UV 范围相交的楼层，避免重复建立一套配准链路。
        """
        img = np.asarray(image_bgr).copy()
        if not isinstance(building_info, dict):
            return img
        origin, u_axis, v_axis, plane_model = [np.asarray(x, dtype=float).reshape(-1)
                                                for x in plane_frame]
        H = np.asarray(uv_to_photo, dtype=float).reshape(3, 3)
        u0, v0, u1, v1 = map(float, uv_bounds)
        floors = FloorAnnotationService.floor_rows(building_info)
        from utils.photo_matching import plane_uv_to_photo
        for row in floors:
            z = float(row["elevation_m"])
            # plane point = origin + u*u_axis + v*v_axis；固定 z 后的 v(u) 直线
            if abs(v_axis[2]) < 1e-9:
                continue
            v_at_u0 = (z - origin[2] - u0 * u_axis[2]) / v_axis[2]
            v_at_u1 = (z - origin[2] - u1 * u_axis[2]) / v_axis[2]
            if max(v_at_u0, v_at_u1) < v0 or min(v_at_u0, v_at_u1) > v1:
                continue
            pts = plane_uv_to_photo(np.array([[u0, v_at_u0], [u1, v_at_u1]]), H)
            p0, p1 = [tuple(np.rint(p).astype(int)) for p in pts]
            cv2.line(img, p0, p1, (40, 220, 255), 2, cv2.LINE_AA)
            text = f"{row['floor_no']}F  ±{z:.3f}m" if abs(z) < 1e-9 else f"{row['floor_no']}F  {z:.3f}m"
            cv2.putText(img, text, (p0[0] + 5, p0[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 40, 40), 2,
                        cv2.LINE_AA)
            cv2.putText(img, text, (p0[0] + 5, p0[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)
        return img

    @staticmethod
    def floor_rows(building_info):
        from services.building_info_service import floor_elevations
        return floor_elevations(building_info)

    @staticmethod
    def from_project(project):
        from services.building_info_service import parse_building_info
        params = getattr(project, "inspection_params_json", None)
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except (TypeError, ValueError):
                params = {}
        return parse_building_info(params or {})