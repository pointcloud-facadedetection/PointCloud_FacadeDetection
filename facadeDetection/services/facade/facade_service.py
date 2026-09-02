"""FacadeService 兼容门面。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from services.facade.facade_index_service import FacadeIndexService
from services.facade.facade_detection_service import FacadeDetectionService
from services.facade.facade_quality_service import FacadeQualityService


class FacadeService:
    """外立面检测与质量评估统一门面。"""

    def __init__(self, viewport, db=None, render_service=None,
                 pointcloud_service=None, index_service: Optional[FacadeIndexService] = None):
        self._viewport = viewport
        self._db = db
        self._render = render_service
        self._pointcloud_service = pointcloud_service
        self._index_service = index_service or FacadeIndexService(pointcloud_service)
        self._detection_service = FacadeDetectionService(viewport, pointcloud_service, self._index_service)
        self._quality_service = FacadeQualityService(pointcloud_service, self._index_service)

    def set_pointcloud_service(self, pointcloud_service):
        self._pointcloud_service = pointcloud_service
        self._index_service.set_pointcloud_service(pointcloud_service)
        self._detection_service._pointcloud_service = pointcloud_service
        self._quality_service._pointcloud_service = pointcloud_service

    # ==================== 检测接口 ====================

    def detect(self, cloud_name: str, project_uuid: Optional[str] = None) -> list[dict]:
        """全局立面检测。"""
        return self._detection_service.detect(cloud_name, project_uuid)

    def detect_on_roi(self, cloud_name: str,
                      roi_indices: list[int] | None = None,
                      roi_bounds: tuple | None = None,
                      roi_box: dict | None = None,
                      project_uuid: Optional[str] = None,
                      min_facade_area: float | None = None,
                      roi_scope: str = 'seed') -> list[dict]:
        """ROI 立面检测。"""
        return self._detection_service.detect_on_roi(
            cloud_name, roi_indices, roi_bounds, roi_box,
            project_uuid, min_facade_area, roi_scope)

    # ==================== 质量评估接口 ====================

    def compute_quality(self, cloud_name: str, facade: dict,
                        grid_size: float | None = None,
                        flatness_limit: float = 0.004,
                        verticality_limit_mm: float = 4.0,
                        ruler_size: float | None = None,
                        ruler_step: float | None = None,
                        profile=None, results_dir=None) -> Optional[dict]:
        """质量评估计算。"""
        return self._quality_service.compute_quality(
            cloud_name, facade,
            grid_size=grid_size,
            flatness_limit=flatness_limit,
            verticality_limit_mm=verticality_limit_mm,
            ruler_size=ruler_size,
            ruler_step=ruler_step,
            profile=profile,
            results_dir=results_dir)

    # ==================== 索引映射接口 ====================

    def map_roi_to_raw(self, cloud_name: str, roi_proxy_indices: np.ndarray) -> np.ndarray:
        """将代理空间的 ROI 决策结果映射回原始空间。"""
        return self._index_service.map_roi_to_raw(cloud_name, roi_proxy_indices)

    def map_raw_to_proxy(self, cloud_name: str, raw_indices: np.ndarray) -> np.ndarray:
        """将原始空间索引映射到代理空间。"""
        return self._index_service.map_raw_to_proxy(cloud_name, raw_indices)

    def get_raw_count_for_facade(self, cloud_name: str, facade: dict) -> int:
        """获取立面对应的原始点数量（不分配内存）。"""
        return self._index_service.get_raw_count_for_facade(cloud_name, facade)
