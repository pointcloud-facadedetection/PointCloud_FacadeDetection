from __future__ import annotations

from pathlib import Path

from services.dal.file_repo import FileRepo
from utils.logging_utils import log_event
from models import Facade, QualityMetric
from db.connection import project_session
from sqlalchemy import select


class ProjectRestoreService:
    """Coordinates project reopening without changing any algorithm pipeline."""

    def __init__(self, viewport, render_service, file_service):
        self.viewport = viewport
        self.render_service = render_service
        self.file_service = file_service

    def restore(self, project_uuid: str) -> dict:
        assets = FileRepo.list_assets(project_uuid, pointcloud_only=True)
        loaded, skipped = [], []
        for asset in assets:
            valid, reason = FileRepo.validate_asset(asset)
            log_event(project_uuid, "asset.validate", path=asset.path, result=reason)
            if not valid:
                skipped.append({"path": asset.path, "reason": reason})
                continue
            try:
                self.file_service.upload_files(project_uuid, asset.path, copy_into_project=False)
                loaded.append(asset.path)
                log_event(project_uuid, "asset.loaded", path=asset.path)
            except Exception as exc:
                skipped.append({"path": asset.path, "reason": str(exc)})
                log_event(project_uuid, "asset.load_failed", path=asset.path, error=str(exc))
        facades = self.load_facades(project_uuid)
        return {"loaded": loaded, "skipped": skipped, "facades": facades}

    @staticmethod
    def load_facades(project_uuid: str) -> list[dict]:
        with project_session(project_uuid) as session:
            rows = session.execute(select(Facade).where(Facade.is_deleted == 0).order_by(Facade.display_no, Facade.id)).scalars().all()
            result = []
            for row in rows:
                metrics = session.execute(select(QualityMetric).where(QualityMetric.facade_id == row.id)).scalars().all()
                geometry = row.plane_json or {}
                display_no = int(row.display_no or 0)
                if display_no <= 0:
                    try:
                        display_no = int(str(row.label).rsplit(' ', 1)[-1])
                    except (ValueError, IndexError):
                        display_no = len(result) + 1
                point_count = int(row.point_count or geometry.get('point_count') or
                                  len(geometry.get('proxy_indices') or geometry.get('inlier_indices') or []))
                raw_point_count = int(row.raw_point_count or geometry.get('raw_point_count') or point_count)
                item = {"id": row.id, "facade_db_id": row.id, "display_no": display_no,
                        "point_count": point_count, "raw_point_count": raw_point_count,
                        "type": row.label, "type_label": row.label,
                        "area": row.area or 0.0, "plane": row.plane_json,
                         "bbox": row.bbox_json, "quality_status": row.quality_status,
                         "quality_report": row.quality_report_json,
                         "color": row.color_json, "dataset_revision": row.dataset_revision,
                         "quality_metrics": [
                            {"name": m.metric_name, "value": m.value, "unit": m.unit, "pass": m.pass_flag}
                            for m in metrics],
                        **{key: geometry[key] for key in (
                            'plane_model', 'normal', 'center', 'inlier_indices',
                         'proxy_indices', 'measurement_indices', 'voxel_ids',
                         'review_status',
                            'cloud_name', '__index_space') if key in geometry}}
                result.append(item)
            return result