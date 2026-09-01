from __future__ import annotations

from sqlalchemy import select
from db.connection import project_session
from models import Project, QualityInspectionRun


class QualityRunRepo:
    @staticmethod
    def upsert(project_uuid, *, station_id, facade_id, facade_key, facade_display_no,
               cloud_name, dataset_id, dataset_fingerprint, dataset_revision,
               profile_snapshot, standard_id, standard_name, standard_version,
               interval_size_m, parameter_fingerprint, quality):
        with project_session(project_uuid) as s:
            project = s.execute(select(Project).where(Project.uuid == str(project_uuid))).scalar_one()
            stmt = select(QualityInspectionRun).where(
                QualityInspectionRun.project_id == project.id,
                QualityInspectionRun.station_id == station_id,
                QualityInspectionRun.facade_key == str(facade_key),
                QualityInspectionRun.parameter_fingerprint == parameter_fingerprint)
            row = s.execute(stmt).scalar_one_or_none()
            values = dict(project_id=project.id, station_id=station_id, facade_id=facade_id,
                          facade_key=str(facade_key), facade_display_no=facade_display_no,
                          cloud_name=cloud_name, dataset_id=dataset_id,
                          dataset_fingerprint=dataset_fingerprint, dataset_revision=dataset_revision,
                          profile_snapshot_json=profile_snapshot, standard_id=standard_id,
                          standard_name=standard_name, standard_version=standard_version,
                          interval_size_m=interval_size_m, parameter_fingerprint=parameter_fingerprint,
                          quality_status='complete', quality_report_json=quality)
            if row is None:
                row = QualityInspectionRun(**values)
                s.add(row)
            else:
                for key, value in values.items():
                    if key != 'project_id':
                        setattr(row, key, value)
            s.flush()
            return row.id

    @staticmethod
    def list_runs(project_uuid, station_id=None):
        with project_session(project_uuid) as s:
            stmt = select(QualityInspectionRun).join(Project).where(Project.uuid == str(project_uuid))
            if station_id is not None:
                stmt = stmt.where(QualityInspectionRun.station_id == int(station_id))
            return s.execute(stmt.order_by(QualityInspectionRun.created_at, QualityInspectionRun.id)).scalars().all()