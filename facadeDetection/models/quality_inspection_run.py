from __future__ import annotations

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, Index
from sqlalchemy.dialects.sqlite import JSON
from . import Base


class QualityInspectionRun(Base):
    """Immutable-ish archive of one facade quality evaluation configuration."""
    __tablename__ = "quality_inspection_runs"
    __table_args__ = (
        Index("ix_quality_runs_project_station", "project_id", "station_id"),
        Index("ix_quality_runs_facade", "facade_id"),
        Index("ix_quality_runs_key", "project_id", "station_id", "facade_key",
              "parameter_fingerprint"),
    )

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    station_id = Column(Integer, ForeignKey("pointcloud_stations.id", ondelete="SET NULL"), nullable=True)
    facade_id = Column(Integer, ForeignKey("facades.id", ondelete="SET NULL"), nullable=True)
    facade_key = Column(String, nullable=False)
    facade_display_no = Column(Integer, nullable=True)
    cloud_name = Column(String, nullable=True)
    dataset_id = Column(String, nullable=True)
    dataset_fingerprint = Column(String, nullable=True)
    dataset_revision = Column(String, nullable=True)
    standard_id = Column(String, nullable=True)
    standard_name = Column(String, nullable=True)
    standard_version = Column(String, nullable=True)
    interval_size_m = Column(Float, nullable=True)
    parameter_fingerprint = Column(String, nullable=False)
    profile_snapshot_json = Column(JSON, nullable=True)
    quality_status = Column(String, nullable=False, default="complete")
    quality_report_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)