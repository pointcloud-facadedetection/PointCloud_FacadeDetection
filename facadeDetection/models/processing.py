from __future__ import annotations

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Index
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import relationship

from . import Base
from .enums import RunStatus, RunType


class ProcessingRun(Base):
    __tablename__ = "processing_runs"
    __table_args__ = (
        Index("ix_runs_scene_status", "scene_id", "status"),
    )

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), index=True, nullable=False)
    scene_id = Column(Integer, ForeignKey("result_scenes.id", ondelete="CASCADE"), index=True, nullable=False)
    run_type = Column(String, nullable=False, default=RunType.denoise.value)
    status = Column(String, nullable=False, default=RunStatus.pending.value)
    params_json = Column(JSON, nullable=True)
    started_at = Column(DateTime, default=datetime.now, nullable=False)
    finished_at = Column(DateTime, nullable=True)
    log_path = Column(String, nullable=True)  # optional text log on disk

    scene = relationship("ResultScene", back_populates="runs")
