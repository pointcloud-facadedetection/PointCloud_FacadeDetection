from __future__ import annotations

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, Index, Boolean
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import relationship

from . import Base


class Facade(Base):
    __tablename__ = "facades"
    __table_args__ = (
        Index("ix_facades_scene", "scene_id"),
    )

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), index=True, nullable=False)
    scene_id = Column(Integer, ForeignKey("result_scenes.id", ondelete="CASCADE"), index=True, nullable=False)
    label = Column(String, nullable=False)
    plane_json = Column(JSON, nullable=True)  # e.g., normal + d, or 4x4
    bbox_json = Column(JSON, nullable=True)
    area = Column(Float, nullable=True)
    orientation = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)
    is_deleted = Column(Integer, default=0, nullable=False)
    deleted_at = Column(DateTime, nullable=True)
    # Runtime geometry stays in the point-cloud index; these fields contain only
    # the durable decision metadata and the serializable quality report.
    quality_status = Column(String, default="pending", nullable=False)
    quality_report_json = Column(JSON, nullable=True)
    color_json = Column(JSON, nullable=True)
    dataset_revision = Column(String, nullable=True)
    quality_completed_at = Column(DateTime, nullable=True)

    scene = relationship("ResultScene", back_populates="facades")
    metrics = relationship("QualityMetric", back_populates="facade", cascade="all, delete-orphan")
    heatmap = relationship("Heatmap", back_populates="facade", uselist=False, cascade="all, delete-orphan")


class QualityMetric(Base):
    __tablename__ = "quality_metrics"

    id = Column(Integer, primary_key=True)
    facade_id = Column(Integer, ForeignKey("facades.id", ondelete="CASCADE"), index=True, nullable=False)
    metric_name = Column(String, nullable=False)  # flatness, verticality, etc.
    value = Column(Float, nullable=False)
    unit = Column(String, nullable=True)
    pass_flag = Column(Integer, nullable=True)  # 1/0/None
    thresholds_json = Column(JSON, nullable=True)

    facade = relationship("Facade", back_populates="metrics")


class Heatmap(Base):
    __tablename__ = "heatmaps"

    id = Column(Integer, primary_key=True)
    facade_id = Column(Integer, ForeignKey("facades.id", ondelete="CASCADE"), index=True, nullable=False)
    file_id = Column(Integer, ForeignKey("file_assets.id", ondelete="SET NULL"), nullable=True)
    vmin = Column(Float, nullable=True)
    vmax = Column(Float, nullable=True)
    cmap = Column(String, nullable=True)

    facade = relationship("Facade", back_populates="heatmap")
