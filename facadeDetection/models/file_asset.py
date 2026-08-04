from __future__ import annotations

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, BigInteger, Index
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import relationship

from . import Base
from .enums import FileKind, PersistPolicy


class FileAsset(Base):
    __tablename__ = "file_assets"
    __table_args__ = (
        Index("ix_file_assets_gc", "project_id", "scene_id", "persist_policy", "is_deleted"),
        Index("ix_file_assets_project_scene", "project_id", "scene_id"),
    )

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    scene_id = Column(Integer, ForeignKey("result_scenes.id", ondelete="SET NULL"), nullable=True, index=True)

    # Kind can be derived from extension; keep nullable and fill on write for convenience
    kind = Column(String, nullable=True)
    persist_policy = Column(String, default=PersistPolicy.PERSIST.value, nullable=False)

    path = Column(String, nullable=False)  # absolute path on disk
    original_name = Column(String, nullable=True)
    ext = Column(String, nullable=True, index=True)
    size_bytes = Column(BigInteger, nullable=True)
    sha256 = Column(String, nullable=True, index=True)
    meta_json = Column(JSON, nullable=True)
    is_deleted = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)

    project = relationship("Project", back_populates="files")
    scene = relationship("ResultScene", back_populates="files")
