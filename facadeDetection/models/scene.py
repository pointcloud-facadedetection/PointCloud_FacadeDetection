from __future__ import annotations

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship

from . import Base
from .enums import SceneStatus


class ResultScene(Base):
    __tablename__ = "result_scenes"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String, nullable=False)
    status = Column(String, default=SceneStatus.active.value, nullable=False)
    notes = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    activated_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=False, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime, nullable=True)

    project = relationship("Project", back_populates="scenes")
    files = relationship("FileAsset", back_populates="scene")
    facades = relationship("Facade", back_populates="scene", cascade="all, delete-orphan")
