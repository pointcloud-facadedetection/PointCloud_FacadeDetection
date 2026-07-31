from __future__ import annotations

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import relationship

from . import Base


class Project(Base):
    __tablename__ = 'projects'

    id = Column(Integer, primary_key=True)
    uuid = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    org_unit = Column(String, nullable=True)
    address = Column(String, nullable=True)
    remarks = Column(String, nullable=True)
    root_dir = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)

    # Soft delete + audit
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime, nullable=True)

    # Relationships
    scenes = relationship("ResultScene", back_populates="project", cascade="all, delete-orphan")
    files = relationship("FileAsset", back_populates="project", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="project", cascade="all, delete-orphan")
