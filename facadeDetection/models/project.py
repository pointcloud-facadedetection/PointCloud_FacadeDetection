from __future__ import annotations

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import relationship

from . import Base


class Project(Base):
    __tablename__ = 'projects'

    id = Column(Integer, primary_key=True)
    # Stable UUID used across DAL APIs
    uuid = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    org_unit = Column(String, nullable=True)
    address = Column(String, nullable=True)
    remarks = Column(String, nullable=True)
    building_floor = Column(String, nullable=True)
    root_dir = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)
    # backward compatibility field
    directory_path = Column(String, nullable=True)
    # Soft delete + audit (optional)
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime, nullable=True)

    # Relationships
    scenes = relationship("ResultScene", back_populates="project", cascade="all, delete-orphan")
    files = relationship("FileAsset", back_populates="project", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="project", cascade="all, delete-orphan")
