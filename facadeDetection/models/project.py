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
    # Soft delete + audit (optional)
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime, nullable=True)

    # ── PDF 报告元信息 ──
    construction_unit = Column(String, nullable=True)           # 建设单位
    construction_unit_executor = Column(String, nullable=True)  # 施工单位
    inspection_unit = Column(String, nullable=True)             # 检测单位
    supervision_unit = Column(String, nullable=True)            # 监理单位
    client_unit = Column(String, nullable=True)                 # 委托单位
    report_no = Column(String, nullable=True)                   # 报告编号
    inspection_date = Column(DateTime, nullable=True)           # 检测/测量日期
    report_date = Column(DateTime, nullable=True)               # 报告日期

    # ── 质量检测参数快照（创建项目时录入，作为该项目的默认参数） ──
    inspection_params_json = Column(String, nullable=True)      # JSON 字符串存储

    # Relationships
    scenes = relationship("ResultScene", back_populates="project", cascade="all, delete-orphan")
    files = relationship("FileAsset", back_populates="project", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="project", cascade="all, delete-orphan")