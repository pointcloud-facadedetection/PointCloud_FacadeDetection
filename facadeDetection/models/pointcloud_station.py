from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, JSON, Float, UniqueConstraint
from sqlalchemy.orm import relationship
from . import Base

class PointCloudStation(Base):
    __tablename__ = 'pointcloud_stations'
    __table_args__ = (UniqueConstraint('project_id', 'station_key', name='uq_station_project_key'),)
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('projects.id', ondelete='CASCADE'), nullable=False, index=True)
    file_asset_id = Column(Integer, ForeignKey('file_assets.id', ondelete='SET NULL'), nullable=True, index=True)
    station_key = Column(String, nullable=False, index=True)
    display_name = Column(String, nullable=False)
    source_path = Column(String, nullable=False)
    display_order = Column(Integer, default=0, nullable=False)
    is_selected = Column(Boolean, default=False, nullable=False)
    is_deleted = Column(Boolean, default=False, nullable=False)
    registration_status = Column(String, default='raw', nullable=False)
    transform_json = Column(JSON, nullable=True)
    fitness = Column(Float, nullable=True)
    inlier_rmse = Column(Float, nullable=True)
    registered_path = Column(String, nullable=True)
    # JSON index state allows denoised proxy reconstruction without a derived
    # point-cloud file in the project results directory.
    denoise_state_json = Column(JSON, nullable=True)
    last_error = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)
    project = relationship('Project')
    asset = relationship('FileAsset')
