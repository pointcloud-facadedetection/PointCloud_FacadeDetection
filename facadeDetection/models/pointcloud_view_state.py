from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from . import Base

class PointCloudViewState(Base):
    __tablename__ = 'pointcloud_view_states'
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('projects.id', ondelete='CASCADE'), nullable=False, unique=True)
    display_mode = Column(String, default='single', nullable=False)
    active_station_id = Column(Integer, nullable=True)
    selected_station_ids = Column(JSON, default=list, nullable=False)
    registration_id = Column(Integer, nullable=True)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)
