from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from datetime import datetime

from . import Base


class PointCloud(Base):
    __tablename__ = 'pointclouds'

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('projects.id'))
    filename = Column(String, nullable=False)
    point_count = Column(Integer)
    voxel_size = Column(Float)
    transform_matrix = Column(Text)
    cache_path = Column(String)
    created_at = Column(DateTime, default=datetime.now)
