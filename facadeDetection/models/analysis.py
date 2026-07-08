from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Text
from datetime import datetime

from . import Base


class Analysis(Base):
    __tablename__ = 'analysis_results'

    id = Column(Integer, primary_key=True)
    pointcloud_id = Column(Integer, ForeignKey('pointclouds.id'))
    result_type = Column(String)
    result_data = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
