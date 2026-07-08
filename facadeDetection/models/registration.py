from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey, Text
from datetime import datetime

from . import Base


class Registration(Base):
    __tablename__ = 'registrations'

    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey('pointclouds.id'))
    target_id = Column(Integer, ForeignKey('pointclouds.id'))
    transform_matrix = Column(Text)
    rmse = Column(Float)
    overlap_ratio = Column(Float)
    created_at = Column(DateTime, default=datetime.now)
