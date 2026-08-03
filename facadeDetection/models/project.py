from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime

from . import Base


class Project(Base):
    __tablename__ = 'projects'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    directory_path = Column(String)
