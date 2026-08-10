from __future__ import annotations

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from . import Base


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), index=True, nullable=False)
    file_id = Column(Integer, ForeignKey("file_assets.id", ondelete="SET NULL"), nullable=True)
    version = Column(Integer, default=1, nullable=False)
    title = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)

    project = relationship("Project", back_populates="reports")
