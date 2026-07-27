from datetime import datetime 
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, BigInteger 
from sqlalchemy.dialects.sqlite import JSON 
from . import Base 
 
class FileRecord(Base): 
    __tablename__ = 'files' 
    id = Column(Integer, primary_key=True) 
    project_id = Column(Integer, ForeignKey('projects.id'), index=True, nullable=False) 
    file_type = Column(String, nullable=False)  # pointcloud/image/other 
    path = Column(String, nullable=False) 
    original_name = Column(String, nullable=True) 
    size_bytes = Column(BigInteger, nullable=True) 
    sha256 = Column(String, nullable=True, index=True) 
    # 'metadata' is a reserved attribute in SQLAlchemy Declarative; use 'meta_json' attribute and map to column name 'metadata'
    meta_json = Column('metadata', JSON, nullable=True) 
    is_deleted = Column(Boolean, default=False, nullable=False) 
    created_at = Column(DateTime, default=datetime.now, nullable=False)
