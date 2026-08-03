from sqlalchemy.orm import declarative_base

Base = declarative_base()

from .project import Project
from .pointcloud import PointCloud
from .analysis import Analysis
from .registration import Registration
from .files import FileRecord
