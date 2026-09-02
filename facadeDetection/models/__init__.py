from sqlalchemy.orm import declarative_base

Base = declarative_base()

# Re-export models for convenience
from .project import Project  # noqa: E402,F401
from .scene import ResultScene  # noqa: E402,F401
from .file_asset import FileAsset  # noqa: E402,F401
from .facade import Facade, QualityMetric  # noqa: E402,F401
from .report import Report  # noqa: E402,F401
from .pointcloud_station import PointCloudStation  # noqa: E402,F401
from .pointcloud_view_state import PointCloudViewState  # noqa: E402,F401