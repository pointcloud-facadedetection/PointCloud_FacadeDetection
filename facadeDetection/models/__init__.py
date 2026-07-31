from sqlalchemy.orm import declarative_base

Base = declarative_base()

# Re-export models for convenience
from .project import Project  # noqa: E402,F401
from .scene import ResultScene  # noqa: E402,F401
from .file_asset import FileAsset  # noqa: E402,F401
from .processing import ProcessingRun  # noqa: E402,F401
from .facade import Facade, QualityMetric, Heatmap  # noqa: E402,F401
from .report import Report  # noqa: E402,F401