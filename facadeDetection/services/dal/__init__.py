"""Data access layer (repositories) for the desktop app.

Repositories provide high-level CRUD focused on business use cases and
are safe to call from PySide slots. All functions use short-lived sessions
through db.crud.get_session().
"""

from .project_repo import ProjectRepo  # noqa: F401
from .file_repo import FileRepo  # noqa: F401
from .results_repo import ResultsRepo  # noqa: F401
from .report_repo import ReportRepo  # noqa: F401
