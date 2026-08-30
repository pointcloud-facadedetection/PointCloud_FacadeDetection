from __future__ import annotations

from pathlib import Path
from typing import Optional

from sqlalchemy import select
from db.connection import project_session
from models import FileAsset, Project, Report
from models.enums import FileKind, PersistPolicy


class ReportRepo:
    @staticmethod
    def register_pdf(project_id, pdf_path: str, title: str | None = None) -> Optional[Report]:
        """Register a report in the selected project's database.

        ``project_id`` is the project UUID used by the current architecture.
        Keep accepting an integer for legacy callers, but never query the
        project database through the global index session.
        """
        path = Path(pdf_path).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        with project_session(str(project_id)) as s:
            proj = s.execute(
                select(Project).where(Project.uuid == str(project_id))
            ).scalar_one_or_none()
            if not proj:
                return None
            asset = FileAsset(
                project_id=proj.id,
                scene_id=None,
                kind=FileKind.pdf_report.value,
                persist_policy=PersistPolicy.PERSIST.value,
                path=str(path),
                original_name=path.name,
                ext=path.suffix.lower(),
                size_bytes=path.stat().st_size,
            )
            s.add(asset)
            s.flush()
            rep = Report(project_id=proj.id, file_id=asset.id, title=title)
            s.add(rep)
            s.flush()
            return rep
