from __future__ import annotations

from pathlib import Path
from typing import Optional

from db.connection import index_session
from models import FileAsset, Project, Report
from models.enums import FileKind, PersistPolicy


class ReportRepo:
    @staticmethod
    def register_pdf(project_id: int, pdf_path: str, title: str | None = None) -> Optional[Report]:
        path = Path(pdf_path).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        with index_session() as s:
            proj = s.get(Project, project_id)
            if not proj:
                return None
            asset = FileAsset(
                project_id=project_id,
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
            rep = Report(project_id=project_id, file_id=asset.id, title=title)
            s.add(rep)
            s.flush()
            return rep
