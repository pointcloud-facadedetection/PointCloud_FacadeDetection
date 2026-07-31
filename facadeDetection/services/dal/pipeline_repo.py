from __future__ import annotations

from datetime import datetime
from typing import Optional

from db.connection import project_session
from models import ProcessingRun, Project
from models.enums import RunStatus, RunType


class PipelineRepo:
    @staticmethod
    def start_run(project_uuid: str, scene_id: int, run_type: RunType, params: dict | None = None) -> ProcessingRun:
        with project_session(project_uuid) as s:
            run = ProcessingRun(
                project_id=None,
                scene_id=scene_id,
                run_type=run_type.value,
                status=RunStatus.running.value,
                params_json=params or {},
                started_at=datetime.now(),
            )
            s.add(run)
            s.flush()
            return run

    @staticmethod
    def finish_run(project_uuid: str, run_id: int, success: bool = True, log_path: str | None = None) -> Optional[ProcessingRun]:
        with project_session(project_uuid) as s:
            run = s.get(ProcessingRun, run_id)
            if not run:
                return None
            run.status = RunStatus.done.value if success else RunStatus.failed.value
            run.finished_at = datetime.now()
            if log_path:
                run.log_path = log_path
            s.flush()
            return run
