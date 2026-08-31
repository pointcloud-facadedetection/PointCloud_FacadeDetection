from __future__ import annotations

from dataclasses import dataclass
from threading import Event


class TaskCancelledError(RuntimeError):
    """Raised when a superseded task reaches a cancellation checkpoint."""


@dataclass(frozen=True)
class TaskContext:
    """Immutable identity carried by every asynchronous operation."""

    kind: str
    project_uuid: str | None
    project_generation: int
    dataset_id: str | None = None
    dataset_revision: str | None = None
    operation_token: int = 0
    cancel_event: Event | None = None

    def check_cancelled(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise TaskCancelledError(f"{self.kind} task cancelled")
