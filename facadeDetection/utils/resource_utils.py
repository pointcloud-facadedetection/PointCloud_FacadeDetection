from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def resource_path(*parts: str) -> Path:
    """Return a path to bundled resources that works in source and frozen mode."""
    return project_root().joinpath(*parts)
