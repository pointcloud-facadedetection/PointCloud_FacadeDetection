from __future__ import annotations

from pathlib import Path


def guess_file_kind(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".pts"}:
        return "raw_pointcloud"
    if suffix in {".png", ".jpg", ".jpeg", ".bmp"}:
        return "raw_image"
    if suffix in {".pdf"}:
        return "pdf_report"
    return "other"
