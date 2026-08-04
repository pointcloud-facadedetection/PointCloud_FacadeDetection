from __future__ import annotations

try:
    from enum import StrEnum  # Python 3.11+
except ImportError:  # pragma: no cover
    from enum import Enum

    class StrEnum(str, Enum):
        """Minimal backport of StrEnum for Python < 3.11."""
        pass


class FileKind(StrEnum):
    raw_pointcloud = "raw_pointcloud"
    raw_image = "raw_image"
    denoised_pc = "denoised_pc"
    registered_pc = "registered_pc"
    heatmap_img = "heatmap_img"
    texture_tmp = "texture_tmp"
    pdf_report = "pdf_report"
    other = "other"


class PersistPolicy(StrEnum):
    PERSIST = "PERSIST"   # survives project lifetime
    CACHE = "CACHE"       # only active scene retained


class SceneStatus(StrEnum):
    active = "active"
    inactive = "inactive"
    archived = "archived"


class RunType(StrEnum):
    denoise = "denoise"
    registration = "registration"
    facade_detection = "facade_detection"
    quality = "quality"
    segmentation = "segmentation"
    texture_map = "texture_map"


class RunStatus(StrEnum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"
