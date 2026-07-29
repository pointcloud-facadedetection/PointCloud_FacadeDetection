from __future__ import annotations

import os
from pathlib import Path


class Storage:
    """
    桌面应用的集中存储路径。

    规则：
    - 大型二进制文件只存储在磁盘上；数据库只存储元数据的绝对路径
    - 每个项目的文件夹布局在 PROJECTS_ROOT 下
    """

    BASE_DIR = Path("D:/ElevationDetect")
    DATA_DIR = BASE_DIR / "data"
    # Global lightweight index database (projects list)
    INDEX_DB_FILE = DATA_DIR / "index.db"
    PROJECTS_ROOT = BASE_DIR / "projects"

    # Subfolders inside a project root
    RAW_DIRNAME = "raw"
    CACHE_DIRNAME = "cache"
    RESULTS_DIRNAME = "results"
    REPORTS_DIRNAME = "reports"

    @classmethod
    def ensure_base_dirs(cls) -> None:
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)

    @classmethod
    def project_root(cls, project_uuid: str) -> Path:
        root = cls.PROJECTS_ROOT / project_uuid
        # Do not create automatically here; caller decides when project is created
        return root

    @classmethod
    def ensure_project_dirs(cls, project_uuid: str) -> dict[str, Path]:
        root = cls.project_root(project_uuid)
        (root / cls.RAW_DIRNAME).mkdir(parents=True, exist_ok=True)
        (root / cls.CACHE_DIRNAME).mkdir(parents=True, exist_ok=True)
        (root / cls.RESULTS_DIRNAME).mkdir(parents=True, exist_ok=True)
        (root / cls.REPORTS_DIRNAME).mkdir(parents=True, exist_ok=True)
        return {
            "root": root,
            "raw": root / cls.RAW_DIRNAME,
            "cache": root / cls.CACHE_DIRNAME,
            "results": root / cls.RESULTS_DIRNAME,
            "reports": root / cls.REPORTS_DIRNAME,
        }

    @classmethod
    def project_db_path(cls, project_uuid: str) -> Path:
        return cls.project_root(project_uuid) / "project.db"
