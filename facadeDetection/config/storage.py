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

    BASE_DIR = Path(__file__).resolve().parents[1]
    REPO_ROOT = BASE_DIR.parent
    if os.name == "nt":
        LEGACY_DATA_DIR = Path(os.getenv("LOCALAPPDATA", str(BASE_DIR))) / "PointCloudFacadeDetection"
    else:
        LEGACY_DATA_DIR = Path.home() / ".local" / "share" / "PointCloudFacadeDetection"

    _env_root = os.getenv("FACD_DATA_DIR")
    if _env_root:
        DATA_DIR = Path(_env_root)
    else:
        repo_data = REPO_ROOT / "data"
        repo_index = repo_data / "index.db"
        legacy_index = LEGACY_DATA_DIR / "index.db"
        # 仓库 data/ 可能只有立面缓存；历史项目列表在 AppData 的 index.db。
        if legacy_index.exists() and (
            not repo_index.exists() or repo_index.stat().st_size <= legacy_index.stat().st_size
        ):
            DATA_DIR = LEGACY_DATA_DIR
        elif repo_index.exists():
            DATA_DIR = repo_data
        elif repo_data.exists() and not legacy_index.exists():
            DATA_DIR = repo_data
        else:
            DATA_DIR = LEGACY_DATA_DIR
    # 全局轻量级索引数据库（项目列表）
    INDEX_DB_FILE = DATA_DIR / "index.db"
    PROJECTS_ROOT = DATA_DIR / "projects"
    LEGACY_PROJECTS_ROOT = LEGACY_DATA_DIR / "projects"
    PCFD_DIRNAME = "pcfd"

    # 项目根目录下的子文件夹
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
        return cls.resolve_project_root(project_uuid)

    @classmethod
    def ensure_project_dirs(cls, project_uuid: str) -> dict[str, Path]:
        """确保子目录位于已解析的项目根目录，并返回路径字典。"""
        root = cls.project_root(project_uuid)
        (root / cls.RAW_DIRNAME).mkdir(parents=True, exist_ok=True)
        (root / cls.CACHE_DIRNAME).mkdir(parents=True, exist_ok=True)
        (root / cls.RESULTS_DIRNAME).mkdir(parents=True, exist_ok=True)
        (root / cls.REPORTS_DIRNAME).mkdir(parents=True, exist_ok=True)
        (root / cls.PCFD_DIRNAME).mkdir(parents=True, exist_ok=True)
        return {
            "root": root,
            "raw": root / cls.RAW_DIRNAME,
            "cache": root / cls.CACHE_DIRNAME,
            "results": root / cls.RESULTS_DIRNAME,
            "reports": root / cls.REPORTS_DIRNAME,
            "pcfd": root / cls.PCFD_DIRNAME,
        }

    @classmethod
    def project_db_path(cls, project_uuid: str) -> Path:
        return cls.project_root(project_uuid) / "project.db"


    @classmethod
    def _pinyin_abbr(cls, name: str) -> str:
        slug = ""
        try:
            from pypinyin import lazy_pinyin, Style  # type: ignore
            letters = lazy_pinyin(name, style=Style.FIRST_LETTER)
            slug = ''.join([s[:1] for s in letters if s]).lower()
        except Exception:
            # Fallback: try punycode transliteration to ASCII letters/digits
            try:
                ascii_name = name.encode('punycode').decode('ascii')
            except Exception:
                ascii_name = name
            import re, hashlib
            slug = re.sub(r"[^A-Za-z0-9]+", "", ascii_name).lower()
            if not slug:
                # deterministic short hash to avoid generic 'project' directory
                slug = 'p' + hashlib.md5(name.encode('utf-8')).hexdigest()[:8]
        slug = slug or "project"
        return slug[:32]

    @classmethod
    def unique_project_dirname(cls, base: str) -> str:
        base = (base or "project").strip().strip("-_") or "project"
        candidate = base
        i = 1
        while (cls.PROJECTS_ROOT / candidate).exists():
            i += 1
            candidate = f"{base}-{i}"
        return candidate

    @classmethod
    def ensure_project_dirs_by_name(cls, project_uuid: str, project_name: str) -> dict[str, Path]:
        """
        使用项目名称的拼音缩写（必要时添加唯一后缀）创建项目根目录，
        然后创建标准子文件夹和 pcfd 目录。
        """
        cls.ensure_base_dirs()
        abbr = cls._pinyin_abbr(project_name)
        dirname = cls.unique_project_dirname(abbr)
        root = cls.PROJECTS_ROOT / dirname
        root.mkdir(parents=True, exist_ok=True)
        # Ensure standard subdirs
        (root / cls.RAW_DIRNAME).mkdir(parents=True, exist_ok=True)
        (root / cls.CACHE_DIRNAME).mkdir(parents=True, exist_ok=True)
        (root / cls.RESULTS_DIRNAME).mkdir(parents=True, exist_ok=True)
        (root / cls.REPORTS_DIRNAME).mkdir(parents=True, exist_ok=True)
        (root / cls.PCFD_DIRNAME).mkdir(parents=True, exist_ok=True)
        return {
            "root": root,
            "raw": root / cls.RAW_DIRNAME,
            "cache": root / cls.CACHE_DIRNAME,
            "results": root / cls.RESULTS_DIRNAME,
            "reports": root / cls.REPORTS_DIRNAME,
            "pcfd": root / cls.PCFD_DIRNAME,
            "dirname": dirname,
        }

    @classmethod
    def pcfd_index_path(cls, root_dir: Path) -> Path:
        return Path(root_dir) / cls.PCFD_DIRNAME / "index.json"

    @classmethod
    def save_pcfd_index(cls, root_dir: Path, data: dict) -> None:
        import json
        p = cls.pcfd_index_path(root_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_pcfd_index(cls, root_dir: Path) -> dict | None:
        import json
        p = cls.pcfd_index_path(root_dir)
        if not p.exists():
            return None
        try:
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

    _uuid_root_cache: dict[str, Path] = {}

    @classmethod
    def resolve_project_root(cls, project_uuid: str) -> Path:
        # Cached
        if project_uuid in cls._uuid_root_cache:
            return cls._uuid_root_cache[project_uuid]
        # 1) legacy path
        legacy = cls.PROJECTS_ROOT / project_uuid
        if legacy.exists():
            cls._uuid_root_cache[project_uuid] = legacy
            return legacy
        # 2) scan projects/*/pcfd/index.json
        try:
            for sub in cls.PROJECTS_ROOT.iterdir():
                if not sub.is_dir():
                    continue
                idx = cls.pcfd_index_path(sub)
                if not idx.exists():
                    continue
                try:
                    import json
                    with open(idx, 'r', encoding='utf-8') as f:
                        obj = json.load(f)
                    puid = str(((obj or {}).get('project') or {}).get('uuid') or '')
                    if puid and puid == project_uuid:
                        cls._uuid_root_cache[project_uuid] = sub
                        return sub
                except Exception:
                    continue
        except Exception:
            pass
        # fallback to legacy current design (not created)
        fallback = legacy
        cls._uuid_root_cache[project_uuid] = fallback
        return fallback

    @classmethod
    def append_pcfd_asset_for_uuid(cls, project_uuid: str, field: str, path_str: str) -> None:
        """如果可用，将资产路径追加到 pcfd/assets[field] 数组中。"""
        root = cls.resolve_project_root(project_uuid)
        idx = cls.load_pcfd_index(root)
        if idx is None:
            return
        assets = idx.setdefault('assets', {})
        arr = assets.setdefault(field, [])
        if path_str not in arr:
            arr.append(path_str)
            cls.save_pcfd_index(root, idx)
