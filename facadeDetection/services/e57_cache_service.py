"""Single conversion boundary between persisted E57 assets and runtime PLY."""
from __future__ import annotations

import hashlib
import multiprocessing
from pathlib import Path

from config.storage import Storage
from services.dal.file_repo import FileRepo
from utils.e57_reader import convert_e57_to_ply
from utils.logging_utils import log_event


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _e57_convert_worker(
    source_path: str,
    cache_path: str,
    project_uuid: str,
    metadata: dict,
    queue,
) -> None:
    """Run E57→PLY conversion in a clean subprocess (spawn).

    Spawn gives the child a fresh address space so FLS converter's
    libe57.dll (and any other DLLs on its PATH) never pollute pye57.
    """
    try:
        info = convert_e57_to_ply(
            Path(source_path),
            Path(cache_path),
            project_uuid=project_uuid,
            metadata=metadata,
        )
        queue.put(("ok", info))
    except Exception as exc:
        queue.put(("error", str(exc)))


def ensure_e57_cache(project_uuid: str, asset, *, force: bool = False) -> Path:
    """Return a usable cache PLY for an E57 FileAsset.

    Cache PLY is written to the project's own cache/ directory with the same
    basename as the original E57 (e.g. project.e57 -> project.ply).  This
    guarantees:
      - predictable paths for viewport / report / detection pipelines
      - full compatibility with PLY-only business logic (denoise, facade,
        project recovery)
      - no extra hash subdirectories or sha256-prefixed filenames

    The original E57 path is kept in the FileAsset record for provenance only.
    """
    source = Path(asset.path or "").expanduser().resolve()
    if source.suffix.lower() != ".e57":
        return source

    # 项目私有 cache 目录：data/projects/<project_dir>/cache/
    cache_dir = Storage.ensure_project_dirs(project_uuid)["cache"]
    cache_path = cache_dir / f"{source.stem}.ply"

    meta = dict(asset.meta_json or {})

    # 1) Reuse by recorded cache_path (same asset)
    existing_meta = Path(meta.get("cache_path") or "")
    if not force and existing_meta.is_file() and existing_meta.suffix.lower() == ".ply":
        # 校验缓存文件有效性：非空且可读取
        if existing_meta.stat().st_size > 0:
            log_event(
                project_uuid,
                "e57.cache.reused",
                asset_id=asset.id,
                source_path=str(source),
                cache_path=str(existing_meta),
            )
            return existing_meta
        # 缓存损坏，强制重建
        force = True

    # 2) Reuse by same-name PLY in project cache (same project, re-import)
    if not force and cache_path.is_file():
        if cache_path.stat().st_size > 0:
            FileRepo.update_cache_metadata(
                project_uuid,
                asset.id,
                {
                    "cache_path": str(cache_path),
                    "cache_status": "ready",
                    "reused_by_name": True,
                    "source_path": str(source),
                },
            )
            log_event(
                project_uuid,
                "e57.cache.reused_by_name",
                asset_id=asset.id,
                source_path=str(source),
                cache_path=str(cache_path),
            )
            return cache_path
        # 缓存损坏，强制重建
        force = True

    # 3) Force rebuild: remove existing same-name artifacts first
    if force and cache_path.is_file():
        cache_path.unlink(missing_ok=True)
        cache_path.with_suffix(".ply.json").unlink(missing_ok=True)

    if not source.is_file():
        log_event(
            project_uuid,
            "e57.cache.source_missing",
            asset_id=asset.id,
            source_path=str(source),
            cache_path=str(cache_path),
        )
        raise FileNotFoundError(f"原始 E57 不存在，无法生成 cache PLY: {source}")

    # 4) 使用 spawn 子进程隔离 E57 转换，避免 FLS 转换器 DLL 污染 pye57
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(
        target=_e57_convert_worker,
        args=(
            str(source),
            str(cache_path),
            project_uuid,
            {
                "asset_id": asset.id,
                "source_sha256": asset.sha256,
                "source_size_bytes": asset.size_bytes,
            },
            q,
        ),
    )
    p.start()
    p.join()

    status, payload = q.get()
    if status == "error":
        raise RuntimeError(f"E57 转换失败: {payload}")

    info = payload
    info["cache_status"] = "ready"
    # 运行时指纹必须以 cache PLY 为基准，确保去噪/代理缓存与
    # cache 重建后的实际内容一致，而不是与原始 E57 文件。
    info["cache_sha256"] = _sha256(cache_path)
    info["cache_size_bytes"] = cache_path.stat().st_size
    info["cache_path"] = str(cache_path)
    FileRepo.update_cache_metadata(project_uuid, asset.id, info)
    return cache_path