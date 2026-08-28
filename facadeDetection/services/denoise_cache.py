"""项目级去噪点云缓存。

去噪是一次性的重计算过程（分层降采样 + 自适应离群点剔除）。同一个项目里
一旦对某个原始点云执行过去噪，就把去噪后的 proxy 点云连同 CSR 映射等
metadata 落盘缓存到该项目的 cache 目录下；下次重新打开项目时，只要源文件
未发生变化，就直接读取缓存、跳过分层降采样与去噪计算，并且无需用户重新
点击"去噪"按钮。

缓存文件不通过 FileAsset 登记（沿用 cache/ui_prefs.json 的既有约定），
因此不受 gc_project() 的场景级清理影响，也不会污染点云资产列表。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d

from config.storage import Storage

_PLY_SUFFIX = ".denoised.ply"
_META_SUFFIX = ".denoised.meta.json"


def _cache_stem(source_path: str) -> str:
    resolved = str(Path(source_path).resolve())
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:16]
    return f"{Path(source_path).stem}.{digest}"


def cache_paths(project_uuid: str, source_path: str) -> tuple[Path, Path]:
    cache_dir = Storage.ensure_project_dirs(project_uuid)["cache"]
    stem = _cache_stem(source_path)
    return cache_dir / f"{stem}{_PLY_SUFFIX}", cache_dir / f"{stem}{_META_SUFFIX}"


def _fingerprint(source_path: str) -> Optional[dict]:
    try:
        st = Path(source_path).stat()
        return {"size": int(st.st_size), "mtime": float(st.st_mtime)}
    except OSError:
        return None


def save(project_uuid: str, source_path: str, points, colors, metadata: Optional[dict]) -> bool:
    """把去噪后的 proxy 点云与 metadata 写入项目缓存目录。"""
    if not project_uuid or not source_path:
        return False
    try:
        ply_path, meta_path = cache_paths(project_uuid, source_path)
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        if colors is not None:
            cols = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
            if len(cols) == len(pts):
                pcd.colors = o3d.utility.Vector3dVector(cols)
        o3d.io.write_point_cloud(str(ply_path), pcd, write_ascii=False, compressed=True)
        payload = {
            "source_path": str(Path(source_path).resolve()),
            "source_fingerprint": _fingerprint(source_path),
            "point_count": int(len(pts)),
            "metadata": metadata or {},
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        print(f"[PCFD] denoise_cache.saved source={source_path} points={len(pts)}", flush=True)
        return True
    except Exception as exc:
        print(f"[PCFD] denoise_cache.save_failed source={source_path} reason={exc}", flush=True)
        return False


def load(project_uuid: str, source_path: str) -> Optional[dict]:
    """若缓存存在且源文件未变化，返回 {points, colors, metadata}；否则返回 None。"""
    if not project_uuid or not source_path:
        return None
    ply_path, meta_path = cache_paths(project_uuid, source_path)
    if not ply_path.exists() or not meta_path.exists():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None

    fp_now = _fingerprint(source_path)
    fp_cached = payload.get("source_fingerprint")
    if not fp_now or not fp_cached or fp_now != fp_cached:
        print(f"[PCFD] denoise_cache.stale source={source_path}（源文件已变化，缓存失效）", flush=True)
        return None

    try:
        pcd = o3d.io.read_point_cloud(str(ply_path))
        pts = np.ascontiguousarray(np.asarray(pcd.points, dtype=np.float32))
        cols = None
        if pcd.has_colors():
            cols = np.ascontiguousarray(np.asarray(pcd.colors, dtype=np.float32))
        expected = int(payload.get("point_count", -1))
        if expected >= 0 and len(pts) != expected:
            return None
    except Exception as exc:
        print(f"[PCFD] denoise_cache.load_failed source={source_path} reason={exc}", flush=True)
        return None

    print(f"[PCFD] denoise_cache.hit source={source_path} points={len(pts)}", flush=True)
    return {"points": pts, "colors": cols, "metadata": payload.get("metadata") or {}}


def invalidate(project_uuid: str, source_path: str) -> None:
    if not project_uuid or not source_path:
        return
    ply_path, meta_path = cache_paths(project_uuid, source_path)
    for p in (ply_path, meta_path):
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass


__all__ = ["cache_paths", "save", "load", "invalidate"]
