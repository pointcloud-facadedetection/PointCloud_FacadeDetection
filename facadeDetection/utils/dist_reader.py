"""兼容读取 FLS 转换器产生的距离文件和测站元数据。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
import mmap

@dataclass
class DistData:
    ranges_m: np.ndarray
    scan_origins: np.ndarray
    source: str = "computed"
    unit: str = "m"
    warnings: list[str] = field(default_factory=list)


def _origins(metadata) -> np.ndarray:
    if metadata is None:
        return np.zeros((1, 3), dtype=np.float64)
    value = getattr(metadata, "scan_origin", None)
    if value is None:
        value = getattr(metadata, "scan_origins", None)
    if value is None and isinstance(metadata, dict):
        value = metadata.get("scan_origins", metadata.get("scan_origin"))
    if value is None:
        transform = getattr(metadata, "transform_to_global", None)
        if transform is None and isinstance(metadata, dict):
            transform = metadata.get("transformToGlobal")
        if transform is not None:
            try:
                value = np.asarray(transform, dtype=float)[:3, 3]
            except Exception:
                value = None
    if value is None:
        return np.zeros((1, 3), dtype=np.float64)
    arr = np.asarray(value, dtype=float)
    return arr.reshape(1, 3) if arr.size == 3 else arr.reshape(-1, 3)


def _read_values(path: Path, expected_count: int | None = None) -> np.ndarray:
    # .dist 的实现版本可能是文本，也可能是 float32/float64 原始数组。
    # Binary distance files can be hundreds of MB. Avoid an additional bytes
    # copy; mmap is released as soon as the converted array is returned.
    with path.open('rb') as stream:
        size = stream.seek(0, 2)
        stream.seek(0)
        if size and size % 4 == 0 and (expected_count is None or size // 4 == expected_count):
            try:
                with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                    if mapped[:64].find(b'\x00') >= 0:
                        values = np.frombuffer(mapped, dtype='<f4').astype(np.float32)
                        return values.copy()
            except (ValueError, OSError):
                pass
        stream.seek(0)
        raw = stream.read()
    try:
        text = raw.decode("utf-8")
        values = np.fromstring(text.replace(",", " "), sep=" ")
        if len(values):
            return values.astype(np.float64)
    except UnicodeDecodeError:
        pass
    candidates = []
    for dtype in ("<f4", "<f8"):
        if len(raw) % np.dtype(dtype).itemsize == 0:
            candidates.append(np.frombuffer(raw, dtype=dtype).astype(np.float64))
    if not candidates:
        raise ValueError("无法识别 .dist 格式")
    return max(candidates, key=len)


def read_dist(path: str | Path | None, points: np.ndarray, metadata=None) -> DistData:
    """读取并校验距离数组；失败时按真实测站坐标安全回退计算。"""
    # Keep the distance pipeline in float32; this halves the working set for
    # large scans while remaining well below inspection tolerances.
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    origins = _origins(metadata)
    warnings = []
    values = None
    source = "computed"
    unit = "m"
    if path and Path(path).exists():
        p = Path(path)
        try:
            values = _read_values(p, expected_count=len(pts))
            sidecar = p.with_suffix(p.suffix + ".json")
            if sidecar.exists():
                info = json.loads(sidecar.read_text(encoding="utf-8"))
                unit = str(info.get("unit", "m")).lower()
                if info.get("scan_origin") is not None:
                    origins = np.asarray(info["scan_origin"], dtype=float).reshape(-1, 3)
            if len(values) != len(pts):
                raise ValueError(f"点数不匹配 dist={len(values)} ply={len(pts)}")
            if unit in ("mm", "millimeter", "millimeters"):
                values /= 1000.0
                unit = "m"
            if unit not in ("m", "meter", "meters"):
                raise ValueError(f"不支持的距离单位: {unit}")
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError("距离包含 NaN/Inf/负值")
            source = "dist"
        except Exception as exc:
            warnings.append(f".dist 不可用，已重新计算: {exc}")
            values = None
    if values is None:
        if len(pts):
            values = np.full(len(pts), np.inf, dtype=np.float32)
            for origin in origins.astype(np.float32, copy=False):
                delta = pts - origin
                values = np.minimum(values, np.sqrt(np.einsum('ij,ij->i', delta, delta)))
        else:
            values = np.empty(0, dtype=np.float32)
    return DistData(np.asarray(values, dtype=np.float32), origins.astype(np.float32), source, "m", warnings)