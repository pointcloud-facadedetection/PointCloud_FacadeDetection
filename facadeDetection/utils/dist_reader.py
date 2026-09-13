"""兼容读取 FLS 转换器产生的距离文件和测站元数据。

三条导入链路（FLS 转换 PLY / E57 转换 PLY / 原生 PLY 上传）在此统一为
同一份"距离密度域"契约：

- dist: 存在合法的 .dist伴随文件，直接使用其测距值；
- station_origin: 无.dist，但能从元数据/E57 头部取得真实测站坐标，用测站欧氏距离喂给同一张壳层表；
- bbox_normalized: 两者皆无（典型为原生 PLY 上传），把点云的绝对分布距离按
                   d / d_p99归一化后线性映射到壳层区间，保证分层下采样仍然生效。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
import mmap

# 归一化分支把 d/d_p99 映射到的距离跨度（米）。
# 100m 覆盖 stratified_proxy_build 默认壳层表的最后一档，使近处点落到
# 小体素壳层、远处点落到大体素壳层，从而复现"近密远疏"的自适应效果。
NORMALIZED_RANGE_SPAN_M = 100.0
# 归一化时忽略极端外点对 p99 的影响。
NORMALIZED_RANGE_CLIP = 1.5


@dataclass
class DistData:
    ranges_m: np.ndarray
    scan_origins: np.ndarray
    source: str = "computed"
    unit: str = "m"
    warnings: list[str] = field(default_factory=list)
    # 'dist' | 'station_origin' | 'bbox_normalized'
    density_source: str = "dist"

    @property
    def is_metric(self) -> bool:
        """距离是否为真实物理量（米）。 """
        return self.density_source in ("dist", "station_origin")

    def __post_init__(self):
        self.ranges_m = np.ascontiguousarray(
            np.asarray(self.ranges_m, dtype=np.float32).reshape(-1))
        self.scan_origins = np.asarray(self.scan_origins, dtype=np.float32).reshape(-1, 3)


def _origins(metadata) -> np.ndarray:
    """从元数据中提取真实测站坐标；提取不到时返回空数组。 """
    if metadata is None:
        return np.empty((0, 3), dtype=np.float64)
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
        return np.empty((0, 3), dtype=np.float64)
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    arr = arr.reshape(1, 3) if arr.size == 3 else arr.reshape(-1, 3)
    if not np.all(np.isfinite(arr)):
        return np.empty((0, 3), dtype=np.float64)
    return arr


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


def _euclidean_from_origins(pts: np.ndarray, origins: np.ndarray) -> np.ndarray:
    """多测站取最近测站距离。"""
    values = np.full(len(pts), np.inf, dtype=np.float32)
    for origin in origins.astype(np.float32, copy=False):
        delta = pts - origin
        values = np.minimum(
            values, np.sqrt(np.einsum('ij,ij->i', delta, delta)).astype(np.float32))
    return values


def _normalized_bbox_ranges(pts: np.ndarray) -> np.ndarray:
    """无任何测站信息时的确定性距离代理。 """
    if not len(pts):
        return np.empty(0, dtype=np.float32)
    center = np.asarray(pts.mean(axis=0), dtype=np.float32)
    delta = pts - center
    d = np.sqrt(np.einsum('ij,ij->i', delta, delta)).astype(np.float32)
    finite = d[np.isfinite(d)]
    if finite.size == 0:
        return np.zeros(len(pts), dtype=np.float32)
    p99 = float(np.percentile(finite, 99.0))
    if not np.isfinite(p99) or p99 <= 1e-6:
        return np.zeros(len(pts), dtype=np.float32)
    ratio = np.clip(d / p99, 0.0, NORMALIZED_RANGE_CLIP)
    return (ratio * NORMALIZED_RANGE_SPAN_M).astype(np.float32)


def read_dist(path: str | Path | None, points: np.ndarray, metadata=None,
              scan_origin=None) -> DistData:
    """读取并校验距离数组；不可用时按确定性规则生成等价密度域。 """
    # Keep the distance pipeline in float32; this halves the working set for
    # large scans while remaining well below inspection tolerances.
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    explicit_origins = None
    if scan_origin is not None:
        candidate = np.asarray(scan_origin, dtype=float)
        if candidate.size:
            candidate = (candidate.reshape(1, 3) if candidate.size == 3
                         else candidate.reshape(-1, 3))
            if np.all(np.isfinite(candidate)):
                explicit_origins = candidate
    origins = explicit_origins if explicit_origins is not None else _origins(metadata)
    warnings = []
    values = None
    source = "computed"
    unit = "m"
    density_source = "station_origin"
    if path and Path(path).exists():
        p = Path(path)
        try:
            values = _read_values(p, expected_count=len(pts))
            sidecar = p.with_suffix(p.suffix + ".json")
            if sidecar.exists():
                info = json.loads(sidecar.read_text(encoding="utf-8"))
                unit = str(info.get("unit", "m")).lower()
                if info.get("scan_origin") is not None and explicit_origins is None:
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
            density_source = "dist"
        except Exception as exc:
            warnings.append(f".dist 不可用，已重新计算: {exc}")
            values = None
    if values is None:
        if len(pts) and len(origins):
            # 分支 2：有真实测站 → 与 FLS 链路同构的欧氏距离
            values = _euclidean_from_origins(pts, origins)
            density_source = "station_origin"
            source = "computed"
        else:
            # 分支 3：无任何测站 → 确定性归一化距离，保证分层仍然生效
            values = _normalized_bbox_ranges(pts)
            density_source = "bbox_normalized"
            source = "normalized"
            origins = np.empty((0, 3), dtype=np.float64)
            if len(pts):
                warnings.append(
                    "缺少测站信息，已按包围盒归一化距离构建分层密度域")
        if len(values) and not np.all(np.isfinite(values)):
            values = np.nan_to_num(values, nan=0.0, posinf=float(NORMALIZED_RANGE_SPAN_M))
        values = np.maximum(values, 0.0)
    return DistData(np.asarray(values, dtype=np.float32),
                    np.asarray(origins, dtype=np.float32).reshape(-1, 3),
                    source, "m", warnings, density_source)
