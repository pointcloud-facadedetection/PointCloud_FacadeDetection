from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import logging

log = logging.getLogger("facadeDetection.pointcloud_index")

VOXEL_SIZE_M = 0.05


@dataclass
class RawPointStore:
    points: np.ndarray
    colors: np.ndarray | None = None

    @property
    def count(self) -> int:
        return int(len(self.points))

    @classmethod
    def from_arrays(cls, points, colors=None) -> "RawPointStore":
        p = np.ascontiguousarray(np.asarray(points, dtype=np.float32).reshape(-1, 3))
        c = None if colors is None else np.ascontiguousarray(np.asarray(colors).reshape(-1, 3))
        if c is not None and len(c) != len(p):
            raise ValueError("colors length must match points")
        return cls(p, c)


@dataclass
class DecisionSet:
    dataset_id: str
    voxel_ids: np.ndarray
    source: str = "unknown"
    operation_id: str = ""
    version: int = 0

    @property
    def voxel_count(self) -> int:
        return int(len(self.voxel_ids))


@dataclass
class PointCloudDataset:
    dataset_id: str
    raw: RawPointStore
    index: "VoxelCascadeIndex"
    metadata: dict | None = None

    @property
    def revision(self) -> str:
        """处理版本；结果必须绑定到该版本，避免旧索引静默复用。"""
        return str((self.metadata or {}).get("revision") or self.dataset_id)

    @property
    def proxy_points(self) -> np.ndarray:
        return self.index.proxy_points

    @property
    def proxy_colors(self) -> np.ndarray | None:
        return self.index.proxy_colors

    @property
    def proxy_store(self) -> RawPointStore:
        return self.raw

    @property
    def processed_raw_points(self) -> np.ndarray:
        """当前处理域高保真点。分层数据优先使用其 source 数组。"""
        mapped = self.index.get_source_points()
        return mapped if mapped is not None else self.raw.points


class VoxelCascadeIndex:
    """CSR 反向索引：voxel → raw ids，proxy 行号与 voxel 行号一致。

    架构融合后：
    - 距离分层数据：proxy 即 stratified_proxy_build 输出，1:1 映射到 source。
      不再进行二次体素化，CSR 直接记录 proxy→source 映射。
    - 标准体素数据：走传统体素化路径，proxy 为体素重心，voxel→raw 为聚合映射。
    """
    voxel_size = VOXEL_SIZE_M

    def __init__(self, voxel_keys, voxel_offsets, raw_order, proxy_points,
                 proxy_colors=None, origin=None, voxel_size=None,
                 source_points=None,
                 source_colors=None,
                 source_raw_offsets=None,
                 source_raw_indices=None,
                 is_stratified=False):
        self.voxel_keys = np.ascontiguousarray(voxel_keys, dtype=np.int32).reshape(-1, 3)
        self.voxel_offsets = np.ascontiguousarray(voxel_offsets, dtype=np.int64).reshape(-1)
        self.raw_order = np.ascontiguousarray(raw_order, dtype=np.int32).reshape(-1)
        self.proxy_points = np.ascontiguousarray(proxy_points, dtype=np.float32).reshape(-1, 3)
        self.proxy_colors = None if proxy_colors is None else np.ascontiguousarray(proxy_colors)
        self.origin = np.asarray(origin, dtype=np.float32).reshape(3)
        self.voxel_size = float(voxel_size or VOXEL_SIZE_M)
        self._raw_to_voxel = None
        self._is_stratified = bool(is_stratified)

        # === Source 映射 (CSR 格式唯一) ===
        self.source_points = None if source_points is None else np.asarray(source_points, dtype=np.float32)
        self.source_colors = None if source_colors is None else np.asarray(source_colors, dtype=np.float32)
        self.source_raw_offsets = source_raw_offsets
        self.source_raw_indices = source_raw_indices

        if self.source_raw_offsets is not None:
            self.source_raw_offsets = np.ascontiguousarray(self.source_raw_offsets, dtype=np.int64).reshape(-1)
            self.source_raw_indices = np.ascontiguousarray(self.source_raw_indices, dtype=np.int32).reshape(-1)
            if len(self.source_raw_offsets) != len(self.proxy_points) + 1:
                raise ValueError("source CSR offsets must match proxy rows")
            if (np.any(np.diff(self.source_raw_offsets) < 0) or
                    self.source_raw_offsets[-1] != len(self.source_raw_indices)):
                raise ValueError("invalid source CSR")
            if self.source_points is not None and len(self.source_raw_indices):
                if np.any(self.source_raw_indices < 0) or np.any(self.source_raw_indices >= len(self.source_points)):
                    raise ValueError("source CSR index out of bounds")

        if len(self.voxel_offsets) != len(self.voxel_keys) + 1:
            raise ValueError("invalid CSR offsets")

    @classmethod
    def build(cls, raw: RawPointStore, origin=None, voxel_size=None,
              source_points=None, source_colors=None,
              source_raw_offsets=None, source_raw_indices=None):
        """构建索引。

        融合路径：source_raw_offsets 存在时，raw.points 已经是 stratified proxy，
        每个点是一个独立代理，通过 CSR 直接映射到 source。不再体素化。
        """
        if raw.count == 0:
            origin = np.zeros(3, dtype=np.float32) if origin is None else origin
            return cls(np.empty((0, 3), np.int32), np.array([0], np.int64),
                       np.empty(0, np.int32), np.empty((0, 3), np.float32),
                       origin=origin, voxel_size=voxel_size,
                       source_points=source_points,
                       source_raw_offsets=source_raw_offsets,
                       source_raw_indices=source_raw_indices,
                       is_stratified=source_raw_offsets is not None)

        # === 融合路径：距离分层代理，1:1 映射，跳过体素化 ===
        if source_raw_offsets is not None:
            n = raw.count
            keys = np.column_stack((np.arange(n, dtype=np.int32),
                                    np.zeros((n, 2), dtype=np.int32)))
            offsets = np.arange(n + 1, dtype=np.int64)
            raw_order = np.arange(n, dtype=np.int32)
            return cls(keys, offsets, raw_order, raw.points, raw.colors,
                       np.zeros(3, dtype=np.float32), voxel_size=1.0,
                       source_points=source_points, source_colors=source_colors,
                       source_raw_offsets=source_raw_offsets,
                       source_raw_indices=source_raw_indices,
                       is_stratified=True)

        # === 标准路径：体素下采样，需要体素化 ===
        p = raw.points
        vs = float(voxel_size or cls.voxel_size)
        if vs < 1e-6:
            log.warning("[PCFD] voxel_size %.6f too small, clamping to %.6f", vs, cls.voxel_size)
            vs = cls.voxel_size

        origin = (np.floor(np.min(p, axis=0) / vs) * vs
                  if origin is None else np.asarray(origin, dtype=np.float32))

        keys = np.floor((p - origin) / vs).astype(np.int32)
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        sk = keys[order]

        key_changes = np.concatenate([
            [True],
            np.any(sk[1:] != sk[:-1], axis=1)
        ])
        starts = np.flatnonzero(key_changes)
        offsets = np.r_[starts, len(order)].astype(np.int64)
        voxel_keys = sk[starts]

        counts = np.diff(offsets).astype(np.float32)
        sums = np.add.reduceat(p[order].astype(np.float32, copy=False), starts, axis=0)
        proxy = np.ascontiguousarray(sums / counts[:, None], dtype=np.float32)

        pcol = None
        if raw.colors is not None:
            csum = np.add.reduceat(raw.colors[order].astype(np.float32, copy=False), starts, axis=0)
            pcol = np.ascontiguousarray(csum / counts[:, None], dtype=np.float32)

        result = cls(voxel_keys, offsets, order, proxy, pcol, origin, voxel_size=vs,
                     source_points=source_points, source_colors=source_colors,
                     is_stratified=False)

        log.debug("[PCFD] index.build.done raw=%d voxel=%d ratio=%.4f vs=%.4f source=%s",
                  raw.count, len(proxy), len(proxy) / max(raw.count, 1),
                  vs, "yes" if source_points is not None else "no")
        return result

    def proxy_ids_for_points(self, points) -> np.ndarray:
        """【标准路径】坐标反查 proxy 索引。分层数据请勿使用。"""
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if not len(pts) or not len(self.voxel_keys):
            return np.empty(0, dtype=np.int32)
        keys = np.floor((pts - self.origin) / self.voxel_size).astype(np.int32)
        dtype = np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4")])
        table = self.voxel_keys.view(dtype).reshape(-1)
        query = keys.view(dtype).reshape(-1)
        at = np.searchsorted(table, query)
        ids = np.full(len(query), -1, dtype=np.int32)
        valid = at < len(table)
        valid &= table[np.minimum(at, len(table) - 1)] == query
        ids[valid] = at[valid]
        return ids

    def raw_to_voxel_ids(self, raw_ids=None) -> np.ndarray:
        if self._raw_to_voxel is None:
            result = np.empty(len(self.raw_order), dtype=np.int32)
            for voxel_id, (start, end) in enumerate(zip(self.voxel_offsets[:-1], self.voxel_offsets[1:])):
                result[self.raw_order[start:end]] = voxel_id
            self._raw_to_voxel = result
        if raw_ids is None:
            return self._raw_to_voxel
        ids = np.asarray(raw_ids, dtype=np.int64).reshape(-1)
        out = np.full(len(ids), -1, dtype=np.int32)
        valid = (ids >= 0) & (ids < len(self._raw_to_voxel))
        out[valid] = self._raw_to_voxel[ids[valid]]
        return out

    def iter_raw_ids(self, voxel_ids, chunk_size: int = 1_000_000):
        chunk_size = max(int(chunk_size), 1)
        vids = self.voxel_ids_for_proxy(voxel_ids)
        if not len(vids):
            return
        parts = (self.raw_order[self.voxel_offsets[v]:self.voxel_offsets[v + 1]] for v in vids)
        buf = []
        size = 0
        for part in parts:
            if len(part):
                buf.append(part)
                size += len(part)
            if size >= chunk_size:
                yield np.concatenate(buf).astype(np.int32, copy=False)
                buf, size = [], 0
        if size:
            yield np.concatenate(buf).astype(np.int32, copy=False)

    def iter_raw_ids_for_proxy(self, proxy_ids, chunk_size: int = 1_000_000):
        yield from self.iter_raw_ids(self.proxy_to_voxel_ids(proxy_ids), chunk_size)

    def voxel_ids_for_proxy(self, proxy_ids) -> np.ndarray:
        ids = np.asarray(proxy_ids, dtype=np.int64).reshape(-1)
        return np.unique(ids[(ids >= 0) & (ids < len(self.voxel_keys))]).astype(np.int32)

    def proxy_to_voxel_ids(self, proxy_ids) -> np.ndarray:
        return self.voxel_ids_for_proxy(proxy_ids)

    def iter_raw_points(self, voxel_ids, points=None, chunk_size: int = 1_000_000):
        source = self.raw_order if points is None else np.asarray(points)
        for raw_ids in self.iter_raw_ids(voxel_ids, chunk_size=chunk_size):
            yield raw_ids, source[raw_ids]

    def voxel_to_raw_ids(self, voxel_ids, deduplicate=True) -> np.ndarray:
        vids = self.voxel_ids_for_proxy(voxel_ids)
        if not len(vids):
            return np.empty(0, np.int32)
        parts = [self.raw_order[self.voxel_offsets[v]:self.voxel_offsets[v+1]] for v in vids]
        result = np.concatenate(parts).astype(np.int32, copy=False)
        return np.unique(result) if deduplicate else result

    def iter_raw_points_for_proxy(self, proxy_ids, points, colors=None,
                                  chunk_size: int = 1_000_000):
        source = np.asarray(points)
        colour_source = None if colors is None else np.asarray(colors)
        for raw_ids in self.iter_raw_ids_for_proxy(proxy_ids, chunk_size):
            yield raw_ids, source[raw_ids], (None if colour_source is None else colour_source[raw_ids])

    def raw_count_for_proxy(self, proxy_ids) -> int:
        return self.raw_count_for_voxels(self.proxy_to_voxel_ids(proxy_ids))

    def raw_count_for_voxels(self, voxel_ids) -> int:
        vids = self.voxel_ids_for_proxy(voxel_ids)
        return int(np.sum(np.diff(self.voxel_offsets)[vids], dtype=np.int64)) if len(vids) else 0

    def proxy_to_raw_ids(self, proxy_ids, deduplicate=True) -> np.ndarray:
        return self.voxel_to_raw_ids(proxy_ids, deduplicate)

    # ==================== Source 原始空间映射 (CSR 唯一) ====================

    def has_source_mapping(self) -> bool:
        return (self.source_points is not None and
                self.source_raw_offsets is not None and
                self.source_raw_indices is not None)

    def proxy_to_source_ids(self, proxy_ids, deduplicate=True) -> np.ndarray:
        """Proxy → Source 原始点索引 (CSR 向量化查询)。"""
        if not self.has_source_mapping():
            return self.proxy_to_raw_ids(proxy_ids, deduplicate)

        pids = np.asarray(proxy_ids, dtype=np.int64).reshape(-1)
        pids = pids[(pids >= 0) & (pids < len(self.proxy_points))]
        if len(pids) == 0:
            return np.empty(0, dtype=np.int32)

        starts = self.source_raw_offsets[pids]
        ends = self.source_raw_offsets[pids + 1]
        counts = ends - starts
        total = int(counts.sum())
        if total == 0:
            return np.empty(0, dtype=np.int32)

        offsets_in_group = np.arange(total, dtype=np.int64) - np.repeat(
            np.concatenate([[0], np.cumsum(counts[:-1], dtype=np.int64)]),
            counts.astype(np.int64)
        )
        global_offsets = np.repeat(starts, counts.astype(np.int64)) + offsets_in_group
        result = self.source_raw_indices[global_offsets].astype(np.int32, copy=False)
        return np.unique(result) if deduplicate else result

    # Explicit dual-domain names. Keep the old methods as compatibility aliases.
    def proxy_to_processed_raw_ids(self, proxy_ids, deduplicate=True) -> np.ndarray:
        return self.proxy_to_source_ids(proxy_ids, deduplicate)

    def processed_raw_to_proxy_ids(self, raw_ids) -> np.ndarray:
        return self.source_to_proxy_ids(raw_ids)

    def source_count_for_proxy(self, proxy_ids) -> int:
        if not self.has_source_mapping():
            return self.raw_count_for_proxy(proxy_ids)
        pids = np.asarray(proxy_ids, dtype=np.int64).reshape(-1)
        pids = pids[(pids >= 0) & (pids < len(self.proxy_points))]
        if len(pids) == 0:
            return 0
        return int(np.sum(self.source_raw_offsets[pids + 1] - self.source_raw_offsets[pids]))

    def source_to_proxy_ids(self, source_ids) -> np.ndarray:
        if not self.has_source_mapping():
            return self.raw_to_voxel_ids(source_ids)
        wanted = set(np.asarray(source_ids, dtype=np.int64).reshape(-1).tolist())
        if not wanted:
            return np.empty(0, dtype=np.int32)
        hits = []
        for pid, (start, end) in enumerate(zip(self.source_raw_offsets[:-1], self.source_raw_offsets[1:])):
            if any(int(value) in wanted for value in self.source_raw_indices[start:end]):
                hits.append(pid)
        return np.asarray(hits, dtype=np.int32)

    def get_source_points(self):
        return self.source_points

    def get_source_colors(self):
        return self.source_colors

    def query_aabb(self, min_bound, max_bound, raw_points, exact=True) -> np.ndarray:
        lo = np.asarray(min_bound, dtype=np.float32); hi = np.asarray(max_bound, dtype=np.float32)
        k0 = np.floor((lo - self.origin) / self.voxel_size).astype(np.int32)
        k1 = np.floor((hi - self.origin) / self.voxel_size).astype(np.int32)
        mask = np.all((self.voxel_keys >= k0) & (self.voxel_keys <= k1), axis=1)
        ids = self.voxel_to_raw_ids(np.flatnonzero(mask))
        if exact and len(ids):
            pts = np.asarray(raw_points)[ids]
            ids = ids[np.all((pts >= lo) & (pts <= hi), axis=1)]
        return ids