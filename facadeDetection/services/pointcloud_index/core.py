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
    # 预处理参数和 processed-raw -> FLS source-raw 的映射不属于索引核心，
    # 作为旁路元数据保存，避免改变 VoxelCascadeIndex 的三级语义。
    metadata: dict | None = None

    @property
    def proxy_points(self) -> np.ndarray:
        return self.index.proxy_points

    @property
    def proxy_colors(self) -> np.ndarray | None:
        return self.index.proxy_colors

class VoxelCascadeIndex:
    """CSR 反向索引：voxel -> raw ids，proxy 行号与 voxel 行号一致。"""
    voxel_size = VOXEL_SIZE_M

    def __init__(self, voxel_keys, voxel_offsets, raw_order, proxy_points,
                 proxy_colors=None, origin=None, voxel_size=None):
        self.voxel_keys = np.ascontiguousarray(voxel_keys, dtype=np.int32).reshape(-1, 3)
        self.voxel_offsets = np.ascontiguousarray(voxel_offsets, dtype=np.int64).reshape(-1)
        self.raw_order = np.ascontiguousarray(raw_order, dtype=np.int32).reshape(-1)
        self.proxy_points = np.ascontiguousarray(proxy_points, dtype=np.float32).reshape(-1, 3)
        self.proxy_colors = None if proxy_colors is None else np.ascontiguousarray(proxy_colors)
        self.origin = np.asarray(origin, dtype=np.float32).reshape(3)
        self.voxel_size = float(voxel_size or VOXEL_SIZE_M)
        self._raw_to_voxel = None
        if len(self.voxel_offsets) != len(self.voxel_keys) + 1:
            raise ValueError("invalid CSR offsets")

    @classmethod
    def build(cls, raw: RawPointStore, origin=None, voxel_size=None) -> "VoxelCascadeIndex":
        if raw.count == 0:
            origin = np.zeros(3, dtype=np.float32) if origin is None else origin
            return cls(np.empty((0,3), np.int32), np.array([0], np.int64),
                       np.empty(0, np.int32), np.empty((0,3), np.float32), origin=origin,
                       voxel_size=voxel_size)
        p = raw.points
        vs = float(voxel_size or cls.voxel_size)
        origin = (np.floor(np.min(p, axis=0) / vs) * vs
                  if origin is None else np.asarray(origin, dtype=np.float32))
        keys = np.floor((p - origin) / vs).astype(np.int32)
        order = np.lexsort((keys[:,2], keys[:,1], keys[:,0]))
        sk = keys[order]
        starts = np.r_[0, np.flatnonzero(np.any(sk[1:] != sk[:-1], axis=1)) + 1]
        offsets = np.r_[starts, len(order)].astype(np.int64)
        voxel_keys = sk[starts]
        # reduceat avoids one Python loop and the temporary per-voxel index arrays.
        counts = np.diff(offsets).astype(np.float32)
        sums = np.add.reduceat(p[order].astype(np.float32, copy=False), starts, axis=0)
        proxy = np.ascontiguousarray(sums / counts[:, None], dtype=np.float32)
        pcol = None
        if raw.colors is not None:
            csum = np.add.reduceat(raw.colors[order].astype(np.float32, copy=False), starts, axis=0)
            pcol = np.ascontiguousarray(csum / counts[:, None], dtype=np.float32)
        result = cls(voxel_keys, offsets, order, proxy, pcol, origin, voxel_size=vs)
        log.debug("[PCFD] index.build.done raw=%d voxel=%d ratio=%.4f", raw.count,
                  len(proxy), len(proxy) / max(raw.count, 1))
        return result

    def proxy_ids_for_points(self, points) -> np.ndarray:
        """在不进行“全对最近邻搜索”的情况下，将代理坐标映射回代理行。"""
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if not len(pts) or not len(self.voxel_keys):
            return np.empty(0, dtype=np.int32)
        keys = np.floor((pts - self.origin) / self.voxel_size).astype(np.int32)
        # 向量化二进制查找.
        dtype = np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4")])
        table = self.voxel_keys.view(dtype).reshape(-1)
        query = keys.view(dtype).reshape(-1)
        sorted_table = table
        order = np.arange(len(table), dtype=np.int32)
        at = np.searchsorted(sorted_table, query)
        ids = np.full(len(query), -1, dtype=np.int32)
        valid = at < len(sorted_table)
        valid &= sorted_table[np.minimum(at, len(sorted_table) - 1)] == query
        ids[valid] = order[at[valid]]
        return ids

    def raw_to_voxel_ids(self, raw_ids=None) -> np.ndarray:
        """
        返回反向映射，且不构建 Python 字典。
        """
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
        """将原始 ID 按有界分块返回，同时精确保留体素判定结果。"""
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
        """
        将代理决策懒加载扩展至原始ID。

        这是代理空间决策与原始点云之间受支持的边界。该方法有意识地生成CSR块，
        而非显式构建一个可能包含数百万个元素的数组，同时确保
        对于所选体素集，每个原始点仍被精确包含一次。
        """
        yield from self.iter_raw_ids(self.proxy_to_voxel_ids(proxy_ids), chunk_size)

    def voxel_ids_for_proxy(self, proxy_ids) -> np.ndarray:
        ids = np.asarray(proxy_ids, dtype=np.int64).reshape(-1)
        return np.unique(ids[(ids >= 0) & (ids < len(self.voxel_keys))]).astype(np.int32)

    def proxy_to_voxel_ids(self, proxy_ids) -> np.ndarray:
        """
        显式的代理->体素映射。
        """
        return self.voxel_ids_for_proxy(proxy_ids)

    def iter_raw_points(self, voxel_ids, points=None, chunk_size: int = 1_000_000):
        """Yield ``(raw_ids, points)`` in bounded chunks without large copies."""
        source = self.raw_order if points is None else np.asarray(points)
        for raw_ids in self.iter_raw_ids(voxel_ids, chunk_size=chunk_size):
            yield raw_ids, source[raw_ids]

    def voxel_to_raw_ids(self, voxel_ids, deduplicate=True) -> np.ndarray:
        vids = self.voxel_ids_for_proxy(voxel_ids)
        if not len(vids): return np.empty(0, np.int32)
        parts = [self.raw_order[self.voxel_offsets[v]:self.voxel_offsets[v+1]] for v in vids]
        result = np.concatenate(parts).astype(np.int32, copy=False)
        return np.unique(result) if deduplicate else result

    def iter_raw_points_for_proxy(self, proxy_ids, points, colors=None,
                                  chunk_size: int = 1_000_000):
        """
        将代理决策懒加载地展开为原始点块。
        """
        source = np.asarray(points)
        colour_source = None if colors is None else np.asarray(colors)
        for raw_ids in self.iter_raw_ids_for_proxy(proxy_ids, chunk_size):
            yield raw_ids, source[raw_ids], (None if colour_source is None else colour_source[raw_ids])

    def raw_count_for_proxy(self, proxy_ids) -> int:
        """返回精确的原始扩展大小，且不分配原始 ID。"""
        return self.raw_count_for_voxels(self.proxy_to_voxel_ids(proxy_ids))

    def raw_count_for_voxels(self, voxel_ids) -> int:
        vids = self.voxel_ids_for_proxy(voxel_ids)
        return int(np.sum(np.diff(self.voxel_offsets)[vids], dtype=np.int64)) if len(vids) else 0

    def proxy_to_raw_ids(self, proxy_ids, deduplicate=True) -> np.ndarray:
        return self.voxel_to_raw_ids(proxy_ids, deduplicate)

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