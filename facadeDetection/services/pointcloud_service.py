from __future__ import annotations

from typing import Optional, Dict
import logging
import uuid

import numpy as np
import open3d as o3d

from algorithms.preprocess import denoise
from .pointcloud_index import RawPointStore, PointCloudDataset, VoxelCascadeIndex, DecisionSet

log = logging.getLogger("facadeDetection.pointcloud")


class PointCloudService:
    def __init__(self, viewport=None, render_service=None):
        self.viewport = viewport
        self.render_service = render_service
        self.datasets: Dict[str, PointCloudDataset] = {}
        self.source_assets: Dict[str, dict] = {}
        self.decisions: Dict[str, DecisionSet] = {}
        self._decision_versions: Dict[str, int] = {}
        # Runtime ownership is deliberately explicit.  The public service API
        # remains unchanged, while all in-memory datasets are scoped to this
        # project generation and can be disposed atomically on a switch.
        self._project_uuid = None
        self._project_generation = 0
    #Todo：project_generation
    @property
    def project_generation(self) -> int:
        return self._project_generation

    def set_project(self, project_uuid):
        """Bind the registry to a project; switching always drops old data."""
        project_uuid = str(project_uuid) if project_uuid else None
        # ``None`` is the unloaded state.  Restore may populate the registry
        # before the UI binds the project id, so binding from None must not
        # discard the freshly restored snapshot.
        if self._project_uuid is not None and project_uuid != self._project_uuid:
            self.clear_runtime()
            self._project_uuid = project_uuid
            self._project_generation += 1
        else:
            self._project_uuid = project_uuid

    def clear_runtime(self):
        """Release all point/index/decision arrays owned by the active project."""
        for dataset in self.datasets.values():
            dataset.raw.points = np.empty((0, 3), dtype=np.float32)
            dataset.raw.colors = None
            dataset.index.source_points = None
            dataset.index.source_colors = None
            dataset.index.source_raw_offsets = None
            dataset.index.source_raw_indices = None
            dataset.index._raw_to_voxel = None
        self.datasets.clear()
        self.source_assets.clear()
        self.decisions.clear()
        self._decision_versions.clear()

    def close_project(self):
        """Compatibility-friendly alias used by the project disposal pipeline."""
        self.clear_runtime()
        self._project_uuid = None
        self._project_generation += 1

    def register_dataset(self, dataset_id: str, points, colors=None, metadata=None) -> PointCloudDataset:
        raw = RawPointStore.from_arrays(points, colors)
        meta = metadata or {}
        source_points = None
        source_raw_offsets = None
        source_raw_indices = None
        if meta.get('proxy_source_offsets') is not None:
            source_points = self.source_assets.get(meta.get('source_id'), {}).get('points')
            source_raw_offsets = np.asarray(meta['proxy_source_offsets'], dtype=np.int64)
            source_raw_indices = np.asarray(meta['proxy_source_indices'], dtype=np.int32)
        source_colors = self.source_assets.get(meta.get('source_id'), {}).get('colors')
        index = VoxelCascadeIndex.build(
            raw, source_points=source_points, source_colors=source_colors,
            source_raw_offsets=source_raw_offsets, source_raw_indices=source_raw_indices)
        dataset = PointCloudDataset(dataset_id, raw, index, meta)
        self.datasets[dataset_id] = dataset
        self.decisions.pop(dataset_id, None)
        self._decision_versions.pop(dataset_id, None)
        return dataset

    def register_source_asset(self, source_id: str, points, colors=None, metadata=None):
        points = np.ascontiguousarray(np.asarray(points, dtype=np.float32).reshape(-1, 3))
        if colors is not None:
            colors = np.asarray(colors, dtype=np.float32)
            if colors.size % 3 != 0 or len(colors.reshape(-1, 3)) != len(points):
                colors = None
            else:
                colors = np.ascontiguousarray(np.clip(colors.reshape(-1, 3), 0.0, 1.0))
        self.source_assets[source_id] = {
            "points": points,
            "colors": colors,
            "metadata": metadata or {},
        }

    def get_source_asset(self, source_id):
        return self.source_assets.get(source_id)

    def get_metadata(self, dataset_id):
        dataset = self.datasets.get(dataset_id)
        return dataset.metadata if dataset else None

    def get_dataset(self, dataset_id: str) -> Optional[PointCloudDataset]:
        return self.datasets.get(dataset_id)

    def release_station_domain(self, station_id) -> None:
        """Release one station's raw/proxy/index domain after it is removed."""
        token = f":{station_id}"
        dataset_ids = [key for key in self.datasets if str(key).endswith(token)]
        for dataset_id in dataset_ids:
            dataset = self.datasets.pop(dataset_id, None)
            if dataset is not None:
                dataset.raw.points = np.empty((0, 3), dtype=np.float32)
                dataset.raw.colors = None
                dataset.index.source_points = None
                dataset.index.source_colors = None
                dataset.index.source_raw_offsets = None
                dataset.index.source_raw_indices = None
                dataset.index._raw_to_voxel = None
            self.decisions.pop(dataset_id, None)
            self._decision_versions.pop(dataset_id, None)
        for source_id in [key for key in self.source_assets
                          if str(key).endswith(f":{station_id}:source")]:
            asset = self.source_assets.pop(source_id, None)
            if asset is not None:
                asset['points'] = np.empty((0, 3), dtype=np.float32)
                asset['colors'] = None

    def bind_processing_cloud(self, cloud_name: str) -> Optional[str]:
        """Repair a restored viewport cloud's metadata from the dataset registry.

        Open3D scene restoration predates the in-memory dataset contract in
        some projects, so the cloud can exist while ``dataset_id`` is absent.
        Match only the filename suffix; never bind a station preview cloud.
        """
        vp = self.viewport
        if vp is None or not cloud_name or str(cloud_name).startswith('pcfd.station.'):
            return None
        data = vp.get_cloud_data(cloud_name) if hasattr(vp, 'get_cloud_data') else None
        if data is None:
            return None
        current = data.get('dataset_id')
        if current in self.datasets:
            return current
        leaf = str(cloud_name).replace('\\', '/').rsplit('/', 1)[-1]
        matches = [did for did in self.datasets if str(did).rsplit(':', 1)[-1] == leaf]
        if len(matches) != 1:
            return None
        dataset_id = matches[0]
        dataset = self.datasets[dataset_id]
        data.update({'dataset_id': dataset_id, 'domain': 'proxy',
                     'index_space': 'proxy_global',
                     'is_processing_cloud': True,
                     'proxy_ids': np.arange(len(data.get('pos', [])), dtype=np.int32)})
        if len(data.get('pos', [])) != len(dataset.proxy_points):
            print(f'[PCFD] cloud.bind_size_mismatch cloud={cloud_name} '
                  f'display={len(data.get("pos", []))} dataset={len(dataset.proxy_points)}', flush=True)
        print(f'[PCFD] cloud.bind_repaired cloud={cloud_name} dataset={dataset_id}', flush=True)
        return dataset_id

    def map_proxy_decision(self, dataset_id: str, proxy_ids, source: str = "unknown",
                           expand_raw: bool = True) -> np.ndarray | None:
        dataset = self.datasets[dataset_id]
        voxel_ids = dataset.index.voxel_ids_for_proxy(proxy_ids)
        version = self._decision_versions.get(dataset_id, 0) + 1
        self._decision_versions[dataset_id] = version
        decision = DecisionSet(dataset_id, voxel_ids, source,
                               operation_id=uuid.uuid4().hex, version=version)
        self.decisions[dataset_id] = decision
        if expand_raw:
            if dataset.index.has_source_mapping():
                return dataset.index.proxy_to_source_ids(proxy_ids)
            else:
                return dataset.index.voxel_to_raw_ids(voxel_ids)
        return None

    def raw_ids_for_aabb(self, dataset_id: str, min_bound, max_bound) -> np.ndarray:
        dataset = self.datasets[dataset_id]
        index = dataset.index
        proxy_mask = np.all((index.proxy_points >= np.asarray(min_bound)) &
                            (index.proxy_points <= np.asarray(max_bound)), axis=1)
        proxy_ids = np.flatnonzero(proxy_mask)
        if index.has_source_mapping():
            source_ids = index.proxy_to_source_ids(proxy_ids, deduplicate=True)
            source_points = index.get_source_points()
            if len(source_ids) and source_points is not None:
                pts = source_points[source_ids]
                keep = np.all((pts >= np.asarray(min_bound)) &
                              (pts <= np.asarray(max_bound)), axis=1)
                return source_ids[keep]
            return np.empty(0, dtype=np.int32)
        return index.query_aabb(min_bound, max_bound, index.proxy_points, exact=True)

    def set_dependencies(self, viewport=None, render_service=None):
        if viewport is not None:
            self.viewport = viewport
        if render_service is not None:
            self.render_service = render_service

    def _pick_active_cloud_name(self) -> Optional[str]:
        vp = self.viewport
        if vp is None:
            return None
        if hasattr(vp, "get_cloud_names") and hasattr(vp, "get_cloud_data"):
            for candidate in reversed(vp.get_cloud_names() or []):
                if str(candidate).startswith("pcfd.station."):
                    continue
                data = vp.get_cloud_data(candidate)
                if (data and data.get("dataset_id") and
                        data.get("domain", "proxy") == "proxy"):
                    return candidate
        name = getattr(vp, "_active_name", None)
        if (name and not str(name).startswith("pcfd.station.") and
                (not hasattr(vp, "get_cloud_data") or
                 (vp.get_cloud_data(name) or {}).get("domain", "proxy") == "proxy")):
            return name
        if hasattr(vp, "get_cloud_names"):
            names = [n for n in (vp.get_cloud_names() or [])
                     if (not str(n).startswith("pcfd.station.") and
                         (vp.get_cloud_data(n) or {}).get("domain", "proxy") == "proxy")]
            if names:
                return names[-1]
        return None

    def resolve_processing_cloud(self, preferred: Optional[str] = None) -> Optional[str]:
        """Return the proxy cloud bound to a registered dataset.

        Station clouds are deliberately display-only and must never be selected
        merely because they happen to be the last Open3D geometry.
        """
        vp = self.viewport
        if vp is None or not hasattr(vp, "get_cloud_data"):
            return None
        names = vp.get_cloud_names() if hasattr(vp, "get_cloud_names") else []
        candidates = ([preferred] if preferred else []) + list(reversed(names or []))
        seen = set()
        for name in candidates:
            if not name or name in seen or str(name).startswith("pcfd.station."):
                continue
            seen.add(name)
            data = vp.get_cloud_data(name) or {}
            dataset_id = self.bind_processing_cloud(name)
            if dataset_id and dataset_id in self.datasets and data.get("domain", "proxy") == "proxy":
                data.setdefault("is_processing_cloud", True)
                data.setdefault("index_space", "proxy_global")
                data.setdefault("proxy_ids", np.arange(len(data.get("pos", [])), dtype=np.int32))
                return name
        return self._pick_active_cloud_name()

    def _rebuild_csr_for_keep(self, old_offsets, old_indices, keep_proxy):
        """ 从保留的 proxy 索引重建 CSR 映射。"""
        n_new = len(keep_proxy)
        if n_new == 0:
            return np.zeros(1, dtype=np.int64), np.empty(0, dtype=np.int32)
        counts = old_offsets[keep_proxy + 1] - old_offsets[keep_proxy]
        new_offsets = np.zeros(n_new + 1, dtype=np.int64)
        new_offsets[1:] = np.cumsum(counts, dtype=np.int64)
        parts = []
        for kp in keep_proxy:
            s = int(old_offsets[kp])
            e = int(old_offsets[kp + 1])
            parts.append(old_indices[s:e])
        new_indices = (np.concatenate(parts).astype(np.int32, copy=False)
                       if parts else np.empty(0, dtype=np.int32))
        return new_offsets, new_indices

    def denoise(self, method: str = "adaptive", voxel_size: float = 0.05,
                update_viewport: bool = True, **kwargs) -> Optional[Dict]:
        """
        去噪后重建 dataset，确保 proxy/ranges/CSR 始终同步。
        """
        vp = self.viewport
        if vp is None:
            print("PointCloudService: viewport 未注入，无法去噪", flush=True)
            return None

        name = self.resolve_processing_cloud()
        if not name:
            print("PointCloudService: 未找到可去噪的点云", flush=True)
            return None

        data = vp.get_cloud_data(name) if hasattr(vp, "get_cloud_data") else None
        if not data or "pos" not in data:
            print(f"PointCloudService: 点云数据不可用: {name}", flush=True)
            return None

        dataset_id = data.get("dataset_id")
        if not dataset_id:
            print(f"[PCFD] denoise.skip cloud={name} reason=no_dataset_id", flush=True)
            return None
        pts = np.asarray(data["pos"], dtype=np.float32)
        cols = None
        try:
            if "color" in data and data["color"] is not None:
                cols = np.asarray(data["color"], dtype=np.float32)
                if cols.size % 3 != 0 or len(cols.reshape(-1, 3)) != len(pts):
                    cols = None
                else:
                    cols = cols.reshape(-1, 3)
        except Exception:
            cols = None

        dataset = self.datasets.get(dataset_id) if dataset_id else None
        if dataset is None:
            print(f"[PCFD] denoise.skip cloud={name} dataset={dataset_id} reason=dataset_not_registered", flush=True)
            return None
        n_before = len(pts)
        keep_proxy = np.empty(0, dtype=np.int32)

        # ============================================================
        # 分层数据：adaptive_outlier_indices 返回 keep_proxy 索引
        # ============================================================
        if dataset is not None and dataset.index.has_source_mapping():
            meta = dataset.metadata or {}
            ranges = meta.get("ranges")
            if ranges is None:
                from algorithms.geometry import estimate_point_ranges
                scan_origins = meta.get("scan_origins")
                if scan_origins is not None:
                    ranges = estimate_point_ranges(pts, np.asarray(scan_origins, dtype=np.float64))
                else:
                    ranges = np.zeros(len(pts), dtype=np.float32)
            else:
                ranges = np.asarray(ranges, dtype=np.float32)

            # 长度校验
            if len(ranges) != len(pts):
                print(f"[PCFD] denoise.range_mismatch pts={len(pts)} ranges={len(ranges)}, "
                      f"skip denoise", flush=True)
                return None

            from algorithms.geometry import adaptive_outlier_indices
            keep_proxy = adaptive_outlier_indices(
                pts, ranges,
                std_ratio=float(kwargs.get("std_ratio", 2.5)),
                n_shells=int(kwargs.get("n_shells", 8)),
            )
            print(f"[PCFD] denoise.adaptive proxy={n_before} keep={len(keep_proxy)} "
                  f"removed={n_before - len(keep_proxy)}", flush=True)

        # ============================================================
        # 标准数据：走传统 o3d 路径
        # ============================================================
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            if cols is not None and len(cols) == len(pts):
                pcd.colors = o3d.utility.Vector3dVector(cols)

            if method == "adaptive" and dataset is not None:
                scan_origins = kwargs.get("scan_origin") or (dataset.metadata or {}).get("scan_origins")
                if scan_origins is not None:
                    kwargs["scan_origin"] = np.asarray(scan_origins, dtype=np.float64)

            clean = denoise(pcd, voxel_size=voxel_size, method=str(method), **kwargs)
            new_pts = np.asarray(clean.points, dtype=np.float32)

            if dataset is not None:
                keep_proxy = dataset.index.proxy_ids_for_points(new_pts)
                valid = (keep_proxy >= 0) & (keep_proxy < len(dataset.index.proxy_points))
                keep_proxy = keep_proxy[valid]
            else:
                keep_proxy = np.empty(0, dtype=np.int32)
                if len(new_pts) > 0:
                    pcd_orig = o3d.geometry.PointCloud()
                    pcd_orig.points = o3d.utility.Vector3dVector(pts)
                    kdtree = o3d.geometry.KDTreeFlann(pcd_orig)
                    keep_mask = np.zeros(len(pts), dtype=bool)
                    for i, pt in enumerate(new_pts):
                        k, idx, dists = kdtree.search_knn_vector_3d(pt, 1)
                        if k > 0 and dists[0] < 1e-6:
                            keep_mask[idx[0]] = True
                    keep_proxy = np.flatnonzero(keep_mask).astype(np.int32)

        n_after = len(keep_proxy)
        new_pts = pts[keep_proxy] if n_after > 0 else np.empty((0, 3), dtype=np.float32)
        new_cols = cols[keep_proxy] if (cols is not None and n_after > 0) else None

        raw_ids = None
        raw_count = 0

        if dataset is not None and n_after > 0:
            meta = dataset.metadata or {}

            if dataset.index.has_source_mapping():
                # 重建 ranges
                old_ranges = np.asarray(meta.get("ranges", []), dtype=np.float32)
                new_ranges = old_ranges[keep_proxy] if len(old_ranges) == n_before else np.zeros(n_after, dtype=np.float32)

                # 重建 CSR 映射
                old_offsets = dataset.index.source_raw_offsets
                old_indices = dataset.index.source_raw_indices
                new_offsets, new_indices = self._rebuild_csr_for_keep(
                    old_offsets, old_indices, keep_proxy)

                # 重建 metadata
                new_meta = dict(meta)
                new_meta.update({
                    'ranges': new_ranges.tolist(),
                    'proxy_source_offsets': new_offsets.tolist(),
                    'proxy_source_indices': new_indices.tolist(),
                    'denoise_history': new_meta.get('denoise_history', []) + [{
                        'before': n_before,
                        'after': n_after,
                        'method': method,
                    }],
                })
                # 保留 elevations（如果存在）
                if 'elevations' in meta:
                    old_elev = np.asarray(meta['elevations'], dtype=np.float32)
                    if len(old_elev) == n_before:
                        new_meta['elevations'] = old_elev[keep_proxy].tolist()

                # 重新注册 dataset
                dataset = self.register_dataset(dataset_id, new_pts, new_cols, metadata=new_meta)

                # 计算 raw_ids
                new_proxy_ids = np.arange(n_after, dtype=np.int32)
                raw_ids = dataset.index.proxy_to_source_ids(new_proxy_ids, deduplicate=True)
                raw_count = len(raw_ids)

                # 更新 viewport 的 dataset_id 引用
                data["dataset_id"] = dataset_id
                print(f"[PCFD] denoise.dataset_rebuilt proxy={n_after} "
                      f"source_raw={len(new_indices)}", flush=True)

            else:
                # 标准数据：只更新 raw_count
                raw_count = dataset.index.raw_count_for_proxy(keep_proxy)

        if not update_viewport:
            return {
                "name": name, "method": method, "voxel_size": float(voxel_size),
                "points_before": n_before, "points_after": n_after,
                "dataset_id": dataset_id, "raw_ids": raw_ids,
                "raw_count": int(raw_count),
                "station_id": (data.get("station_id") or
                                (dataset.metadata or {}).get("station_id")),
                "proxy_points": new_pts, "proxy_colors": new_cols,
                "proxy_keep_indices": keep_proxy,
                "proxy_base_count": int(n_before),
                "proxy_source_offsets": (dataset.metadata or {}).get('proxy_source_offsets'),
                "proxy_source_indices": (dataset.metadata or {}).get('proxy_source_indices'),
                "ranges": (dataset.metadata or {}).get('ranges'),
            }

        # 更新视口
        if dataset is not None and n_after > 0:
            data["proxy_ids"] = np.arange(n_after, dtype=np.int32)
            data["domain"] = "proxy"
            data["index_space"] = "proxy_global"
            self.map_proxy_decision(dataset_id, np.arange(n_after, dtype=np.int32),
                                    source="denoise", expand_raw=False)

        if hasattr(vp, "queue_update_cloud_points"):
            vp.queue_update_cloud_points(name, new_pts, new_cols)
        elif hasattr(vp, "update_cloud_points"):
            vp.update_cloud_points(name, new_pts, new_cols)
        elif hasattr(vp, "add_cloud"):
            vp.add_cloud(name, new_pts, new_cols)

        stats = {
            "name": name, "method": method,
            "points_before": n_before, "points_after": n_after,
            "dataset_id": dataset_id, "raw_ids": raw_ids,
            "raw_count": int(raw_count),
        }
        print(
            f"PointCloudService: 去噪完成: {name}, "
            f"点数 {stats['points_before']} -> {stats['points_after']} ({method}), "
            f"proxy保留={n_after}, raw子集点数={raw_count}",
            flush=True,
        )
        return stats