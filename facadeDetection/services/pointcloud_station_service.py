from pathlib import Path
import numpy as np
import open3d as o3d
from algorithms.registration import (
    point_to_plane_icp, manual_seeded_icp, build_registration_cloud,
    audit_exported_global_transform, estimate_xy_initial_transform,
)
from services import proxy_cache
from services.dal.file_repo import FileRepo
from services.dal.pointcloud_station_repo import PointCloudStationRepo
from utils.logging_utils import log_event
from algorithms.geometry import (
    build_proxy_domain, stratified_proxy_build, estimate_elevation_angles,
)
from utils.dist_reader import read_dist
from utils.ply_fast_reader import read_ply_fast
from config.storage import Storage
import uuid

def _safe_int(value, default: int = 0) -> int:
    """Convert value to int, returning default if value is None or invalid."""
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_denoise_state(state):
    """将持久化的去噪状态安全归一化，兼容旧字段与 None 污染。

    返回的新 dict 中所有数值字段均为安全 Python int；数组字段保持
    ndarray（sidecar 直读零转换），旧 JSON list 仅在调用方需要时
    就地 np.asarray，绝不在这里做全量 Python 级过滤。
    """
    if not state:
        return {}
    out = {}
    # 标量字段安全提取
    for key in ('enabled', 'proxy_count', 'proxy_base_count'):
        raw = state.get(key)
        if key == 'enabled':
            out[key] = bool(raw) if raw is not None else False
        else:
            out[key] = _safe_int(raw, 0)
    # 数组字段：ndarray 直通；旧 JSON list 原样保留（调用方 asarray）
    for key in ('proxy_source_offsets', 'proxy_source_indices', 'ranges',
                'keep_proxy_indices', 'source_offsets', 'source_indices',
                'proxy_keep_indices'):
        raw = state.get(key)
        if isinstance(raw, np.ndarray):
            out[key] = raw
        elif isinstance(raw, list):
            cleaned = [x for x in raw if x is not None]
            out[key] = cleaned
        else:
            out[key] = []
    return out


class PointCloudStationService:
    def __init__(self, render_service, project_uuid=None, pointcloud_service=None):
        self.render = render_service
        self.pointcloud = pointcloud_service
        self.project_uuid = project_uuid
        self._restoring = False
        self._loaded_project = None
        self._dataset_ids = {}
        self._active_station_id = None
        # Registration snapshots are intentionally separate from raw assets.
        # They are reused across retries and never trigger another PLY read.
        self._registration_cloud_cache = {}
        self._station_fingerprints = {}

    def set_project(self, project_uuid):
        self.project_uuid = project_uuid
        self._loaded_project = None
        self._dataset_ids.clear()
        self._active_station_id = None
        self._registration_cloud_cache.clear()
        self._station_fingerprints.clear()

    def refresh(self):
        if self.project_uuid:
            PointCloudStationRepo.sync_assets(self.project_uuid)

    def list_stations(self):
        return PointCloudStationRepo.list(self.project_uuid) if self.project_uuid else []

    def set_selected(self, station_id, selected):
        PointCloudStationRepo.set_selected(self.project_uuid, station_id, selected)

    @staticmethod
    def _rebuild_csr_for_keep(old_offsets, old_indices, keep_proxy):
        """从保留的 proxy 索引重建 CSR 映射。"""
        n_new = len(keep_proxy)
        if n_new == 0:
            return np.zeros(1, dtype=np.int64), np.empty(0, dtype=np.int32)
        counts = old_offsets[keep_proxy + 1] - old_offsets[keep_proxy]
        new_offsets = np.zeros(n_new + 1, dtype=np.int64)
        new_offsets[1:] = np.cumsum(counts, dtype=np.int64)
        total = int(new_offsets[-1])
        if total == 0:
            return new_offsets, np.empty(0, dtype=np.int32)
        offsets_in_group = np.arange(total, dtype=np.int64) - np.repeat(
            new_offsets[:-1], counts)
        gather = np.repeat(old_offsets[keep_proxy], counts) + offsets_in_group
        new_indices = old_indices[gather].astype(np.int32, copy=False)
        return new_offsets, new_indices

    def delete_selected(self):
        rows = [x for x in self.list_stations() if x.is_selected]
        if not rows:
            raise ValueError('请至少选择一个站点')
        result_paths = {str(x.registered_path) for x in rows if x.registered_path}
        self.render.clear_scene_display()
        if self.pointcloud is not None:
            for row in rows:
                self.pointcloud.release_station_domain(row.id)
        # 删除站点前同步删除关联的 FileAsset 和缓存
        for row in rows:
            if row.file_asset_id is not None:
                FileRepo.delete_file(self.project_uuid, row.file_asset_id,
                                     delete_disk=False)
            proxy_cache.delete_station_cache(self.project_uuid, row.id)
        PointCloudStationRepo.delete(self.project_uuid, [x.id for x in rows])
        for path in result_paths:
            result = Path(path)
            # 配准结果在一次操作中由所有参与站点共享；
            # 仅当没有任何剩余站点引用它时才删除该结果文件。
            still_referenced = any(
                str(x.registered_path) == path for x in self.list_stations())
            if (not still_referenced and result.name.startswith('registration_')
                    and result.exists()):
                try:
                    result.unlink()
                except OSError:
                    pass
        # 已删除的活动站点不能保留在持久化视图中。
        remaining = self.list_stations()
        if not remaining:
            self.render.clear_scene_display()
            PointCloudStationRepo.save_view(self.project_uuid, 'single', None, [])
        else:
            # 删除操作已经清空场景；立即渲染回退站点，
            fallback = remaining[0]
            PointCloudStationRepo.save_view(
                self.project_uuid, 'single', fallback.id,
                [x.id for x in remaining if x.is_selected] or [fallback.id])
            self._loaded_project = None
            self.show_single(fallback)
        log_event(self.project_uuid, 'station.delete', count=len(rows))

    def _load(self, path):
        """Read an already-globalized PLY; never apply transformToGlobal here."""
        # 二进制 PLY 走 memmap 免解析快读；不满足快读条件时回退 Open3D，
        # 两条路径对同一资产逐点一致（uchar rgb /255 归一化语义相同）。
        if Path(path).suffix.lower() == '.e57':
            raise RuntimeError('站点运行时只允许读取 cache PLY；请重新导入原始 E57')
        fast = read_ply_fast(path)
        if fast is not None:
            return fast
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points, dtype=np.float64)
        colors = np.asarray(cloud.colors, dtype=np.float64) if cloud.has_colors() else None
        return points, colors

    @staticmethod
    def _global_coordinate_metadata(path):
        """Audit the export matrix without transforming runtime point data."""
        ply = Path(path)
        # FLS 输出：.json sidecar；E57 cache：.json sidecar
        candidates = (
            ply.with_suffix('.json'),
            ply.parent / 'pointclouds' / f'{ply.stem}.json',
            ply.with_suffix('.json'),
        )
        json_path = next((item for item in candidates if item.exists()), None)
        metadata = {'coordinate_frame': 'global',
                    'transform_applied': True,
                    'transform_applied_at': 'fls_export'}
        if json_path is not None:
            audit = audit_exported_global_transform(json_path)
            metadata['transform_json_path'] = str(audit.json_path)
            metadata['transform_to_global'] = audit.matrix.tolist()
            # 读取统一 pose schema（E57 / FLS 新接口兼容）
            try:
                import json
                with open(json_path, 'r', encoding='utf-8') as f:
                    sidecar = json.load(f)
                # E57 cache sidecar: scan_poses / scan_origins
                if 'scan_poses' in sidecar:
                    metadata['scan_poses'] = sidecar['scan_poses']
                if 'scan_origins' in sidecar:
                    metadata['scan_origins'] = sidecar['scan_origins']
                # FLS JSON sidecar: rotationMatrix / scanPosition
                if 'rotationMatrix' in sidecar:
                    metadata['rotation_matrix'] = sidecar['rotationMatrix']
                if 'scanPosition' in sidecar:
                    metadata['scan_position'] = sidecar['scanPosition']
                    if 'scan_origins' not in metadata:
                        metadata['scan_origins'] = [sidecar['scanPosition']]
            except Exception:
                pass
        return metadata

    def _load_proxy_domain(self, station):
        """加载站点的两个域并返回用于显示的代理数据。

        原始数组被注册为精密度域；只有数据集返回的代理允许穿过视口边界。
        """
        if self.pointcloud is None:
            raise RuntimeError('PointCloudService 未注入，无法建立站点索引')
        dataset_id = f'{self.project_uuid}:{station.id}'
        fingerprint = PointCloudStationRepo.get_asset_fingerprint(
            self.project_uuid, station.id)
        # The asset fingerprint, not the filename, determines whether arrays
        # can be reused.  A changed source must never reuse an old station.
        fingerprint_key = tuple(fingerprint or (str(station.source_path), None, None))
        existing = self.pointcloud.get_dataset(dataset_id)
        if existing is None:
            legacy_id = f'{self.project_uuid}:{Path(station.source_path).name}'
            existing = self.pointcloud.get_dataset(legacy_id)
            if existing is not None:
                self.pointcloud.datasets.pop(legacy_id, None)
                existing.dataset_id = dataset_id
                self.pointcloud.datasets[dataset_id] = existing
                # 上传链已按当前资产完成读取/代理构建，继承指纹后走复用分支，
                # 避免同一文件被完整处理第二遍
                existing.metadata = dict(existing.metadata or {},
                                         station_id=station.id,
                                         asset_fingerprint=list(fingerprint_key))
                existing.metadata.update(
                    self._global_coordinate_metadata(station.source_path))
                self._station_fingerprints[station.id] = fingerprint_key
        if existing is not None and self._station_fingerprints.get(station.id) == fingerprint_key:
            state = _normalize_denoise_state(PointCloudStationRepo.get_denoise_state(
                self.project_uuid, station.id))
            if not (state and state.get('enabled') and
                    not (existing.metadata or {}).get('denoise_restored')):
                self._dataset_ids[station.id] = dataset_id
                return existing
            self.pointcloud.datasets.pop(dataset_id, None)
            self.pointcloud.release_station_domain(station.id)
        cached_raw = proxy_cache.load_raw_cache(
            self.project_uuid, station.id, fingerprint_key)
        if cached_raw is not None:
            # 与 self._load 等价：同一资产的 PLY 解包结果逐点一致。
            # 命中时是 .npy 的只读 memmap，RSS 只计实际触碰的页。
            points, colors = cached_raw
            print(f'[PCFD] raw_cache.restored station={station.id} '
                  f'raw={len(points)}', flush=True)
        else:
            points, colors = self._load(station.source_path)
            # 解包结果对同一资产是确定的，落盘后重开项目跳过 Open3D 读盘
            proxy_cache.save_raw_cache(
                self.project_uuid, station.id, fingerprint_key,
                points=points, colors=colors)
        source_id = f'{dataset_id}:source'
        state = _normalize_denoise_state(PointCloudStationRepo.get_denoise_state(
            self.project_uuid, station.id))
        dist_path = Path(station.source_path).with_suffix('.dist')
        metadata = {'source_id': source_id, 'station_id': station.id,
                    'project_uuid': self.project_uuid,
                    'asset_fingerprint': list(fingerprint_key),
                    'source_raw_count': int(len(points))}
        metadata.update(self._global_coordinate_metadata(station.source_path))
        # 兼容旧项目：source_offsets / source_indices 别名
        # （sidecar 读出即为 ndarray，不能用 or 链做真值判断）
        state_payload = state or {}

        def _pick_array(*keys):
            for key in keys:
                value = state_payload.get(key)
                if value is not None and len(value):
                    return value
            return []

        state_offsets = np.asarray(
            _pick_array('proxy_source_offsets', 'source_offsets'),
            dtype=np.int64)
        state_indices = np.asarray(
            _pick_array('proxy_source_indices', 'source_indices'),
            dtype=np.int64)
        restored_direct = bool(
            (state or {}).get('enabled') and len(state_offsets) >= 2 and
            len(state_offsets) == int((state or {}).get('proxy_count') or 0) + 1 and
            len(state_indices) == int(state_offsets[-1]) and
            # 比较等长切片。
            np.all((state_offsets[:-1] >= 0) &
                   (state_offsets[1:] >= state_offsets[:-1])) and
            np.all((state_indices >= 0) & (state_indices < len(points))) and
            np.all(np.diff(state_offsets) > 0))
        # 代理缓存与 dist 文件独立：只要指纹匹配即可加载。
        # dist 文件仅用于首次重建，后续重载完全依赖缓存的代理数组本体。
        cached_proxy = None
        if not restored_direct:
            cached_proxy = proxy_cache.load_proxy_cache(
                self.project_uuid, station.id, fingerprint_key,
                source_count=len(points))
        if restored_direct:
            representative_ids = state_indices[state_offsets[:-1]]
            proxy = points[representative_ids]
            proxy_colors = colors[representative_ids] if colors is not None else None
            # 运行期 metadata 的 CSR/ranges 一律以 ndarray 传递：
            # 千万级 indices 的 .tolist() 会产生 GB 级 Python int 临时对象，
            # pymalloc 不会把这些内存还给 OS。sidecar 读出即为 ndarray。
            state_ranges = (state or {}).get('ranges')
            has_ranges = (state_ranges is not None
                          and len(state_ranges) > 0)
            metadata.update({
                'proxy_source_offsets': state_offsets,
                'proxy_source_indices': state_indices.astype(np.int32, copy=False),
                'ranges': (np.asarray(state_ranges, dtype=np.float32)
                           if has_ranges else
                           np.zeros(len(proxy), dtype=np.float32)),
                'denoise_restored': True,
            })
            print(f'[PCFD] denoise.restore station={station.id} '
                  f'proxy={len(proxy)} raw={len(points)}', flush=True)
        elif cached_proxy is not None:
            # 与 dist 重建分支等价：优先使用缓存的代理数组本体，完全不触碰
            # raw memmap（代表点散布全文件，fancy 采集会把整个映射换页）；
            # 无 proxy 数组字段的旧缓存回退按持久化代表行采集。
            proxy = cached_proxy.get('proxy_points')
            proxy_colors = cached_proxy.get('proxy_colors')
            if proxy is None:
                representative_ids = cached_proxy['representative_ids']
                proxy = points[representative_ids]
                proxy_colors = colors[representative_ids] if colors is not None else None
            metadata.update({
                'proxy_source_offsets': cached_proxy['offsets'],
                'proxy_source_indices': cached_proxy['indices'],
                'ranges': cached_proxy['ranges'],
                'scan_origins': cached_proxy['scan_origins'].tolist(),
                'distance_source': cached_proxy['distance_source'],
                'proxy_cache': 'restored',
            })
            print(f'[PCFD] proxy_cache.restored station={station.id} '
                  f'proxy={len(proxy)} raw={len(points)}', flush=True)
        else:
            # 三条链路（FLS 转换 PLY / E57 转换 PLY / 原生 PLY 上传）同一个代理域构建器。
            dist = read_dist(dist_path, points, metadata)
            if not len(dist.scan_origins):
                # 快照里的测站可作为补充来源；仍无则归一化分支已生效。
                saved_origins = (state or {}).get('scan_origins')
                if saved_origins:
                    dist.scan_origins = np.asarray(
                        saved_origins, dtype=np.float32).reshape(-1, 3)
            elevations = (estimate_elevation_angles(points, dist.scan_origins)
                          if len(dist.scan_origins) else None)
            built = build_proxy_domain(
                points, colors, dist.ranges_m,
                scan_origin=(dist.scan_origins if len(dist.scan_origins) else None),
                elevations=elevations,
                distance_source=dist.source)
            proxy = built['proxy_points']
            proxy_colors = built['proxy_colors']
            offsets = built['proxy_source_offsets']
            indices = built['proxy_source_indices']
            ranges = built['ranges']
            representatives = built['representatives']
            metadata.update({
                'proxy_source_offsets': offsets,
                'proxy_source_indices': indices,
                'ranges': ranges,
                'scan_origins': dist.scan_origins.tolist(),
                'distance_source': dist.source,
                'density_source': dist.density_source,
                'distance_warnings': dist.warnings,
            })
            # 重建结果对同一资产是确定的，落盘后重开项目可直接命中缓存；
            # 代理数组本体一并持久化，重开轮不再从 raw memmap 采集
            proxy_cache.save_proxy_cache(
                self.project_uuid, station.id, fingerprint_key,
                offsets=offsets, indices=indices, ranges=ranges,
                scan_origins=dist.scan_origins, distance_source=dist.source,
                representative_ids=representatives,
                proxy_points=proxy, proxy_colors=proxy_colors)
            print(f'[PCFD] proxy.built station={station.id} '
                  f'proxy={len(proxy)} raw={len(points)} '
                  f'density={dist.density_source}', flush=True)
        self.pointcloud.register_source_asset(source_id, points, colors,
                                              {'ply_path': station.source_path})
        dataset = self.pointcloud.register_dataset(dataset_id, proxy, proxy_colors,
                                                   metadata=metadata)
        self._station_fingerprints[station.id] = fingerprint_key
        # 从持久化的索引中重建去噪代理。不需要派生点云文件
        if state and state.get('enabled') and not restored_direct:
            # 兼容旧项目：proxy_keep_indices / keep_proxy_indices 别名
            keep = np.asarray(
                state.get('keep_proxy_indices', state.get('proxy_keep_indices', [])),
                dtype=np.int64)
            base_count = int(state.get('proxy_base_count') or len(proxy))
            saved_count = int(state.get('proxy_count') or len(keep))
            valid_keep = (len(keep) == saved_count and
                          len(np.unique(keep)) == len(keep) and
                          np.all((keep >= 0) & (keep < len(proxy))))

            # 后续的去噪运行存储的是相对于上一次去噪后代理的索引。
            # 如果 base_count 与当前 proxy 长度不一致，尝试通过 CSR 映射翻译。
            if base_count != len(proxy):
                translated = self._translate_denoise_keep_to_base(
                    proxy, metadata, state)
                if translated is not None:
                    keep = translated
                    valid_keep = True
                    base_count = len(proxy)
                    print(f'[PCFD] denoise.translated station={station.id} '
                          f'keep={len(keep)} proxy={len(proxy)}', flush=True)

            # 严格校验：去噪索引必须匹配当前代理基数
            if base_count == len(proxy) and valid_keep:
                new_meta = dict(dataset.metadata or {})
                # 去噪快照来自 JSON（list）；运行期统一转回 ndarray
                # 兼容旧项目：source_offsets / source_indices 别名
                for key, dtype, aliases in (
                        ('proxy_source_offsets', np.int64, ('source_offsets',)),
                        ('proxy_source_indices', np.int32, ('source_indices',)),
                        ('ranges', np.float32, ())):
                    value = state.get(key)
                    if value is None:
                        value = next((state.get(alias) for alias in aliases
                                      if state.get(alias) is not None), None)
                    if value is not None:
                        new_meta[key] = np.asarray(value, dtype=dtype)

                # 去噪后需要重建 CSR：keep 索引相对于当前 proxy，
                # 但 source_indices 仍指向原始 raw 点，只需按 keep 重新切片。
                old_offsets = dataset.index.source_raw_offsets
                old_indices = dataset.index.source_raw_indices
                if (old_offsets is not None and old_indices is not None and
                        len(old_offsets) == len(proxy) + 1):
                    new_offsets, new_indices = self._rebuild_csr_for_keep(
                        old_offsets, old_indices, keep)
                    new_meta['proxy_source_offsets'] = new_offsets
                    new_meta['proxy_source_indices'] = new_indices
                    if 'ranges' in new_meta:
                        old_ranges = np.asarray(new_meta['ranges'], dtype=np.float32)
                        if len(old_ranges) == len(proxy):
                            new_meta['ranges'] = old_ranges[keep]
                    new_meta['denoise_restored'] = True

                dataset = self.pointcloud.register_dataset(
                    dataset_id, proxy[keep],
                    proxy_colors[keep] if proxy_colors is not None else None,
                    metadata=new_meta)
                print(f'[PCFD] denoise.restored station={station.id} '
                      f'proxy={len(keep)} raw={len(points)}', flush=True)
            else:
                # 去噪快照无法重放时：兜底加载原始 PLY（不报错），
                # 但保留去噪元数据提示用户重新去噪。
                print(f'[PCFD] denoise.restore_fallback station={station.id} '
                      f'base={len(proxy)} expected={base_count} '
                      f'keep={len(keep)} saved={saved_count}', flush=True)
                new_meta = dict(dataset.metadata or {})
                new_meta['denoise_fallback'] = True
                new_meta['denoise_fallback_reason'] = (
                    f'base_count={base_count} proxy={len(proxy)} '
                    f'keep={len(keep)} saved={saved_count}')
                n_alive = len(proxy)
                alive_offsets = new_meta.get('proxy_source_offsets')
                alive_indices = new_meta.get('proxy_source_indices')
                if (alive_offsets is None or alive_indices is None or
                        len(np.asarray(alive_offsets)) != n_alive + 1 or
                        len(np.asarray(alive_indices)) !=
                        int(np.asarray(alive_offsets)[-1] if n_alive else 0)):
                    new_meta['proxy_source_offsets'] = np.arange(
                        n_alive + 1, dtype=np.int64)
                    new_meta['proxy_source_indices'] = np.arange(
                        n_alive, dtype=np.int32)
                new_meta['ranges'] = np.asarray(
                    new_meta.get('ranges', np.zeros(n_alive, dtype=np.float32)),
                    dtype=np.float32)
                if len(new_meta['ranges']) != n_alive:
                    new_meta['ranges'] = np.zeros(n_alive, dtype=np.float32)
                new_meta['identity_mapping'] = True
                dataset = self.pointcloud.register_dataset(
                    dataset_id, proxy,
                    proxy_colors if proxy_colors is not None else None,
                    metadata=new_meta)
        # 检测估计的代理法向对同一资产是确定的；随缓存恢复挂到 dataset，
        # 下次检测/重开直接复用，不再重复估计。去噪子集长度与缓存代理数
        # 不一致时跳过，由检测按现行逻辑估计。
        if cached_proxy is not None:
            cached_normals = cached_proxy.get('proxy_normals')
            if (cached_normals is not None and
                    len(cached_normals) == len(dataset.proxy_points)):
                dataset.proxy_normals = cached_normals
        return dataset

    @staticmethod
    def _translate_denoise_keep_to_base(base_proxy, base_metadata, state):
        """将链接去噪快照转换回源代理行索引。"""
        base_offsets = np.asarray(base_metadata.get('proxy_source_offsets', []),
                                  dtype=np.int64)
        base_indices = np.asarray(base_metadata.get('proxy_source_indices', []),
                                  dtype=np.int64)
        final_offsets = np.asarray(state.get('proxy_source_offsets', []),
                                   dtype=np.int64)
        final_indices = np.asarray(state.get('proxy_source_indices', []),
                                   dtype=np.int64)
        keep = np.asarray(state.get('keep_proxy_indices', []), dtype=np.int64)
        if (len(base_offsets) != len(base_proxy) + 1 or
                len(final_offsets) != len(keep) + 1 or
                len(base_indices) != int(base_offsets[-1]) or
                len(final_indices) != int(final_offsets[-1]) or
                len(keep) == 0):
            return None

        anchors = base_indices[base_offsets[:-1]]
        order = np.argsort(anchors, kind='mergesort')
        sorted_anchors = anchors[order]
        final_starts = final_offsets[:-1]
        final_ends = final_offsets[1:]
        if np.any(final_ends <= final_starts):
            return None
        final_anchors = final_indices[final_starts]
        positions = np.searchsorted(sorted_anchors, final_anchors, side='left')
        if np.any(positions >= len(sorted_anchors)):
            return None
        if np.any(sorted_anchors[positions] != final_anchors):
            return None
        translated = order[positions]
        base_lengths = base_offsets[translated + 1] - base_offsets[translated]
        if np.any(base_lengths != final_ends - final_starts):
            return None
        if len(np.unique(translated)) != len(translated):
            return None
        return translated

    def prepare_show_single(self, station):
        """计算段：站点域准备（PLY 读取/代理重建/缓存采集），不触碰渲染。

        返回注册的 dataset；站点资产无效时返回 None。可在后台线程执行。
        """
        if station.last_error:
            return None
        return self._load_proxy_domain(station)

    def commit_show_single(self, station, dataset):
        """提交段：渲染提交 + 活动站点与视图持久化，必须在 GUI 线程执行。"""
        if station.last_error or dataset is None:
            self.render.clear_scene_display()
            log_event(self.project_uuid, 'asset.invalid', station_id=station.id, reason=station.last_error)
            return
        self.render.clear_scene_display()
        self.render.show_station_proxy(
            station.id, station.display_name, dataset.proxy_points,
            dataset.proxy_colors, dataset_id=dataset.dataset_id)
        self._active_station_id = station.id
        selected = [x.id for x in self.list_stations() if x.is_selected]
        PointCloudStationRepo.save_view(self.project_uuid, 'single', station.id, selected)
        log_event(self.project_uuid, 'station.switch', station_id=station.id)

    def show_single(self, station):
        """切换至单个站点视图。若目标已是活动站点，直接返回避免重复渲染。"""
        if self._active_station_id is not None and int(self._active_station_id) == int(station.id):
            return
        self.commit_show_single(station, self.prepare_show_single(station))

    def merge_selected(self):
        rows = [x for x in self.list_stations() if x.is_selected and not x.last_error]
        if not rows: raise ValueError('请至少选择一个 PLY 站点')
        registered_paths = {str(x.registered_path) for x in rows if x.registered_path}
        if (len(registered_paths) == 1 and all(x.registered_path for x in rows)
                and all(x.registration_status == 'success' for x in rows)):
            result_path = Path(next(iter(registered_paths)))
            if result_path.exists():
                cloud = o3d.io.read_point_cloud(str(result_path))
                self.render.clear_scene_display()
                self.render.show_result_cloud(
                    'pcfd.result.registration', np.asarray(cloud.points),
                    np.asarray(cloud.colors) if cloud.has_colors() else None)
                PointCloudStationRepo.save_view(
                    self.project_uuid, 'registered_merge', rows[0].id,
                    [x.id for x in rows])
                log_event(self.project_uuid, 'station.merge',
                          mode='registered_merge', count=len(rows))
                return
        self.render.clear_scene_display()
        for row in rows:
            dataset = self._load_proxy_domain(row)
            self.render.show_station_proxy(row.id, row.display_name,
                                           dataset.proxy_points, dataset.proxy_colors,
                                           dataset_id=dataset.dataset_id)
        # 部分或混合配准集不是有效的结果快照。
        mode = 'registered_merge' if (
            len(registered_paths) == 1 and all(x.registered_path for x in rows)
            and all(x.registration_status == 'success' for x in rows)
        ) else 'raw_overlay'
        PointCloudStationRepo.save_view(self.project_uuid, mode, rows[0].id, [x.id for x in rows])
        log_event(self.project_uuid, 'station.merge', mode=mode, count=len(rows))

    def prepare_registration_view(self, rows=None):
        """Publish only proxy clouds for operator correspondence picking."""
        rows = rows or [x for x in self.list_stations() if x.is_selected]
        if len(rows) != 2:
            raise ValueError('人工点配准当前需要恰好选择两个站点')
        datasets = [self._load_proxy_domain(row) for row in rows]
        self.render.clear_scene_display()
        names = []
        for row, dataset in zip(rows, datasets):
            names.append(self.render.show_station_proxy(
                row.id, row.display_name, dataset.proxy_points,
                dataset.proxy_colors, dataset_id=dataset.dataset_id))
        return rows, names

    def prepare_restore_view(self):
        """计算段：恢复视图的数据准备（选中状态持久化 + 首站代理域加载）。

        返回 (station, dataset) 供 commit_restore_view 在 GUI 线程提交；
        无站点时为 (None, None)；已恢复过（或恢复进行中）返回 None。
        全程不触碰渲染，可在后台线程执行。
        """
        if self._restoring or self._loaded_project == self.project_uuid:
            return None
        self._restoring = True
        try:
            rows = self.list_stations()
            if not rows:
                return (None, None)
            state = PointCloudStationRepo.get_view(self.project_uuid)
            by_id = {row.id: row for row in rows}
            selected_ids = [sid for sid in (state.selected_station_ids if state else []) if sid in by_id]
            if not selected_ids:
                selected_ids = [row.id for row in rows if row.is_selected]
            if not selected_ids:
                selected_ids = [rows[0].id]
            for row in rows:
                PointCloudStationRepo.set_selected(self.project_uuid, row.id, row.id in selected_ids)
            # Opening a project must have a deterministic, low-cost initial view.
            # Persisted merge/active state remains available through explicit user
            # actions, but never hides the first station on project activation.
            first = rows[0]
            return first, self.prepare_show_single(first)
        finally:
            self._restoring = False

    def commit_restore_view(self, prepared):
        """提交段：渲染提交与恢复标记，必须在 GUI 线程执行。"""
        if prepared is None:
            return
        station, dataset = prepared
        if station is None:
            self.render.clear_scene_display()
        else:
            self.commit_show_single(station, dataset)
        self._loaded_project = self.project_uuid

    def register_selected(self, update_viewport=True, manual_points=None,
                          proxy_clouds=None):
        rows = [x for x in self.list_stations() if x.is_selected]
        if len(rows) < 2: raise ValueError('点云配准至少需要选择两个 PLY 站点')
        if manual_points is not None and len(rows) != 2:
            raise ValueError('人工对应点配准当前只支持两个站点')
        if manual_points is not None:
            src_pairs, tgt_pairs = manual_points
            src_pairs = np.asarray(src_pairs, dtype=np.float64).reshape(-1, 3)
            tgt_pairs = np.asarray(tgt_pairs, dtype=np.float64).reshape(-1, 3)
            if len(src_pairs) != len(tgt_pairs) or len(src_pairs) < 3:
                raise ValueError('人工对应点至少需要 3 对且数量必须一致')
            if not np.isfinite(src_pairs).all() or not np.isfinite(tgt_pairs).all():
                raise ValueError('人工对应点包含无效坐标')
            if proxy_clouds is not None:
                snapshots = tuple(np.asarray(x, dtype=np.float64).reshape(-1, 3)
                                  for x in proxy_clouds)
                if len(snapshots) != 2:
                    raise ValueError('人工点云快照数量必须为 2')
                # 用户界面从这些快照中提取物理坐标。
                for points, snapshot in zip((src_pairs, tgt_pairs),
                                             (snapshots[1], snapshots[0])):
                    if len(snapshot) == 0:
                        raise ValueError('人工点云快照为空')
                    # Do not allocate an NxM distance matrix for large proxy
                    # clouds; every selected point must be an exact snapshot
                    # member (the picker already returns snapshot coordinates).
                    tree = o3d.geometry.KDTreeFlann(
                        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(snapshot)))
                    distances = []
                    for point in points:
                        count, _, squared = tree.search_knn_vector_3d(point, 1)
                        distances.append(np.sqrt(squared[0]) if count else np.inf)
                    if np.any(np.asarray(distances) > 1e-3):
                        raise ValueError('人工对应点不属于当前站点快照，请重新选点')
        log_event(self.project_uuid, 'station.registration.started', count=len(rows))
        staged = []
        transformed_clouds = []
        registration_clouds = []
        for row in rows:
            dataset = self._load_proxy_domain(row)
            # 5 cm is the fine registration domain.  The algorithm builds its
            # own 20/10/5 cm pyramid, so this snapshot is not needlessly
            # downsampled twice at the coarse levels.
            voxel_size = 0.05
            key = (dataset.dataset_id, int(len(dataset.proxy_points)), voxel_size)
            reg = self._registration_cloud_cache.get(key)
            if reg is None:
                reg = build_registration_cloud(
                    dataset.proxy_points, dataset.proxy_colors, voxel_size=voxel_size)
                self._registration_cloud_cache[key] = reg
            if len(reg.points) < 3:
                raise ValueError(f'{row.display_name} 配准下采样点不足')
            registration_clouds.append(reg)

        reference = registration_clouds[0]
        reference_cloud = reference.as_open3d()
        transformed_clouds.append(reference_cloud)
        for row, moving_reg in zip(rows[1:], registration_clouds[1:]):
            # 输入点云已经过 GPS/全球坐标校正。ICP 只估计残余精化，不重新应用 JSON 变换。
            if manual_points is not None:
                src_pairs, tgt_pairs = manual_points
                if proxy_clouds is None:
                    raise ValueError('人工点配准缺少代理点云快照')
                result = manual_seeded_icp(
                    moving_reg.points, reference.points, src_pairs, tgt_pairs,
                    voxel_size=voxel_size, max_correspondence_distance=0.25,
                    pyramid_scales=(4.0, 2.0, 1.0), max_iteration=40)
            else:
                # GPS has already established the global frame.  This only
                # estimates the residual XY pose and is not a second GPS
                # transform application.
                initial = estimate_xy_initial_transform(
                    moving_reg.points, reference.points)
                result = point_to_plane_icp(
                    moving_reg.points, reference.points, voxel_size=voxel_size,
                    init=initial,
                    max_correspondence_distance=0.25,
                    pyramid_scales=(4.0, 2.0, 1.0), max_iteration=40)
            if not result.accepted:
                raise ValueError(f'{row.display_name} 配准失败：{result.message}，RMSE={result.inlier_rmse:.4f}')
            cloud = moving_reg.as_open3d()
            cloud.transform(result.transformation)
            transformed_clouds.append(cloud)
            staged.append((row, result))

        merged = o3d.geometry.PointCloud()
        for cloud in transformed_clouds:
            merged += cloud
        # The transform is estimated on the cached registration domain, but
        # do not quantize the published result a second time.  This preserves
        # the physical facade edge used for the final overlay while keeping
        # the expensive ICP work bounded by the proxy domain.
        merged = merged.voxel_down_sample(0.05)
        result_dir = Storage.ensure_project_dirs(self.project_uuid)['results']
        operation_id = uuid.uuid4().hex
        out = result_dir / f'registration_{operation_id}.ply'
        # 保留 .ply 后缀，以便 Open3D 为临时文件选择 PLY 写入器。
        tmp = result_dir / f'.registration_{operation_id}.tmp.ply'
        if not o3d.io.write_point_cloud(str(tmp), merged):
            raise IOError(f'配准结果写入失败：{tmp}')
        tmp.replace(out)

        payload = {
            'station_ids': [x.id for x in rows],
            'reference_id': rows[0].id,
            'result_path': str(out),
            'metric_domain': 'registration_downsample',
            'registration_voxel_size': 0.05,
            'registration_cloud_counts': [int(len(x.points)) for x in registration_clouds],
            'transforms': {
                rows[0].id: (np.eye(4).tolist(), 1.0, 0.0),
                **{row.id: (result.transformation.tolist(), result.fitness,
                            result.inlier_rmse) for row, result in staged},
            },
        }
        if update_viewport:
            self.commit_registration(payload)
        return payload

    def commit_registration(self, payload):
        """提交一次 ICP 配准事务并发布其合并视图。"""
        station_ids = [int(x) for x in payload['station_ids']]
        PointCloudStationRepo.update_registrations(
            self.project_uuid, station_ids, payload['transforms'],
            payload['result_path'])
        result_cloud = o3d.io.read_point_cloud(payload['result_path'])
        self.render.clear_scene_display()
        self.render.show_result_cloud(
            'pcfd.result.registration', np.asarray(result_cloud.points),
            np.asarray(result_cloud.colors) if result_cloud.has_colors() else None)
        PointCloudStationRepo.save_view(
            self.project_uuid, 'registered_merge', payload['reference_id'],
            station_ids, registration_id=None)
        log_event(self.project_uuid, 'station.registration.completed',
                  count=len(station_ids))