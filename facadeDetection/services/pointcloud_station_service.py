from pathlib import Path
import numpy as np
import open3d as o3d
from algorithms.registration import (
    point_to_plane_icp, manual_seeded_icp, build_registration_cloud,
    audit_exported_global_transform,
)
from services.dal.pointcloud_station_repo import PointCloudStationRepo
from utils.logging_utils import log_event
from algorithms.geometry import stratified_proxy_build, estimate_elevation_angles
from utils.dist_reader import read_dist
from config.storage import Storage
import uuid

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

    def set_project(self, project_uuid):
        self.project_uuid = project_uuid
        self._loaded_project = None
        self._dataset_ids.clear()
        self._active_station_id = None
        self._registration_cloud_cache.clear()

    def refresh(self):
        if self.project_uuid:
            PointCloudStationRepo.sync_assets(self.project_uuid)

    def list_stations(self):
        return PointCloudStationRepo.list(self.project_uuid) if self.project_uuid else []

    def set_selected(self, station_id, selected):
        PointCloudStationRepo.set_selected(self.project_uuid, station_id, selected)

    def delete_selected(self):
        rows = [x for x in self.list_stations() if x.is_selected]
        if not rows:
            raise ValueError('请至少选择一个站点')
        result_paths = {str(x.registered_path) for x in rows if x.registered_path}
        self.render.clear_scene_display()
        if self.pointcloud is not None:
            for row in rows:
                self.pointcloud.release_station_domain(row.id)
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
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points, dtype=np.float64)
        colors = np.asarray(cloud.colors, dtype=np.float64) if cloud.has_colors() else None
        return points, colors

    @staticmethod
    def _global_coordinate_metadata(path):
        """Audit the export matrix without transforming runtime point data."""
        ply = Path(path)
        candidates = (ply.with_suffix('.json'),
                      ply.parent / 'pointclouds' / f'{ply.stem}.json')
        json_path = next((item for item in candidates if item.exists()), None)
        metadata = {'coordinate_frame': 'global',
                    'transform_applied': True,
                    'transform_applied_at': 'fls_export'}
        if json_path is not None:
            audit = audit_exported_global_transform(json_path)
            metadata['transform_json_path'] = audit.json_path
            metadata['transform_to_global'] = audit.matrix.tolist()
        return metadata

    def _load_proxy_domain(self, station):
        """加载站点的两个域并返回用于显示的代理数据。

        原始数组被注册为精密度域；只有数据集返回的代理允许穿过视口边界。
        """
        if self.pointcloud is None:
            raise RuntimeError('PointCloudService 未注入，无法建立站点索引')
        dataset_id = f'{self.project_uuid}:{station.id}'
        existing = self.pointcloud.get_dataset(dataset_id)
        if existing is None:
            legacy_id = f'{self.project_uuid}:{Path(station.source_path).name}'
            existing = self.pointcloud.get_dataset(legacy_id)
            if existing is not None:
                self.pointcloud.datasets.pop(legacy_id, None)
                existing.dataset_id = dataset_id
                self.pointcloud.datasets[dataset_id] = existing
        if existing is not None:
            # A dataset can survive a view refresh while the persisted denoise
            # snapshot was written afterwards (or while an older runtime
            # dataset was restored before the station projection).  Returning
            # it unconditionally silently re-published the noisy proxy.  Only
            # reuse it when it already represents the persisted snapshot.
            state = PointCloudStationRepo.get_denoise_state(
                self.project_uuid, station.id)
            if not (state and state.get('enabled') and
                    not (existing.metadata or {}).get('denoise_restored')):
                self._dataset_ids[station.id] = dataset_id
                return existing
            self.pointcloud.datasets.pop(dataset_id, None)
        points, colors = self._load(station.source_path)
        source_id = f'{dataset_id}:source'
        state = PointCloudStationRepo.get_denoise_state(self.project_uuid, station.id)
        dist_path = Path(station.source_path).with_suffix('.dist')
        metadata = {'source_id': source_id, 'station_id': station.id,
                    'source_raw_count': int(len(points))}
        # FLS export has already applied transformToGlobal. Keep it as an
        # audit trail only; proxy and ICP inputs remain in PLY global coords.
        metadata.update(self._global_coordinate_metadata(station.source_path))
        state_offsets = np.asarray((state or {}).get('proxy_source_offsets', []),
                                   dtype=np.int64)
        state_indices = np.asarray((state or {}).get('proxy_source_indices', []),
                                   dtype=np.int64)
        restored_direct = bool(
            (state or {}).get('enabled') and len(state_offsets) >= 2 and
            len(state_offsets) == int((state or {}).get('proxy_count', 0)) + 1 and
            len(state_indices) == int(state_offsets[-1]) and
            # 比较等长切片。
            np.all((state_offsets[:-1] >= 0) &
                   (state_offsets[1:] >= state_offsets[:-1])) and
            np.all((state_indices >= 0) & (state_indices < len(points))) and
            np.all(np.diff(state_offsets) > 0))
        if restored_direct:
            representative_ids = state_indices[state_offsets[:-1]]
            proxy = points[representative_ids]
            proxy_colors = colors[representative_ids] if colors is not None else None
            metadata.update({
                'proxy_source_offsets': state_offsets.tolist(),
                'proxy_source_indices': state_indices.tolist(),
                'ranges': (state or {}).get('ranges') or [0.0] * len(proxy),
                'denoise_restored': True,
            })
            print(f'[PCFD] denoise.restore station={station.id} '
                  f'proxy={len(proxy)} raw={len(points)}', flush=True)
        elif dist_path.exists():
            dist = read_dist(dist_path, points, metadata)
            elevations = (estimate_elevation_angles(points, dist.scan_origins)
                          if len(dist.scan_origins) else None)
            proxy, proxy_colors, offsets, indices, ranges = stratified_proxy_build(
                points, colors, dist.ranges_m,
                scan_origin=dist.scan_origins if len(dist.scan_origins) else None,
                elevations=elevations)
            metadata.update({
                'proxy_source_offsets': offsets.tolist(),
                'proxy_source_indices': indices.tolist(),
                'ranges': ranges.tolist(),
                'scan_origins': dist.scan_origins.tolist(),
                'distance_source': dist.source,
                'distance_warnings': dist.warnings,
            })
        else:
            proxy, proxy_colors = points, colors
        self.pointcloud.register_source_asset(source_id, points, colors,
                                              {'ply_path': station.source_path})
        dataset = self.pointcloud.register_dataset(dataset_id, proxy, proxy_colors,
                                                   metadata=metadata)
        # 从持久化的索引中重建去噪代理。不需要派生点云文件
        if state and state.get('enabled') and not restored_direct:
            keep = np.asarray(state.get('keep_proxy_indices', []), dtype=np.int64)
            base_count = int(state.get('proxy_base_count', len(proxy)))
            saved_count = int(state.get('proxy_count', len(keep)))
            valid_keep = (len(keep) == saved_count and
                          len(np.unique(keep)) == len(keep) and
                          np.all((keep >= 0) & (keep < len(proxy))))
            # 后续的去噪运行存储的是相对于上一次去噪后代理的索引。
            if base_count != len(proxy):
                translated = self._translate_denoise_keep_to_base(
                    proxy, metadata, state)
                if translated is not None:
                    keep = translated
                    valid_keep = True
                    base_count = len(proxy)
            # 在代理基数计数被持久化之前创建的旧状态可能由逐点代理生成。
            legacy_direct = (valid_keep and base_count != len(proxy) and
                             not state.get('proxy_source_offsets') and
                             len(keep) == saved_count and
                             np.all((keep >= 0) & (keep < len(proxy))))
            if (base_count == len(proxy) and valid_keep):
                metadata = dict(dataset.metadata or {})
                for key in ('proxy_source_offsets', 'proxy_source_indices', 'ranges'):
                    if state.get(key) is not None:
                        metadata[key] = state[key]
                dataset = self.pointcloud.register_dataset(
                    dataset_id, proxy[keep],
                    proxy_colors[keep] if proxy_colors is not None else None,
                    metadata=metadata)
                print(f'[PCFD] denoise.restored station={station.id} '
                      f'proxy={len(keep)} raw={len(points)}', flush=True)
            elif legacy_direct:
                dataset = self.pointcloud.register_dataset(
                    dataset_id, proxy[keep],
                    proxy_colors[keep] if proxy_colors is not None else None,
                    metadata=dict(dataset.metadata or {}))
                print(f'[PCFD] denoise.restored_legacy station={station.id} '
                      f'proxy={len(keep)} base={len(proxy)} expected={base_count}',
                      flush=True)
            else:
                # 当存在去噪快照但无法重放时，绝不静默发布含噪基础代理。
                print(f'[PCFD] denoise.restore_invalid station={station.id} '
                      f'base={len(proxy)} expected={base_count} '
                      f'keep={len(keep)} saved={saved_count}', flush=True)
                raise RuntimeError(
                    f'站点 {station.display_name} 的去噪状态与当前代理点云不一致，'
                    '为避免展示原始噪点云，请重新执行去噪。')
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

    def _show_dataset(self, station):
        dataset = self._load_proxy_domain(station)
        self.render.clear_scene_display()
        self.render.show_station_proxy(
            station.id, station.display_name, dataset.proxy_points,
            dataset.proxy_colors, dataset_id=dataset.dataset_id)
        return dataset

    def show_single(self, station):
        if station.last_error:
            self.render.clear_scene_display()
            log_event(self.project_uuid, 'asset.invalid', station_id=station.id, reason=station.last_error)
            return
        self._show_dataset(station)
        self._active_station_id = station.id
        selected = [x.id for x in self.list_stations() if x.is_selected]
        PointCloudStationRepo.save_view(self.project_uuid, 'single', station.id, selected)
        log_event(self.project_uuid, 'station.switch', station_id=station.id)

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

    def restore_view(self):
        """恢复持久化的站点视图，若无则回退到第一个站点。"""
        if self._restoring or self._loaded_project == self.project_uuid:
            return
        self._restoring = True
        try:
            self._restore_view_once()
            self._loaded_project = self.project_uuid
        finally:
            self._restoring = False

    def _restore_view_once(self):
        rows = self.list_stations()
        if not rows:
            self.render.clear_scene_display()
            return
        state = PointCloudStationRepo.get_view(self.project_uuid)
        by_id = {row.id: row for row in rows}
        selected_ids = [sid for sid in (state.selected_station_ids if state else []) if sid in by_id]
        if not selected_ids:
            selected_ids = [row.id for row in rows if row.is_selected]
        if not selected_ids:
            selected_ids = [rows[0].id]
        for row in rows:
            PointCloudStationRepo.set_selected(self.project_uuid, row.id, row.id in selected_ids)
        active = by_id.get(state.active_station_id) if state else None
        mode = state.display_mode if state else 'single'
        if mode not in ('single', 'registered_merge', 'raw_overlay'):
            mode = 'single'
        if mode in ('registered_merge', 'raw_overlay') and len(selected_ids) > 1:
            self.merge_selected()
        else:
            self.show_single(active or by_id[selected_ids[0]])

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
                # The UI picks physical coordinates from these snapshots. A
                # small tolerance allows float32 display conversion without
                # accepting points from a stale station/domain.
                for points, snapshot in zip((src_pairs, tgt_pairs),
                                            (snapshots[1], snapshots[0])):
                    if len(snapshot) == 0:
                        raise ValueError('人工点云快照为空')
                    distances = np.min(
                        np.linalg.norm(points[:, None, :] - snapshot[None, :, :], axis=2),
                        axis=1)
                    if np.any(distances > 1e-3):
                        raise ValueError('人工对应点不属于当前站点快照，请重新选点')
        log_event(self.project_uuid, 'station.registration.started', count=len(rows))
        staged = []
        transformed_clouds = []
        registration_clouds = []
        for row in rows:
            dataset = self._load_proxy_domain(row)
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
                    voxel_size=voxel_size, max_correspondence_distance=0.12)
            else:
                result = point_to_plane_icp(
                    moving_reg.points, reference.points, voxel_size=voxel_size,
                    max_correspondence_distance=0.12)
            if not result.accepted:
                raise ValueError(f'{row.display_name} 配准失败：{result.message}，RMSE={result.inlier_rmse:.4f}')
            cloud = moving_reg.as_open3d()
            cloud.transform(result.transformation)
            transformed_clouds.append(cloud)
            staged.append((row, result))

        merged = o3d.geometry.PointCloud()
        for cloud in transformed_clouds:
            merged += cloud
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