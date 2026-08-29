from pathlib import Path
import numpy as np
import open3d as o3d
from algorithms.registration import point_to_plane_icp
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

    def set_project(self, project_uuid):
        self.project_uuid = project_uuid
        self._loaded_project = None
        self._dataset_ids.clear()
        self._active_station_id = None

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
            # A registration result is shared by all stations in one
            # operation; remove it only when no remaining station references it.
            still_referenced = any(
                str(x.registered_path) == path for x in self.list_stations())
            if (not still_referenced and result.name.startswith('registration_')
                    and result.exists()):
                try:
                    result.unlink()
                except OSError:
                    pass
        # A deleted active station must not remain in the persisted view.
        remaining = self.list_stations()
        if not remaining:
            self.render.clear_scene_display()
            PointCloudStationRepo.save_view(self.project_uuid, 'single', None, [])
        else:
            # Deletion already cleared the scene; render the fallback station
            # immediately instead of relying on restore_view's once-per-project
            # guard, which would otherwise leave the viewport blank.
            fallback = remaining[0]
            PointCloudStationRepo.save_view(
                self.project_uuid, 'single', fallback.id,
                [x.id for x in remaining if x.is_selected] or [fallback.id])
            self._loaded_project = None
            self.show_single(fallback)
        log_event(self.project_uuid, 'station.delete', count=len(rows))

    def _load(self, path):
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points)
        colors = np.asarray(cloud.colors) if cloud.has_colors() else None
        return points, colors

    def _load_proxy_domain(self, station):
        """Load one station's two domains and return proxy data for display.

        The raw array is registered as the precision domain; only the proxy
        returned by the dataset is allowed to cross the viewport boundary.
        """
        if self.pointcloud is None:
            raise RuntimeError('PointCloudService 未注入，无法建立站点索引')
        dataset_id = f'{self.project_uuid}:{station.id}'
        existing = self.pointcloud.get_dataset(dataset_id)
        # FLS import may have already built the proxy under its filename
        # before the station projection receives the database id. Reuse that
        # dataset instead of reading the raw PLY and building a second index.
        if existing is None:
            legacy_id = f'{self.project_uuid}:{Path(station.source_path).name}'
            existing = self.pointcloud.get_dataset(legacy_id)
            if existing is not None:
                self.pointcloud.datasets.pop(legacy_id, None)
                existing.dataset_id = dataset_id
                self.pointcloud.datasets[dataset_id] = existing
        if existing is not None:
            self._dataset_ids[station.id] = dataset_id
            return existing
        points, colors = self._load(station.source_path)
        source_id = f'{dataset_id}:source'
        dist_path = Path(station.source_path).with_suffix('.dist')
        metadata = {'source_id': source_id, 'station_id': station.id,
                    'source_raw_count': int(len(points))}
        if dist_path.exists():
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
        return dataset

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
        # A partial or mixed registration set is not a valid result snapshot.
        mode = 'registered_merge' if (
            len(registered_paths) == 1 and all(x.registered_path for x in rows)
            and all(x.registration_status == 'success' for x in rows)
        ) else 'raw_overlay'
        PointCloudStationRepo.save_view(self.project_uuid, mode, rows[0].id, [x.id for x in rows])
        log_event(self.project_uuid, 'station.merge', mode=mode, count=len(rows))

    def restore_view(self):
        """Restore the persisted station view, falling back to station one."""
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

    def register_selected(self, update_viewport=True):
        self.refresh()
        rows = [x for x in self.list_stations() if x.is_selected]
        if len(rows) < 2: raise ValueError('点云配准至少需要选择两个 PLY 站点')
        reference, _ = self._load(rows[0].source_path)
        log_event(self.project_uuid, 'station.registration.started', count=len(rows))
        staged = []
        transformed_clouds = []
        reference_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(reference))
        transformed_clouds.append(reference_cloud)
        for row in rows[1:]:
            target, colors = self._load(row.source_path)
            # Inputs are already GPS/global corrected.  ICP estimates only the
            # residual refinement and must not reapply the JSON transform.
            result = point_to_plane_icp(target, reference, voxel_size=0.05)
            if not result.accepted:
                raise ValueError(f'{row.display_name} 配准失败：{result.message}，RMSE={result.inlier_rmse:.4f}')
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
            if colors is not None and len(colors) == len(target):
                cloud.colors = o3d.utility.Vector3dVector(colors)
            cloud.transform(result.transformation)
            transformed_clouds.append(cloud)
            staged.append((row, result))

        # Export one authoritative, downsampled registration result.
        merged = o3d.geometry.PointCloud()
        for cloud in transformed_clouds:
            merged += cloud
        merged = merged.voxel_down_sample(0.05)
        result_dir = Storage.ensure_project_dirs(self.project_uuid)['results']
        operation_id = uuid.uuid4().hex
        out = result_dir / f'registration_{operation_id}.ply'
        # Keep the PLY suffix so Open3D selects the PLY writer for the stage file.
        tmp = result_dir / f'.registration_{operation_id}.tmp.ply'
        if not o3d.io.write_point_cloud(str(tmp), merged):
            raise IOError(f'配准结果写入失败：{tmp}')
        tmp.replace(out)

        payload = {
            'station_ids': [x.id for x in rows],
            'reference_id': rows[0].id,
            'result_path': str(out),
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
        """Commit one completed ICP transaction and publish its result view."""
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
