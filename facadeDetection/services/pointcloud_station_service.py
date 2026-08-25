from pathlib import Path
import numpy as np
import open3d as o3d
from algorithms.registration import point_to_plane_icp
from services.dal.pointcloud_station_repo import PointCloudStationRepo
from utils.logging_utils import log_event

class PointCloudStationService:
    def __init__(self, render_service, project_uuid=None):
        self.render = render_service
        self.project_uuid = project_uuid
        self._restoring = False
        self._loaded_project = None

    def set_project(self, project_uuid):
        self.project_uuid = project_uuid
        self._loaded_project = None
        if project_uuid:
            PointCloudStationRepo.sync_assets(project_uuid)

    def refresh(self):
        if self.project_uuid:
            PointCloudStationRepo.sync_assets(self.project_uuid)

    def list_stations(self):
        return PointCloudStationRepo.list(self.project_uuid) if self.project_uuid else []

    def set_selected(self, station_id, selected):
        PointCloudStationRepo.set_selected(self.project_uuid, station_id, selected)

    def delete_selected(self):
        rows = [x for x in self.list_stations() if x.is_selected]
        PointCloudStationRepo.delete(self.project_uuid, [x.id for x in rows])
        # A deleted active station must not remain in the persisted view.
        remaining = self.list_stations()
        if not remaining:
            self.render.clear_station_scene()
            PointCloudStationRepo.save_view(self.project_uuid, 'single', None, [])
        else:
            PointCloudStationRepo.save_view(
                self.project_uuid, 'single', remaining[0].id,
                [x.id for x in remaining if x.is_selected] or [remaining[0].id])
        log_event(self.project_uuid, 'station.delete', count=len(rows))

    def _load(self, path):
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points)
        colors = np.asarray(cloud.colors) if cloud.has_colors() else None
        return points, colors

    def show_single(self, station):
        if station.last_error:
            self.render.clear_station_scene()
            log_event(self.project_uuid, 'asset.invalid', station_id=station.id, reason=station.last_error)
            return
        points, colors = self._load(station.source_path)
        self.render.clear_station_scene()
        self.render.show_station_cloud(station.id, station.display_name, points, colors)
        selected = [x.id for x in self.list_stations() if x.is_selected]
        PointCloudStationRepo.save_view(self.project_uuid, 'single', station.id, selected)
        log_event(self.project_uuid, 'station.switch', station_id=station.id)

    def merge_selected(self):
        rows = [x for x in self.list_stations() if x.is_selected and not x.last_error]
        if not rows: raise ValueError('请至少选择一个 PLY 站点')
        self.render.clear_station_scene()
        for row in rows:
            points, colors = self._load(row.registered_path or row.source_path)
            self.render.show_station_cloud(row.id, row.display_name, points, colors)
        mode = 'registered_merge' if any(x.registered_path for x in rows) else 'raw_overlay'
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
            self.render.clear_station_scene()
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

    def register_selected(self):
        self.refresh()
        rows = [x for x in self.list_stations() if x.is_selected]
        if len(rows) < 2: raise ValueError('点云配准至少需要选择两个 PLY 站点')
        reference, _ = self._load(rows[0].source_path)
        log_event(self.project_uuid, 'station.registration.started', count=len(rows))
        staged = []
        for row in rows[1:]:
            target, colors = self._load(row.source_path)
            # Inputs are already GPS/global corrected.  ICP estimates only the
            # residual refinement and must not reapply the JSON transform.
            result = point_to_plane_icp(target, reference, voxel_size=0.05)
            if not result.accepted:
                raise ValueError(f'{row.display_name} 配准失败：{result.message}，RMSE={result.inlier_rmse:.4f}')
            # Stage outputs first; database is updated only after every station
            # passes quality validation.
            out = Path(row.source_path).parent / f'{Path(row.source_path).stem}.registered.ply.tmp'
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
            if colors is not None and len(colors) == len(target):
                cloud.colors = o3d.utility.Vector3dVector(colors)
            cloud.transform(result.transformation)
            o3d.io.write_point_cloud(str(out), cloud)
            staged.append((row, result, out, out.with_suffix('')))
        for row, result, tmp, out in staged:
            tmp.replace(out)
            PointCloudStationRepo.update_registration(
                self.project_uuid, row.id, result.transformation.tolist(),
                result.fitness, result.inlier_rmse, str(out))
        # 重新从数据库读取，确保提交后的 registered_path 被合并分支看到。
        # Re-read persisted rows before rendering, then keep the selection and
        # view state bound to this project.
        self.merge_selected()
        log_event(self.project_uuid, 'station.registration.completed', count=len(rows))
