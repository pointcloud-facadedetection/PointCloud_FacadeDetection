from pathlib import Path
from sqlalchemy import select
from db.connection import project_session
from models import Project, FileAsset, PointCloudStation, PointCloudViewState
from models.enums import FileKind
from services.dal.file_repo import FileRepo

class PointCloudStationRepo:
    @staticmethod
    def list(project_uuid):
        with project_session(project_uuid) as s:
            p = s.execute(select(Project).where(Project.uuid == project_uuid)).scalar_one_or_none()
            if not p: return []
            return s.execute(select(PointCloudStation).where(
                PointCloudStation.project_id == p.id, PointCloudStation.is_deleted == False
            ).order_by(PointCloudStation.display_order, PointCloudStation.id)).scalars().all()

    @staticmethod
    def sync_assets(project_uuid):
        """Reconcile FileAsset rows into the station projection.

        Return counts so the caller can distinguish a valid empty project from
        a failed restore.  This is intentionally one transaction: the UI must
        never continue with a partially refreshed station list.
        """
        with project_session(project_uuid) as s:
            p = s.execute(select(Project).where(Project.uuid == project_uuid)).scalar_one()
            assets = s.execute(select(FileAsset).where(
                FileAsset.project_id == p.id, FileAsset.is_deleted == False,
            ).order_by(FileAsset.id)).scalars().all()
            # Older project databases used inconsistent kind values.  The
            # extension is the authoritative fallback for PLY station assets.
            assets = [asset for asset in assets if (
                str(asset.kind) in (FileKind.raw_pointcloud.value, 'raw_pointcloud')
                or Path(asset.path or '').suffix.lower() == '.ply'
            )]
            all_rows = s.execute(select(PointCloudStation).where(
                PointCloudStation.project_id == p.id
            )).scalars().all()
            existing = {x.file_asset_id: x for x in all_rows if x.file_asset_id is not None}
            by_key = {x.station_key: x for x in all_rows if x.station_key}
            for order, asset in enumerate(assets):
                valid, reason = FileRepo.validate_asset(asset)
                asset.meta_json = dict(asset.meta_json or {}, validation_status=reason)
                key = f'{project_uuid}:{asset.sha256 or asset.id}'
                if asset.id in existing:
                    row = existing[asset.id]
                    row.is_deleted = False
                    row.source_path = asset.path
                    row.last_error = None if valid else reason
                    row.display_name = asset.original_name or Path(asset.path).name
                    row.display_order = order
                    continue
                # Legacy databases may contain a station with the same key but
                # a missing/stale file_asset_id. Reuse it instead of inserting
                # into the globally-unique station_key column.
                if key in by_key:
                    row = by_key[key]
                    row.file_asset_id = asset.id
                    row.is_deleted = False
                    row.source_path = asset.path
                    row.last_error = None if valid else reason
                    row.display_name = asset.original_name or Path(asset.path).name
                    row.display_order = order
                    continue
                station = PointCloudStation(project_id=p.id, file_asset_id=asset.id,
                    station_key=key,
                    display_name=asset.original_name or Path(asset.path).name,
                    source_path=asset.path, display_order=order,
                    is_selected=(order == 0))
                station.last_error = None if valid else reason
                s.add(station)
            s.flush()
            active_rows = s.execute(select(PointCloudStation).where(
                PointCloudStation.project_id == p.id,
                PointCloudStation.is_deleted == False,
            )).scalars().all()
            return {
                'assets': len(assets),
                'stations': len(active_rows),
                'invalid': sum(1 for x in active_rows if x.last_error),
            }

    @staticmethod
    def set_selected(project_uuid, station_id, selected):
        with project_session(project_uuid) as s:
            row = s.get(PointCloudStation, station_id)
            if row: row.is_selected = bool(selected)

    @staticmethod
    def get_view(project_uuid):
        with project_session(project_uuid) as s:
            p = s.execute(select(Project).where(Project.uuid == project_uuid)).scalar_one_or_none()
            if not p:
                return None
            return s.execute(select(PointCloudViewState).where(
                PointCloudViewState.project_id == p.id)).scalar_one_or_none()

    @staticmethod
    def delete(project_uuid, station_ids):
        with project_session(project_uuid) as s:
            station_ids = {int(sid) for sid in station_ids}
            for sid in station_ids:
                row = s.get(PointCloudStation, sid)
                if row:
                    # Keep the legacy soft-delete flag for old databases, but
                    # also unlink the asset so the station cannot reappear on
                    # the next synchronization.
                    row.is_deleted = True
                    if row.file_asset_id is not None:
                        asset = s.get(FileAsset, row.file_asset_id)
                        if asset is not None:
                            asset.is_deleted = True

    @staticmethod
    def update_registration(project_uuid, station_id, transform, fitness, rmse, path):
        with project_session(project_uuid) as s:
            row = s.get(PointCloudStation, station_id)
            if row:
                row.transform_json = transform
                row.fitness, row.inlier_rmse = fitness, rmse
                row.registration_status, row.registered_path = 'success', path

    @staticmethod
    def update_registrations(project_uuid, station_ids, transforms, path):
        """Commit one ICP operation atomically for every participating station."""
        ids = [int(value) for value in station_ids]
        with project_session(project_uuid) as s:
            rows = {row.id: row for row in s.execute(select(PointCloudStation).where(
                PointCloudStation.id.in_(ids),
                PointCloudStation.is_deleted == False,
            )).scalars().all()}
            if len(rows) != len(set(ids)):
                raise ValueError('配准站点已不存在或已被删除，结果未提交')
            for station_id in ids:
                transform, fitness, rmse = transforms[station_id]
                row = rows[station_id]
                row.transform_json = transform
                row.fitness, row.inlier_rmse = fitness, rmse
                row.registration_status, row.registered_path = 'success', path

    @staticmethod
    def save_view(project_uuid, mode, active_id, selected_ids, registration_id=None):
        with project_session(project_uuid) as s:
            p = s.execute(select(Project).where(Project.uuid == project_uuid)).scalar_one()
            row = s.execute(select(PointCloudViewState).where(PointCloudViewState.project_id == p.id)).scalar_one_or_none()
            if row is None: row = PointCloudViewState(project_id=p.id); s.add(row)
            row.display_mode, row.active_station_id = mode, active_id
            row.selected_station_ids, row.registration_id = list(selected_ids), registration_id