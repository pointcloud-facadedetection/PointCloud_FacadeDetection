from pathlib import Path
from sqlalchemy import select
from db.connection import project_session
from models import Project, FileAsset, PointCloudStation, PointCloudViewState
from models.enums import FileKind
from services.dal.file_repo import FileRepo
from utils.logging_utils import log_event

class PointCloudStationRepo:
    @staticmethod
    def _runtime_path(asset):
        """Return the PLY consumed by runtime, never an original E57 path."""
        raw_path = Path(asset.path or '')
        if raw_path.suffix.lower() != '.e57':
            return raw_path, None
        cache_path = Path((asset.meta_json or {}).get('cache_path') or '')
        if cache_path.is_file() and cache_path.suffix.lower() == '.ply':
            return cache_path, None
        if raw_path.is_file():
            return None, 'E57 已导入但 cache PLY 丢失，请重新导入原始 E57'
        return None, '原始 E57 与 cache PLY 均不存在，请重新选择 E57 文件'

    @staticmethod
    def get_asset_fingerprint(project_uuid, station_id):
        """返回某个站点的持久化source fingerprint

        将此查找操作保存在存储库中，既避免了依赖脱离的 ORM关系，
        又使运行时缓存键与文件名无关。

        对于 E57 资产，若已生成 cache PLY，则返回 cache PLY 的指纹；
        这样去噪/代理缓存与 cache 内容严格绑定，原始 E57 移动/删除
        不影响已就绪站点的恢复。
        """
        with project_session(project_uuid) as s:
            row = s.get(PointCloudStation, int(station_id))
            if row is None or row.is_deleted:
                return None
            asset = s.get(FileAsset, row.file_asset_id) if row.file_asset_id else None
            if asset is None:
                return (str(row.source_path), None, None)
            # E57 运行时身份是其 cache PLY，指纹必须反映 cache 内容
            if Path(asset.path or '').suffix.lower() == '.e57':
                meta = dict(asset.meta_json or {})
                cache_path = meta.get('cache_path')
                cache_sha = meta.get('cache_sha256')
                cache_size = meta.get('cache_size_bytes')
                if cache_path and cache_sha:
                    return (str(cache_path), cache_sha, cache_size)
            return (str(asset.path), asset.sha256, asset.size_bytes)

    @staticmethod
    def get_by_asset_id(project_uuid, asset_id):
        """Find the active station projection created for a FileAsset."""
        with project_session(project_uuid) as s:
            p = s.execute(select(Project).where(Project.uuid == project_uuid)).scalar_one_or_none()
            if p is None:
                return None
            return s.execute(select(PointCloudStation).where(
                PointCloudStation.project_id == p.id,
                PointCloudStation.file_asset_id == int(asset_id),
                PointCloudStation.is_deleted == False,
            )).scalar_one_or_none()
    @staticmethod
    def get_denoise_state(project_uuid, station_id):
        with project_session(project_uuid) as s:
            row = s.get(PointCloudStation, int(station_id))
            return dict(row.denoise_state_json or {}) if row else None

    @staticmethod
    def save_denoise_state(project_uuid, station_id, state):
        """Atomically persist the final proxy index mapping for one station."""
        with project_session(project_uuid) as s:
            row = s.get(PointCloudStation, int(station_id))
            if row is None or row.is_deleted:
                raise ValueError(f'去噪站点不存在: {station_id}')
            row.denoise_state_json = dict(state or {})

    @staticmethod
    def clear_denoise_state(project_uuid, station_id):
        with project_session(project_uuid) as s:
            row = s.get(PointCloudStation, int(station_id))
            if row is not None:
                row.denoise_state_json = None

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
        """将 FileAsset 行与站点投影进行对齐。

        返回计数，以便调用方能够区分有效的空项目与恢复失败的情况。
        """
        with project_session(project_uuid) as s:
            p = s.execute(select(Project).where(Project.uuid == project_uuid)).scalar_one()
            assets = s.execute(select(FileAsset).where(
                FileAsset.project_id == p.id, FileAsset.is_deleted == False,
            ).order_by(FileAsset.id)).scalars().all()
            assets = [asset for asset in assets if (
                str(asset.kind) in (FileKind.raw_pointcloud.value, 'raw_pointcloud')
                or Path(asset.path or '').suffix.lower() in {'.ply', '.e57'}
            )]
            all_rows = s.execute(select(PointCloudStation).where(
                PointCloudStation.project_id == p.id
            )).scalars().all()
            existing = {x.file_asset_id: x for x in all_rows if x.file_asset_id is not None}
            by_key = {x.station_key: x for x in all_rows if x.station_key}
            for order, asset in enumerate(assets):
                valid, reason = FileRepo.validate_asset(asset)
                asset.meta_json = dict(asset.meta_json or {}, validation_status=reason)
                runtime_path, cache_reason = PointCloudStationRepo._runtime_path(asset)
                if cache_reason:
                    valid, reason = False, cache_reason
                    log_event(project_uuid, 'e57.cache.missing', asset_id=asset.id,
                              source_path=asset.path,
                              cache_path=(asset.meta_json or {}).get('cache_path'))
                elif Path(asset.path or '').suffix.lower() == '.e57':
                    # 导入完成后 cache PLY 是唯一运行时资产。原 E57 可移动/归档，
                    # 不能因此阻断项目恢复，也绝不回退去读取其复杂元数据。
                    valid, reason = True, 'ok'
                key = f'{project_uuid}:{asset.sha256 or asset.id}'
                if asset.id in existing:
                    row = existing[asset.id]
                    row.is_deleted = False
                    row.source_path = str(runtime_path or asset.path)
                    row.last_error = None if valid else reason
                    row.display_name = asset.original_name or Path(asset.path).name
                    row.display_order = order
                    continue
                # 兼容旧数据库
                if key in by_key:
                    row = by_key[key]
                    row.file_asset_id = asset.id
                    row.is_deleted = False
                    row.source_path = str(runtime_path or asset.path)
                    row.last_error = None if valid else reason
                    row.display_name = asset.original_name or Path(asset.path).name
                    row.display_order = order
                    continue
                station = PointCloudStation(project_id=p.id, file_asset_id=asset.id,
                    station_key=key,
                    display_name=asset.original_name or Path(asset.path).name,
                    source_path=str(runtime_path or asset.path), display_order=order,
                    is_selected=(order == 0))
                station.last_error = None if valid else reason
                s.add(station)
            active_asset_ids = {asset.id for asset in assets}
            for row in all_rows:
                if (row.file_asset_id is not None and
                        row.file_asset_id not in active_asset_ids):
                    row.is_deleted = True
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