from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from services.dal.project_repo import ProjectRepo
from services.file_service import FileService
from services.dal.file_repo import FileRepo
from utils.logging_utils import log_event
from models import Facade, QualityMetric
from db.connection import project_session
from sqlalchemy import select
from services.viewport_render_service import ViewportRenderService
from services.dal.pointcloud_station_repo import PointCloudStationRepo
from utils.workers import PointCloudLoadWorker
from models.enums import FileKind
from services.e57_cache_service import ensure_e57_cache


@dataclass
class ProjectCard:
    """供 UI 卡片渲染的最小数据。"""
    project_id: str
    name: str
    directory_path: str
    org_unit: str | None = None
    address: str | None = None
    building_floor: str | None = None
    remarks: str | None = None
    # PDF 报告元信息
    construction_unit: str | None = None
    construction_unit_executor: str | None = None
    inspection_unit: str | None = None
    supervision_unit: str | None = None
    client_unit: str | None = None
    report_no: str | None = None
    inspection_date: str | None = None
    report_date: str | None = None
    inspection_params_json: str | None = None
    fls_directories: list[str] | None = None
    pointcloud_files: list[str] | None = None
    photo_files: list[str] | None = None


class ProjectOverviewService:
    """项目概览页调度器：供 MainWindow 直接调用。"""

    def __init__(self, viewport=None, render_service: ViewportRenderService | None = None, db=None):
        self.viewport = viewport
        self.render_service = render_service
        self.db = db
        self._file_service: Optional[FileService] = None

    def create_load_worker(self, operation: str, project_uuid: str,
                           *, file_paths=None, directory=None, directories=None):
        """构建后台加载 worker：计算段（解析/注册/转换）在池线程执行。

        结果中的 prepared 对象携带全部待提交数组，Open3D 提交段由 GUI
        完成回调执行（见 ProjectLifecycleController.on_load_finished）。
        activate 的 worker 由 ProjectLifecycleController 自行构建
        （计算段依赖 pointcloud/station 服务，不在本服务内）。
        """
        if operation == 'upload':
            def run(worker):
                worker.check_cancelled()
                prepared, uploaded = self.prepare_upload_files(
                    list(file_paths or []), project_uuid, worker=worker)
                return {'operation': operation, 'project_uuid': project_uuid,
                        'prepared': prepared, 'uploaded': uploaded}
            return PointCloudLoadWorker(run)
        if operation == 'fls':
            dirs = list(directories or [])
            if directory:
                dirs.append(directory)
            def run(worker):
                all_results = []
                total_success = 0
                for idx, d in enumerate(dirs):
                    worker.check_cancelled()
                    result = self.import_fls_directory(
                        d, project_uuid,
                        progress_cb=lambda done, total, name, i=idx, n=len(dirs):
                            worker.signals.progress.emit(
                                int(10 + 80 * (i + done / max(total, 1)) / max(n, 1)),
                                f'正在导入站点 {name} ({done}/{total})'))
                    all_results.append(result)
                    if result.get('success'):
                        total_success += result.get('uploaded', 1)
                # 兼容单结果接口：合并为统一的 result
                merged = {
                    'success': total_success > 0,
                    'message': '; '.join(
                        r.get('message', '') for r in all_results
                        if not r.get('success') and r.get('message')) or '',
                    'ply_paths': [p for r in all_results for p in r.get('ply_paths', [])],
                    'uploaded': total_success,
                }
                return {'operation': operation, 'project_uuid': project_uuid,
                        'result': merged, 'results': all_results}
            return PointCloudLoadWorker(run)
        raise ValueError(f'未知加载操作: {operation}')

    # -------------- 项目管理 --------------
    def list_projects(self) -> list[ProjectCard]:
        items = ProjectRepo.list_projects()
        return [self._card_from_info(i) for i in items]

    @staticmethod
    def _card_from_info(i: dict) -> ProjectCard:
        return ProjectCard(
            project_id=i["project_id"], name=i["name"],
            directory_path=i["directory_path"],
            org_unit=i.get("org_unit"), address=i.get("address"),
            building_floor=i.get("building_floor"), remarks=i.get("remarks"),
            construction_unit=i.get("construction_unit"),
            construction_unit_executor=i.get("construction_unit_executor"),
            inspection_unit=i.get("inspection_unit"),
            supervision_unit=i.get("supervision_unit"),
            client_unit=i.get("client_unit"),
            report_no=i.get("report_no"),
            inspection_date=i.get("inspection_date"),
            report_date=i.get("report_date"),
            inspection_params_json=i.get("inspection_params_json"),
            fls_directories=list(i.get("fls_directories") or []),
            pointcloud_files=list(i.get("pointcloud_files") or []),
            photo_files=list(i.get("photo_files") or []),
        )

    def create_project(
        self,
        name: str,
        org_unit: str | None = None,
        address: str | None = None,
        remarks: str | None = None,
        building_floor: str | None = None,
        construction_unit: str | None = None,
        construction_unit_executor: str | None = None,
        inspection_unit: str | None = None,
        supervision_unit: str | None = None,
        client_unit: str | None = None,
        report_no: str | None = None,
        inspection_date: str | None = None,
        report_date: str | None = None,
        inspection_params_json: str | None = None,
        fls_directories: list[str] | None = None,
        pointcloud_files: list[str] | None = None,
        photo_files: list[str] | None = None,
    ) -> ProjectCard:
        """创建持久化项目，并向 UI 返回统一的项目卡片模型。"""
        info = ProjectRepo.create_project(
            name=name,
            org_unit=org_unit,
            address=address,
            remarks=remarks,
            building_floor=building_floor,
            construction_unit=construction_unit,
            construction_unit_executor=construction_unit_executor,
            inspection_unit=inspection_unit,
            supervision_unit=supervision_unit,
            client_unit=client_unit,
            report_no=report_no,
            inspection_date=inspection_date,
            report_date=report_date,
            inspection_params_json=inspection_params_json,
        )
        card = ProjectCard(
            project_id=info["project_uuid"],
            name=info["name"],
            directory_path=info.get("root_dir", info.get("directory_path", "")),
            org_unit=info.get("org_unit"),
            address=info.get("address"),
            building_floor=info.get("building_floor"),
            remarks=info.get("remarks"),
            construction_unit=info.get("construction_unit"),
            construction_unit_executor=info.get("construction_unit_executor"),
            inspection_unit=info.get("inspection_unit"),
            supervision_unit=info.get("supervision_unit"),
            client_unit=info.get("client_unit"),
            report_no=info.get("report_no"),
            inspection_date=info.get("inspection_date"),
            report_date=info.get("report_date"),
            inspection_params_json=info.get("inspection_params_json"),
        )
        self.update_project_assets(card.project_id, fls_directories or [], pointcloud_files or [], photo_files or [])
        return self.get_project(card.project_id) or card

    def open_project(self, directory_path: str) -> ProjectCard:
        # TODO(生命周期/性能): open_project：目录扫描、索引同步和项目激活应避免在 GUI 线程同步执行，并核查重复索引读取与激活失败回滚。
        path = Path(directory_path).expanduser().resolve()
        if self.viewport is not None and hasattr(self.viewport, 'clear'):
            self.viewport.clear()
        # 如果该文件夹中存在 pcfd 索引，则采用该索引并将其更新或插入到索引数据库中，然后加载资源
        try:
            from config.storage import Storage
            from db.connection import upsert_index_project
            idx = Storage.load_pcfd_index(path)
            if idx is not None:
                proj = (idx.get('project') or {})
                puid = str(proj.get('uuid') or '')
                pname = str(proj.get('name_cn') or path.name or '未命名项目')
                if puid:
                    # 确保索引行存在
                    upsert_index_project(puid, pname, str(path))
                    # 激活项目（确保采用按项目划分的数据库架构）
                    ProjectRepo.load_and_activate(puid)
                    # 索引分支会提前返回，因此必须显式地重建站点投影 
                    PointCloudStationRepo.sync_assets(puid)
                    return ProjectCard(
                        project_id=puid,
                        name=pname,
                        directory_path=str(path),
                        org_unit=proj.get("org_unit"),
                        address=proj.get("address"),
                        building_floor=proj.get("building_floor"),
                        remarks=proj.get("remarks"),
                        construction_unit=proj.get("construction_unit"),
                        construction_unit_executor=proj.get("construction_unit_executor"),
                        inspection_unit=proj.get("inspection_unit"),
                        supervision_unit=proj.get("supervision_unit"),
                        client_unit=proj.get("client_unit"),
                        report_no=proj.get("report_no"),
                        inspection_date=proj.get("inspection_date"),
                        report_date=proj.get("report_date"),
                        inspection_params_json=proj.get("inspection_params_json"),
                        **self.get_project_assets(puid),
                    )
        except Exception:
            pass
        # 尝试匹配已登记项目
        for p in self.list_projects():
            if Path(p.directory_path).resolve() == path:
                # 激活场景（若需要）
                ProjectRepo.load_and_activate(p.project_id)
                PointCloudStationRepo.sync_assets(p.project_id)
                return p
        # 未登记则创建新项目（名称取目录名）
        info = ProjectRepo.create_project(name=path.name or "未命名项目")
        pc = ProjectCard(
            project_id=info["project_uuid"],
            name=info["name"],
            directory_path=info["root_dir"],
            org_unit=info.get("org_unit"),
            address=info.get("address"),
            building_floor=info.get("building_floor"),
            remarks=info.get("remarks"),
            construction_unit=info.get("construction_unit"),
            construction_unit_executor=info.get("construction_unit_executor"),
            inspection_unit=info.get("inspection_unit"),
            supervision_unit=info.get("supervision_unit"),
            client_unit=info.get("client_unit"),
            report_no=info.get("report_no"),
            inspection_date=info.get("inspection_date"),
            report_date=info.get("report_date"),
            inspection_params_json=info.get("inspection_params_json"),
        )
        # 新登记项目：登记目录中的全部 PLY/E57；实际代理构建与渲染由
        # StationService 在项目激活后按需完成，避免产生第二条加载链路。
        try:
            pointcloud_candidates = [item for item in path.iterdir()
                                     if item.is_file() and
                                     item.suffix.lower() in {'.ply', '.e57'}]
            for pointcloud_path in pointcloud_candidates:
                asset = FileRepo.import_file(project_uuid=pc.project_id,
                                             src_path=str(pointcloud_path),
                                             kind=FileKind.raw_pointcloud,
                                             copy_into_project=False)
                if pointcloud_path.suffix.lower() == '.e57':
                    ensure_e57_cache(pc.project_id, asset)
            if pointcloud_candidates:
                PointCloudStationRepo.sync_assets(pc.project_id)
        except Exception:
            pass
        return pc

    def activate_project(self, project_id: str) -> None:
        """Validate and activate a project without hiding restore failures."""
        if not project_id:
            raise ValueError('项目标识为空，无法激活项目')
        ProjectRepo.load_and_activate(project_id)
        # 补偿旧项目及历史旁路登记的 E57：先生成 cache，再同步站点。
        # 原始 E57 缺失时保留结构化错误，由站点层提示重导，不回退解析 E57。
        for asset in FileRepo.list_assets_by_kind(project_id, FileKind.raw_pointcloud):
            if Path(asset.path or '').suffix.lower() != '.e57':
                continue
            try:
                ensure_e57_cache(project_id, asset)
            except Exception as exc:
                log_event(project_id, 'e57.cache.ensure_failed', asset_id=asset.id,
                          source_path=asset.path, error=repr(exc))
        # 在用户界面请求列表之前，先重建站点投影。
        stats = PointCloudStationRepo.sync_assets(project_id)
        log_event(project_id, 'stations.synced', **stats)

    def load_historical_facades(self, project_id: str, station_id: int | None = None) -> list[dict]:
        from config.storage import Storage
        with project_session(project_id) as s:
            query = select(Facade).where(Facade.is_deleted == 0)
            # 旧数据行的 station_id 为 NULL，绝不能将其推测为当前活动站；
            # 否则可能会再次引入跨站的颜色和索引。
            if station_id is not None:
                query = query.where(Facade.station_id == int(station_id))
            rows = s.execute(query.order_by(Facade.display_no, Facade.id)).scalars().all()
            result = []
            for row in rows:
                metrics = s.execute(select(QualityMetric).where(QualityMetric.facade_id == row.id)).scalars().all()
                geometry = row.plane_json or {}
                display_no = int(row.display_no or 0)
                if display_no <= 0:
                    try:
                        display_no = int(str(row.label).rsplit(' ', 1)[-1])
                    except (ValueError, IndexError):
                        display_no = len(result) + 1
                point_count = int(row.point_count or geometry.get('point_count') or
                                  len(geometry.get('proxy_indices') or geometry.get('inlier_indices') or []))
                raw_point_count = int(row.raw_point_count or geometry.get('raw_point_count') or point_count)
                item = {'id': row.id, 'facade_db_id': row.id, 'display_no': display_no,
                               'point_count': point_count, 'raw_point_count': raw_point_count,
                               'type': row.label, 'type_label': row.label,
                               'area': row.area or 0.0, 'plane': row.plane_json,
                               'bbox': row.bbox_json,
                               **{key: geometry[key] for key in (
                                   'plane_model', 'normal', 'center', 'inlier_indices',
                                    'proxy_indices', 'measurement_indices', 'voxel_ids',
                                    'cloud_name', '__index_space', 'review_status')
                                   if key in geometry},
                                'quality_metrics': [{'name': m.metric_name, 'value': m.value,
                                                    'unit': m.unit, 'pass': m.pass_flag}
                                                    for m in metrics],
                               'quality_status': row.quality_status,
                               'quality_report': row.quality_report_json,
                               'color': row.color_json,
                                'dataset_revision': row.dataset_revision,
                                'dataset_id': row.dataset_id,          # 新增
                                'dataset_fingerprint': row.dataset_fingerprint,  # 新增
                                'station_id': row.station_id}          # 新增
                artifact = (row.quality_report_json or {}).get('quality_artifact_path')
                if artifact:
                    path = Path(artifact)
                    if not path.is_absolute():
                        path = Path(Storage.ensure_project_dirs(project_id)['results']) / path
                    from services.dal.results_repo import ResultsRepo
                    ids = ResultsRepo.load_quality_artifact(path)
                    if len(ids):
                        item['quality_report'] = dict(item.get('quality_report') or {})
                        item['quality_report']['__global_indices'] = ids
                        item['quality_report']['__index_space'] = 'facade_local_to_raw_global'
                result.append(item)
            log_event(project_id, 'results.loaded', facades=len(result))
            return result

    def load_all_historical_facades(self, project_id: str) -> dict[int, list[dict]]:
        """加载项目中所有站点的历史立面检测结果，按 station_id 分组返回。
        用于项目激活时一次性恢复全量报告数据，支持多站点增量拓展。"""
        from services.dal.pointcloud_station_repo import PointCloudStationRepo
        stations = PointCloudStationRepo.list(project_id)
        all_results: dict[int, list[dict]] = {}
        for station in stations:
            if getattr(station, 'last_error', None):
                continue
            sid = int(station.id)
            facades = self.load_historical_facades(project_id, station_id=sid)
            if facades:
                all_results[sid] = facades
        return all_results

    def remove_project(self, project_id: str) -> bool:
        return ProjectRepo.delete_project(project_id, hard=True)

    def get_project(self, project_id: str) -> Optional[ProjectCard]:
        for p in self.list_projects():
            if p.project_id == project_id:
                return p
        return None

    def update_project(self, project_id: str, **fields) -> ProjectCard:
        resources = {key: fields.pop(key, None) for key in (
            'fls_directories', 'pointcloud_files', 'photo_files')}
        info = ProjectRepo.update_project(project_id, **fields)
        if info is None:
            raise ValueError('项目不存在或已被删除。')
        card = ProjectCard(
            project_id=info["project_id"], name=info["name"],
            directory_path=info["directory_path"],
            org_unit=info.get("org_unit"), address=info.get("address"),
            building_floor=info.get("building_floor"), remarks=info.get("remarks"),
            construction_unit=info.get("construction_unit"),
            construction_unit_executor=info.get("construction_unit_executor"),
            inspection_unit=info.get("inspection_unit"),
            supervision_unit=info.get("supervision_unit"),
            client_unit=info.get("client_unit"),
            report_no=info.get("report_no"),
            inspection_date=info.get("inspection_date"),
            report_date=info.get("report_date"),
            inspection_params_json=info.get("inspection_params_json"),
        )
        # The dialog always supplies all three lists.  An empty list is an
        # intentional clear operation and must be persisted as such.
        if any(value is not None for value in resources.values()):
            self.update_project_assets(project_id, resources['fls_directories'] or [],
                                       resources['pointcloud_files'] or [], resources['photo_files'] or [])
            # Keep station projections in lockstep with the edited asset list;
            # otherwise deleted rows reappear until the next application start.
            PointCloudStationRepo.sync_assets(project_id)
        return self.get_project(project_id) or card

    def get_project_assets(self, project_uuid: str) -> dict[str, list[str]]:
        from config.storage import Storage
        root = Storage.project_root(project_uuid)
        index = Storage.load_pcfd_index(root) or {}
        assets = index.get('assets') or {}
        pointclouds = [a.path for a in FileRepo.list_assets_by_kind(project_uuid, FileKind.raw_pointcloud)]
        photos = [a.path for a in FileRepo.list_assets_by_kind(project_uuid, FileKind.raw_image)]
        return {
            'fls_directories': list(dict.fromkeys(assets.get('fls_folders') or [])),
            'pointcloud_files': list(dict.fromkeys(pointclouds)),
            'photo_files': list(dict.fromkeys(photos)),
        }

    def update_project_assets(self, project_uuid: str, fls_directories: list[str],
                              pointcloud_files: list[str], photo_files: list[str]) -> None:
        """以弹窗列表为事实来源同步资源，支持新增、删除、清空后保存。

        FLS 目录会被实际转换并导入；E57 生成 cache PLY 后统一按 PLY 处理；
        原生 PLY 直接导入；照片仅记录元数据。
        """
        from config.storage import Storage
        from models.enums import FileKind
        from services import proxy_cache
        from services.dal.pointcloud_station_repo import PointCloudStationRepo
        from pathlib import Path
        import os
        fls = [str(Path(p).expanduser().resolve()) for p in fls_directories if p]
        pcs = [str(Path(p).expanduser().resolve()) for p in pointcloud_files if p]
        photos = [str(Path(p).expanduser().resolve()) for p in photo_files if p]
        for p in fls:
            if not Path(p).is_dir():
                raise ValueError(f'FLS 目录不存在：{p}')
        for p in pcs:
            if not Path(p).is_file() or Path(p).suffix.lower() not in FileService.SUPPORTED_POINT_CLOUD_EXT:
                raise ValueError(f'点云文件无效或格式不支持：{p}')
        for p in photos:
            if not Path(p).is_file() or Path(p).suffix.lower() not in FileService.SUPPORTED_IMAGE_EXT:
                raise ValueError(f'照片文件无效或格式不支持：{p}')
        old = self.get_project_assets(project_uuid)
        old_fls = set(old.get('fls_directories') or [])

        # 删除阶段：清理已移除资源的 FileAsset、缓存和站点投影
        # 1) FLS：删除由已移除 FLS 目录生成的全部 PLY 资产
        for asset in FileRepo.list_assets_by_kind(project_uuid, FileKind.raw_pointcloud):
            meta = dict(asset.meta_json or {})
            fls_src = meta.get('fls_source_dir')
            if fls_src and fls_src not in fls:
                station = PointCloudStationRepo.get_by_asset_id(project_uuid, asset.id)
                if station is not None:
                    proxy_cache.delete_station_cache(project_uuid, station.id)
                FileRepo.delete_file(project_uuid, asset.id)
                continue
            # 2) 普通点云：不在新列表中的直接删除
            if asset.path not in pcs and not fls_src:
                station = PointCloudStationRepo.get_by_asset_id(project_uuid, asset.id)
                if station is not None:
                    proxy_cache.delete_station_cache(project_uuid, station.id)
                FileRepo.delete_file(project_uuid, asset.id)
        # 3) 照片
        for asset in FileRepo.list_assets_by_kind(project_uuid, FileKind.raw_image):
            if asset.path not in photos:
                FileRepo.delete_file(project_uuid, asset.id)

        # 导入阶段：顺序执行，避免并发竞争
        # 1) FLS 目录（实际转换 → PLY → 注册）
        new_fls = [d for d in fls if d not in old_fls]
        for d in new_fls:
            res = self.import_fls_directory(d, project_uuid)
            if not res.get('success'):
                raise RuntimeError(f"FLS 导入失败：{res.get('message', d)}")

        # 2) E57 / 原生 PLY
        existing_pcs = {a.path for a in FileRepo.list_assets_by_kind(
            project_uuid, FileKind.raw_pointcloud)}
        for p in pcs:
            if p in existing_pcs:
                continue
            asset = FileRepo.import_file(
                project_uuid, p, FileKind.raw_pointcloud,
                copy_into_project=False)
            if Path(p).suffix.lower() == '.e57':
                ensure_e57_cache(project_uuid, asset)

        # 3) 照片
        existing_photos = {a.path for a in FileRepo.list_assets_by_kind(
            project_uuid, FileKind.raw_image)}
        for p in photos:
            if p not in existing_photos:
                FileRepo.import_file(
                    project_uuid, p, FileKind.raw_image,
                    copy_into_project=False)

        # 同步站点投影
        PointCloudStationRepo.sync_assets(project_uuid)

        # 持久化 pcfd 索引
        root = Storage.project_root(project_uuid)
        index = Storage.load_pcfd_index(root) or {}
        assets = index.setdefault('assets', {})
        assets['fls_folders'] = fls
        assets['raw_pointclouds'] = pcs
        assets['raw_images'] = photos
        Storage.save_pcfd_index(root, index)

    # -------------- 文件导入 --------------
    def _ensure_file_service(self):
        if self._file_service is None:
            if self.viewport is None or self.render_service is None:
                raise RuntimeError("文件依赖未就绪：视口缺失。")
            self._file_service = FileService(self.viewport, self.db, self.render_service)
        return self._file_service

    def rename_project(self, project_id, new_name):
        """持久化修改项目名称，并同步项目索引文件。"""
        normalized_name = (new_name or '').strip()
        if not normalized_name:
            raise ValueError('项目名称不能为空。')

        info = ProjectRepo.update_project(project_id, name=normalized_name)
        if info is None:
            raise ValueError('项目不存在或已被删除。')

        # list_projects() 优先读取 pcfd 索引中的名称，因此需要同步更新。
        try:
            from config.storage import Storage

            root = Path(info['root_dir'])
            index_data = Storage.load_pcfd_index(root) or {}
            project_data = index_data.setdefault('project', {})
            project_data['name_cn'] = normalized_name
            Storage.save_pcfd_index(root, index_data)
        except Exception:
            # 数据库与全局索引已经更新；旧项目缺少 pcfd 时仍可继续使用。
            pass

        return ProjectCard(
            project_id=info["project_id"], name=info["name"],
            directory_path=info["directory_path"],
            org_unit=info.get("org_unit"), address=info.get("address"),
            building_floor=info.get("building_floor"), remarks=info.get("remarks"),
            construction_unit=info.get("construction_unit"),
            construction_unit_executor=info.get("construction_unit_executor"),
            inspection_unit=info.get("inspection_unit"),
            supervision_unit=info.get("supervision_unit"),
            client_unit=info.get("client_unit"),
            report_no=info.get("report_no"),
            inspection_date=info.get("inspection_date"),
            report_date=info.get("report_date"),
            inspection_params_json=info.get("inspection_params_json"),
        )

    def prepare_upload_files(self, file_paths: list[str], project_uuid: Optional[str],
                             *, worker=None):
        """批量上传计算段：逐文件解析/注册，全程不触碰 Open3D 视口。

        返回 (prepared, uploaded)；prepared 由 commit_prepared_uploads 在
        GUI 线程提交。增量语义：不清空当前场景，已有运行时 dataset 保持不变。
        worker 提供时逐文件回报进度并响应取消。
        """
        if not project_uuid:
            raise ValueError('请先新建或选择项目，再上传点云文件。')
        svc = self._ensure_file_service()
        normalized = [str(Path(p).expanduser().resolve()) for p in file_paths if p]
        # 将同名 PLY/.dist 组合成一个上传任务；.dist 不是独立点云资产。
        dist_by_stem = {
            Path(p).stem.lower(): p for p in normalized
            if Path(p).suffix.lower() == '.dist'
        }
        pointclouds = [p for p in normalized if Path(p).suffix.lower() != '.dist']
        total = len(pointclouds)
        prepared = []
        uploaded: list[str] = []
        for index, p in enumerate(pointclouds):
            if worker is not None:
                worker.check_cancelled()
                worker.signals.progress.emit(
                    int(10 + 80 * index / max(total, 1)), f'正在解析 {Path(p).name}')
            try:
                item = svc.prepare_upload(
                    project_uuid=project_uuid, file_path=p,
                    distance_path=dist_by_stem.get(Path(p).stem.lower()),
                    copy_into_project=False)
                prepared.append(item)
                uploaded.append(p)
            except Exception as e:
                print(f"上传失败: {p} -> {e}", flush=True)
        # FileService 在渲染前会将 FileAsset 持久化。
        PointCloudStationRepo.sync_assets(project_uuid)
        return prepared, uploaded

    def commit_prepared_uploads(self, prepared) -> None:
        """批量上传提交段：Open3D 几何提交与视口状态写入，必须在 GUI 线程。"""
        svc = self._ensure_file_service()
        for item in prepared or []:
            svc.commit_prepared(item)

    def import_fls_directory(self, dir_path: str, project_uuid: Optional[str],
                             *, progress_cb=None) -> dict:
        svc = self._ensure_file_service()
        # FLS 也采用增量导入语义，避免已有站点被清空后重新读取。
        res = svc.import_fls_directory(dir_path, project_uuid, progress_cb=progress_cb)
        # FLS 导入会为每个生成的 PLY 文件创建一个 FileAsset。
        if project_uuid and res.get('success'):
            PointCloudStationRepo.sync_assets(project_uuid)
        return res