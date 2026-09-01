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


class ProjectOverviewService:
    """项目概览页调度器：供 MainWindow 直接调用。"""

    def __init__(self, viewport=None, render_service: ViewportRenderService | None = None, db=None):
        self.viewport = viewport
        self.render_service = render_service
        self.db = db
        self._file_service: Optional[FileService] = None

    def create_load_worker(self, operation: str, project_uuid: str,
                           *, file_paths=None, directory=None):
        def run(worker):
            if operation == 'activate':
                worker.signals.progress.emit(10, '正在读取项目资源')
                worker.check_cancelled()
                self.activate_project(project_uuid)
                return {'operation': operation, 'project_uuid': project_uuid}
            if operation == 'upload':
                paths = list(file_paths or [])
                worker.signals.progress.emit(10, f'准备加载 {len(paths)} 个文件')
                worker.check_cancelled()
                result = self.upload_files(paths, project_uuid)
                return {'operation': operation, 'project_uuid': project_uuid,
                        'uploaded': result}
            if operation == 'fls':
                worker.signals.progress.emit(10, '正在转换 FLS 目录')
                worker.check_cancelled()
                result = self.import_fls_directory(directory, project_uuid)
                return {'operation': operation, 'project_uuid': project_uuid,
                        'result': result}
            raise ValueError(f'未知加载操作: {operation}')
        return PointCloudLoadWorker(run)

    # -------------- 项目管理 --------------
    def list_projects(self) -> list[ProjectCard]:
        items = ProjectRepo.list_projects()
        return [ProjectCard(
            project_id=i["project_id"], name=i["name"],
            directory_path=i["directory_path"],
            org_unit=i.get("org_unit"), address=i.get("address"),
            building_floor=i.get("building_floor"), remarks=i.get("remarks"),
        ) for i in items]

    def create_project(
        self,
        name: str,
        org_unit: str | None = None,
        address: str | None = None,
        remarks: str | None = None,
        building_floor: str | None = None,
    ) -> ProjectCard:
        """创建持久化项目，并向 UI 返回统一的项目卡片模型。"""
        info = ProjectRepo.create_project(
            name=name,
            org_unit=org_unit,
            address=address,
            remarks=remarks,
            building_floor=building_floor,
        )
        return ProjectCard(
            project_id=info["project_uuid"],
            name=info["name"],
            directory_path=info.get("root_dir", info.get("directory_path", "")),
            org_unit=info.get("org_unit"),
            address=info.get("address"),
            building_floor=info.get("building_floor"),
            remarks=info.get("remarks"),
        )

    def open_project(self, directory_path: str) -> ProjectCard:
        path = Path(directory_path).expanduser().resolve()
        if self.viewport is not None and hasattr(self.viewport, 'clear'):
            self.viewport.clear()
        # 如果该文件夹中存在 pcfd 索引，则采用该索引并将其更新或插入到索引数据库中，然后加载资源
        try:
            from config.storage import Storage
            from db.connection import upsert_index_project
            idx = Storage.load_pcfd_index(path)
            if idx is None:
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
        )
        # 新登记项目：登记目录中的全部 PLY；实际代理构建与渲染由
        # StationService 在项目激活后按需完成，避免产生第二条加载链路。
        try:
            ply_candidates = list(path.glob("*.ply"))
            for ply_path in ply_candidates:
                FileRepo.import_file(project_uuid=pc.project_id,
                                     src_path=str(ply_path),
                                     kind=FileKind.raw_pointcloud,
                                     copy_into_project=False)
            if ply_candidates:
                PointCloudStationRepo.sync_assets(pc.project_id)
        except Exception:
            pass
        return pc

    def activate_project(self, project_id: str) -> None:
        """Validate and activate a project without hiding restore failures."""
        if not project_id:
            raise ValueError('项目标识为空，无法激活项目')
        ProjectRepo.load_and_activate(project_id)
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
                                'dataset_revision': row.dataset_revision}
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

    def remove_project(self, project_id: str) -> bool:
        return ProjectRepo.delete_project(project_id, hard=True)

    def get_project(self, project_id: str) -> Optional[ProjectCard]:
        for p in self.list_projects():
            if p.project_id == project_id:
                return p
        return None

    def update_project(self, project_id: str, **fields) -> ProjectCard:
        info = ProjectRepo.update_project(project_id, **fields)
        if info is None:
            raise ValueError('项目不存在或已被删除。')
        return ProjectCard(
            project_id=info["project_id"], name=info["name"],
            directory_path=info["directory_path"],
            org_unit=info.get("org_unit"), address=info.get("address"),
            building_floor=info.get("building_floor"), remarks=info.get("remarks"),
        )

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
        )

    def upload_files(self, file_paths: list[str], project_uuid: Optional[str]) -> list[str]:
        if not project_uuid:
            raise ValueError('请先新建或选择项目，再上传点云文件。')
        svc = self._ensure_file_service()
        # 增量上传不能清空当前场景；项目切换由 MainWindow 的统一销毁门
        # 处理，当前方法只负责导入本批资源。
        normalized = [str(Path(p).expanduser().resolve()) for p in file_paths if p]
        # 将同名 PLY/.dist 组合成一个上传任务；.dist 不是独立点云资产。
        dist_by_stem = {
            Path(p).stem.lower(): p for p in normalized
            if Path(p).suffix.lower() == '.dist'
        }
        pointclouds = [p for p in normalized if Path(p).suffix.lower() != '.dist']
        uploaded: list[str] = []
        for p in pointclouds:
            try:
                dist_path = dist_by_stem.get(Path(p).stem.lower())
                svc.upload_files(project_uuid=project_uuid, file_path=p,
                                 distance_path=dist_path,
                                 copy_into_project=False)
                uploaded.append(p)
            except Exception as e:
                print(f"上传失败: {p} -> {e}", flush=True)
        # FileService 在渲染前会将 FileAsset 持久化。 
        PointCloudStationRepo.sync_assets(project_uuid)
        return uploaded

    def import_fls_directory(self, dir_path: str, project_uuid: Optional[str]) -> dict:
        svc = self._ensure_file_service()
        # FLS 也采用增量导入语义，避免已有站点被清空后重新读取。
        res = svc.import_fls_directory(dir_path, project_uuid)
        # FLS 导入会为每个生成的 PLY 文件创建一个 FileAsset。
        if project_uuid and res.get('success'):
            PointCloudStationRepo.sync_assets(project_uuid)
        return res
