from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from services.dal.project_repo import ProjectRepo
from services.file_service import FileService
from services.dal.file_repo import FileRepo
from services.viewport_render_service import ViewportRenderService


@dataclass
class ProjectCard:
    """供 UI 卡片渲染的最小数据。"""
    project_id: str
    name: str
    directory_path: str


class ProjectOverviewService:
    """项目概览页调度器：供 MainWindow 直接调用。"""

    def __init__(self, viewport=None, render_service: ViewportRenderService | None = None, db=None):
        self.viewport = viewport
        self.render_service = render_service
        self.db = db
        self._file_service: Optional[FileService] = None

    # -------------- 项目管理 --------------
    def list_projects(self) -> list[ProjectCard]:
        items = ProjectRepo.list_projects()
        return [ProjectCard(project_id=i["project_uuid"], name=i["name"], directory_path=i["root_dir"]) for i in items]

    def create_project(
        self,
        name: str,
        org_unit: str | None = None,
        address: str | None = None,
        remarks: str | None = None,
    ) -> ProjectCard:
        """创建持久化项目，并向 UI 返回统一的项目卡片模型。"""
        info = ProjectRepo.create_project(
            name=name,
            org_unit=org_unit,
            address=address,
            remarks=remarks,
        )
        return ProjectCard(
            project_id=info["project_uuid"],
            name=info["name"],
            directory_path=info["root_dir"],
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
                    # 尝试通过 PCFD 资源或 DAL 加载最新的点云数据
                    try:
                        assets = idx.get('assets') or {}
                        cand = None
                        # 在项目中优先使用 generated_ply
                        gp = assets.get('generated_ply') or []
                        if gp:
                            cand = Path(path) / gp[-1]
                        else:
                            raws = assets.get('raw_pointclouds') or []
                            if raws:
                                cand = Path(raws[-1])
                        if cand is not None and self.render_service is not None:
                            svc = self._ensure_file_service()
                            svc.upload_files(project_uuid=puid, file_path=str(cand), copy_into_project=False)
                    except Exception:
                        # fallback to DAL helper
                        try:
                            asset = FileRepo.get_latest_raw_pointcloud(puid)
                            if asset is not None and self.render_service is not None:
                                svc = self._ensure_file_service()
                                svc.upload_files(project_uuid=puid, file_path=asset.path, copy_into_project=False)
                        except Exception:
                            pass
                    return ProjectCard(project_id=puid, name=pname, directory_path=str(path))
        except Exception:
            pass
        # 尝试匹配已登记项目
        for p in self.list_projects():
            if Path(p.directory_path).resolve() == path:
                # 激活场景（若需要）
                ProjectRepo.load_and_activate(p.project_id)
                # 尝试加载最新原始点云到视口
                try:
                    asset = FileRepo.get_latest_raw_pointcloud(p.project_id)
                    if asset is not None and self.render_service is not None:
                        # 直接读取并渲染（保持与 upload_files 一致的体验）
                        svc = self._ensure_file_service()
                        svc.upload_files(project_uuid=p.project_id, file_path=asset.path, copy_into_project=False)
                except Exception:
                    pass
                return p
        # 未登记则创建新项目（名称取目录名）
        info = ProjectRepo.create_project(name=path.name or "未命名项目")
        pc = ProjectCard(
            project_id=info["project_uuid"],
            name=info["name"],
            directory_path=info["root_dir"],
        )
        # 新登记项目：若目录中存在 PLY 文件，录入并渲染
        try:
            ply_candidates = list(path.glob("*.ply"))
            if ply_candidates:
                svc = self._ensure_file_service()
                svc.upload_files(project_uuid=pc.project_id, file_path=str(ply_candidates[0]), copy_into_project=False)
        except Exception:
            pass
        return pc

    def activate_project(self, project_id: str) -> None:
        # 项目切换必须先清空旧场景，否则 add_cloud() 会认为已有点云，
        # 不会建立新项目默认视角，Open3D 还可能沿用旧 bounding box。
        if self.viewport is not None and hasattr(self.viewport, 'clear'):
            self.viewport.clear()
        try:
            ProjectRepo.load_and_activate(project_id)
        except Exception:
            return
        try:
            asset = FileRepo.get_latest_raw_pointcloud(project_id)
            if asset is not None and self.render_service is not None:
                svc = self._ensure_file_service()
                # 请勿将原始文件复制到项目缓存中；原始文件的绝对路径已保存在数据库中
                svc.upload_files(project_uuid=project_id, file_path=asset.path, copy_into_project=False)
        except Exception:
            pass

    def remove_project(self, project_id: str) -> bool:
        return ProjectRepo.delete_project(project_id, hard=False)

    def get_project(self, project_id: str) -> Optional[ProjectCard]:
        for p in self.list_projects():
            if p.project_id == project_id:
                return p
        return None

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
            project_id=info['project_uuid'],
            name=info['name'],
            directory_path=info['root_dir'],
        )

    def upload_files(self, file_paths: list[str], project_uuid: Optional[str]) -> list[str]:
        svc = self._ensure_file_service()
        # 这是项目首次载入入口；避免旧项目 geometry 影响 Open3D 初始 fit。
        if self.viewport is not None and hasattr(self.viewport, 'clear'):
            self.viewport.clear()
        normalized = [str(Path(p).expanduser().resolve()) for p in file_paths if p]
        uploaded: list[str] = []
        for p in normalized:
            try:
                svc.upload_files(project_uuid=project_uuid, file_path=p, copy_into_project=False)
                uploaded.append(p)
            except Exception as e:
                print(f"上传失败: {p} -> {e}", flush=True)
        return uploaded

    def import_fls_directory(self, dir_path: str, project_uuid: Optional[str]) -> dict:
        svc = self._ensure_file_service()
        if self.viewport is not None and hasattr(self.viewport, 'clear'):
            self.viewport.clear()
        res = svc.import_fls_directory(dir_path, project_uuid)
        return res
