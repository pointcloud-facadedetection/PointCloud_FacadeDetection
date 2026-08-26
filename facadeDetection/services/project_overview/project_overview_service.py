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

    def create_project(self, name: str, org_unit: str | None = None, address: str | None = None, remarks: str | None = None) -> dict:
        return ProjectRepo.create_project(name=name, org_unit=org_unit, address=address, remarks=remarks)

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
        pc = ProjectCard(project_id=info["project_uuid"], name=info["name"], directory_path=info["root_dir"]) 
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
        except Exception as exc:
            print(f'激活项目失败: {project_id} -> {exc}', flush=True)
            return
        cloud_path = self._find_project_pointcloud_path(project_id)
        if not cloud_path:
            print(f'项目没有可加载的点云: {project_id}', flush=True)
            return
        if self.render_service is None:
            return
        try:
            svc = self._ensure_file_service()
            dist_path = Path(cloud_path).with_suffix('.dist')
            svc.upload_files(
                project_uuid=project_id,
                file_path=cloud_path,
                distance_path=str(dist_path) if dist_path.exists() else None,
                copy_into_project=False,
            )
        except Exception as exc:
            print(f'加载项目点云失败: {cloud_path} -> {exc}', flush=True)

    def _find_project_pointcloud_path(self, project_id: str) -> Optional[str]:
        try:
            asset = FileRepo.get_latest_raw_pointcloud(project_id)
            if asset is not None and Path(asset.path).exists():
                return asset.path
        except Exception:
            pass

        project = self.get_project(project_id)
        root = Path(project.directory_path) if project and project.directory_path else None
        if root is None:
            try:
                from config.storage import Storage
                root = Storage.project_root(project_id)
            except Exception:
                root = None
        if root is None:
            return None

        candidates: list[Path] = []
        try:
            from config.storage import Storage
            idx = Storage.load_pcfd_index(root) or {}
            assets = idx.get('assets') or {}
            for key in ('generated_ply', 'raw_pointclouds'):
                for item in reversed(list(assets.get(key) or [])):
                    path = Path(item)
                    if not path.is_absolute():
                        path = root / path
                    candidates.append(path)
        except Exception:
            pass

        search_roots = [root, root / 'raw', root / 'cache', root / 'results']
        suffixes = {'.ply', '.pcd', '.xyz', '.pts', '.las', '.laz', '.e57'}
        for folder in search_roots:
            if not folder.exists():
                continue
            for path in folder.rglob('*'):
                if path.is_file() and path.suffix.lower() in suffixes:
                    candidates.append(path)

        existing = [path for path in candidates if path.exists() and path.is_file()]
        if not existing:
            return None
        newest = max(existing, key=lambda path: path.stat().st_mtime)
        return str(newest.resolve())

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

    def upload_files(self, file_paths: list[str], project_uuid: Optional[str]) -> list[str]:
        svc = self._ensure_file_service()
        # 这是项目首次载入入口；避免旧项目 geometry 影响 Open3D 初始 fit。
        if self.viewport is not None and hasattr(self.viewport, 'clear'):
            self.viewport.clear()
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
        return uploaded

    def import_fls_directory(self, dir_path: str, project_uuid: Optional[str]) -> dict:
        svc = self._ensure_file_service()
        if self.viewport is not None and hasattr(self.viewport, 'clear'):
            self.viewport.clear()
        res = svc.import_fls_directory(dir_path, project_uuid)
        return res
