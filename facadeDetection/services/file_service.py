from __future__ import annotations

import hashlib
import mimetypes
import os
from pathlib import Path
from typing import Optional, Tuple, Union
import open3d as o3d

import numpy as np

from algorithms.preprocess import voxel_downsample
from models import FileAsset
from models.enums import FileKind
from services.dal.file_repo import FileRepo
from config.storage import Storage

# Reuse the existing converter implementation as the official reference
from utils.convert_fls2ply import convert_fls_to_ply, EXE_PATH as DEFAULT_FLS_EXE


class FileService:
    """
    服务：协调文件上传和可视化。
    统一入口：upload_files() 处理点云和图像文件
    提供 project_uuid通过 FileRepo 保存，并触发渲染
    """

    SUPPORTED_POINT_CLOUD_EXT = {'.ply', '.pcd', '.xyz', '.xyzn', '.xyzrgb', '.pts'}
    SUPPORTED_IMAGE_EXT = {'.png', '.jpg', '.jpeg', '.bmp'}

    def __init__(self, viewport, db, render_service):
        self.viewport = viewport
        self.db = db          # session factory (callable) – kept for compatibility
        self.render_service = render_service

    # Public API
    def upload_files(
        self,
        project_uuid: Optional[str],
        file_path: str,
        *,
        voxel_size: float = 0.05,
        copy_into_project: bool = False
    ) -> Optional[FileAsset]:
        """
        上传一个文件（点云或图片）并进行渲染。
        
        :param project_uuid: UUID of the target project (can be None for ephemeral loading)
        :param file_path: absolute or relative path to the file
        """
        src = Path(file_path).resolve()
        if not src.exists():
            raise FileNotFoundError(f"未找到文件: {src}")

        # 1. Determine file kind
        kind = self._guess_file_kind(src.suffix.lower())
        if kind is None:
            raise ValueError(f"不支持的文件格式: {src.suffix}")

        # 2. Persist via FileRepo if project is active
        asset = None
        if project_uuid:
            asset = FileRepo.import_file(
                project_uuid=project_uuid,
                src_path=str(src),
                kind=kind,
                copy_into_project=copy_into_project,
            )

        # 3. Load and render the data (use asset path if persisted, otherwise original)
        load_path = Path(asset.path) if asset else src
        if kind == FileKind.raw_pointcloud:
            pts, cols = self._load_point_cloud(str(load_path))
            pts_ds, cols_ds = voxel_downsample(pts, cols, voxel_size=voxel_size)
            name = asset.original_name if asset else src.name
            self.render_service.show_point_cloud(name=name, points=pts_ds, colors=cols_ds)
        elif kind == FileKind.raw_image:
            img = self._load_image(str(load_path))
            name = asset.original_name if asset else src.name
            self.render_service.show_image(name=name, image=img)
        else:
            # Should not happen because we only support those two kinds
            raise RuntimeError(f"无法处理的文件类型: {kind}")

        return asset

    # New: unified FLS import workflow (folder -> convert -> persist -> render)
    def import_fls_directory(self, dir_path: str, project_uuid: Optional[str]) -> dict:
        """
        Import a FARO FLS directory by converting it to PLY via the subprocess-based
        converter, then persist and render the produced PLY files through the
        same upload_files pipeline for cohesion.
        """
        print("[FLS] Import started", flush=True)
        Storage.ensure_base_dirs()
        src = Path(dir_path).resolve()
        if not src.exists() or not src.is_dir():
            msg = f"FLS 目录不存在: {src}"
            print(f"[FLS] {msg}", flush=True)
            return {"success": False, "message": msg, "ply_paths": [], "output_dir": ""}

        # Decide output location
        if project_uuid:
            project_root = Storage.project_root(project_uuid)
            out_dir = project_root / Storage.CACHE_DIRNAME
        else:
            out_dir = Storage.DATA_DIR / "tmp_fls_import"
        out_dir.mkdir(parents=True, exist_ok=True)

        project_name = src.name
        exe_path = os.getenv("FLSREAD_EXE", DEFAULT_FLS_EXE)

        # Run conversion
        try:
            result = convert_fls_to_ply(
                fls_folder=str(src), output_dir=str(out_dir), project_name=project_name,
                exe_path=exe_path, timeout_sec=3600
            )
        except Exception as e:
            msg = f"FLS 转换器调用失败: {e}"
            print(f"[FLS] {msg}", flush=True)
            return {"success": False, "message": msg, "ply_paths": [], "output_dir": str(out_dir)}

        ply_paths: list[str] = []
        if result.success:
            for s in result.scans:
                p = Path(s.ply_path)
                if p.exists():
                    ply_paths.append(str(p))
            if not ply_paths and result.output_dir:
                for p in Path(result.output_dir).rglob("*.ply"):
                    ply_paths.append(str(p))
        else:
            print(f"[FLS] 转换未成功: {result.message}", flush=True)
            return {
                "success": False,
                "message": result.message,
                "ply_paths": [],
                "output_dir": result.output_dir,
            }

        # Persist and render
        success_count = 0
        for p in ply_paths:
            try:
                self.upload_files(
                    project_uuid=project_uuid,
                    file_path=p,
                    copy_into_project=False,
                )
                success_count += 1
            except Exception as e:
                print(f"[FLS] 导入 PLY 失败: {p} -> {e}", flush=True)

        try:
            if hasattr(self.viewport, 'auto_range'):
                self.viewport.auto_range()
        except Exception:
            pass

        print(
            f"[FLS] Import finished: {success_count}/{len(ply_paths)} uploaded",
            flush=True,
        )
        return {
            "success": success_count > 0,
            "message": result.message,
            "output_dir": result.output_dir,
            "ply_paths": ply_paths,
            "uploaded": success_count,
        }

    # ------------------ Helpers ------------------
    def _guess_file_kind(self, suffix: str) -> Optional[FileKind]:
        """Map file extension to FileKind."""
        if suffix in self.SUPPORTED_POINT_CLOUD_EXT:
            return FileKind.raw_pointcloud
        if suffix in self.SUPPORTED_IMAGE_EXT:
            return FileKind.raw_image
        # Fallback via mimetypes
        mime, _ = mimetypes.guess_type('x' + suffix)
        if mime and mime.startswith('image/'):
            return FileKind.raw_image
        return None

    def _load_point_cloud(self, path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points, dtype=np.float64)
        cols = None
        if pcd.has_colors():
            cols = np.asarray(pcd.colors, dtype=np.float64)
        return pts, cols

    def _load_image(self, path: str) -> np.ndarray:
        img = o3d.io.read_image(path)
        arr = np.asarray(img)
        if arr.ndim == 2:               # grayscale -> RGB
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = arr.astype(np.float64)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return arr

    # (Optional) Keep sha256 utility if needed elsewhere
    @staticmethod
    def _file_sha256(path: str) -> str:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()