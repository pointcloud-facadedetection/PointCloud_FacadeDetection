from __future__ import annotations

import hashlib
import mimetypes
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from models.files import FileRecord
from algorithms.geometry import voxel_downsample


class FileService:
    """
    Service: Orchestrate file upload and visualization.
    Pattern: Get data -> Run algorithm -> Write result -> Notify render
    """

    SUPPORTED_POINT_CLOUD_EXT = {'.ply', '.pcd', '.xyz', '.xyzn', '.xyzrgb', '.pts'}
    SUPPORTED_IMAGE_EXT = {'.png', '.jpg', '.jpeg', '.bmp'}

    def __init__(self, viewport, db, render_service):
        self.viewport = viewport
        self.db = db  # session factory (callable)
        self.render_service = render_service

    # UI facing method
    def upload_files(self, file_path: str, voxel_size: float = 0.02, project_id: int = 1):
        # Step 1: Get data (read file + sniff type)
        file_path = str(Path(file_path).resolve())
        suffix = Path(file_path).suffix.lower()
        file_type = self._infer_type_from_suffix(suffix)

        # Step 2: Run algorithm (if point cloud -> downsample; if image -> passthrough)
        if file_type == 'pointcloud':
            pts, cols = self._load_point_cloud(file_path)
            pts_ds, cols_ds = voxel_downsample(pts, cols, voxel_size=voxel_size)
            algo_payload = ('pointcloud', pts_ds, cols_ds)
        elif file_type == 'image':
            img = self._load_image(file_path)
            algo_payload = ('image', img)
        else:
            raise ValueError(f'Unsupported file type for: {file_path}')

        # Step 3: Write result (persist FileRecord metadata)
        st = os.stat(file_path)
        sha256 = self._file_sha256(file_path)
        with self.db() as session:  # type: ignore
            rec = FileRecord(
                project_id=project_id,
                file_type=file_type,
                path=file_path,
                original_name=os.path.basename(file_path),
                size_bytes=st.st_size,
                sha256=sha256,
                meta_json={"voxel_size": voxel_size} if file_type == 'pointcloud' else None,
            )
            session.add(rec)
            session.commit()

        # Step 4: Notify render
        if algo_payload[0] == 'pointcloud':
            _, pts_ds, cols_ds = algo_payload
            name = os.path.basename(file_path)
            self.render_service.show_point_cloud(name=name, points=pts_ds, colors=cols_ds)
        else:
            _, img = algo_payload
            name = os.path.basename(file_path)
            self.render_service.show_image(name=name, image=img)

    def _infer_type_from_suffix(self, suffix: str) -> str:
        if suffix in self.SUPPORTED_POINT_CLOUD_EXT:
            return 'pointcloud'
        if suffix in self.SUPPORTED_IMAGE_EXT:
            return 'image'
        # Fallback using mimetypes
        mime, _ = mimetypes.guess_type('x' + suffix)
        if mime and mime.startswith('image/'):
            return 'image'
        raise ValueError(f'Unsupported file extension: {suffix}')

    def _file_sha256(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()

    def _load_point_cloud(self, path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points, dtype=np.float64)
        cols = None
        if pcd.has_colors():
            cols = np.asarray(pcd.colors, dtype=np.float64)
        return pts, cols

    def _load_image(self, path: str) -> np.ndarray:
        # Use Open3D to keep dependencies consistent
        import open3d as o3d

        img = o3d.io.read_image(path)
        arr = np.asarray(img)
        # Ensure RGB if grayscale
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = arr.astype(np.float64)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return arr
