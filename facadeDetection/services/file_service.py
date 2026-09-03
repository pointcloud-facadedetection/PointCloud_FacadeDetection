from __future__ import annotations

import hashlib
import json
import mimetypes
import os
from pathlib import Path
from typing import Optional, Tuple
import open3d as o3d

import numpy as np
import time

from algorithms.preprocess import voxel_downsample
from models import FileAsset
from models.enums import FileKind
from services.dal.file_repo import FileRepo
from config.storage import Storage

from utils.convert_fls2ply import convert_fls_to_ply
from utils.dist_reader import read_dist
from utils.logging_utils import log_event, trace
from algorithms.geometry import stratified_proxy_build


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
        self.pointcloud_service = getattr(render_service, "pointcloud_service", None)

    # Public API
    def upload_files(
        self,
        project_uuid: Optional[str],
        file_path: str,
        *,
        voxel_size: float = 0.05,
        copy_into_project: bool = False,
        dataset_metadata: Optional[dict] = None,
        distance_path: Optional[str] = None,
    ) -> Optional[FileAsset]:
        # TODO(性能/内存): upload_files：read_point_cloud、距离解析、代理构建和 dataset 注册会同时持有大数组，需后台化并设计分阶段释放/内存预算。
        """
        上传一个文件（点云或图片）并进行渲染。
        """
        src = Path(file_path).resolve()
        if not src.exists():
            raise FileNotFoundError(f"未找到文件: {src}")

        kind = self._guess_file_kind(src.suffix.lower())
        if kind is None:
            raise ValueError(f"不支持的文件格式: {src.suffix}")

        asset = None
        if project_uuid:
            asset = FileRepo.import_file(
                project_uuid=project_uuid,
                src_path=str(src),
                kind=kind,
                copy_into_project=copy_into_project,
            )

        load_path = Path(asset.path) if asset else src
        if kind == FileKind.raw_pointcloud:
            started = time.perf_counter()
            print(f"[PCFD] load.begin path={load_path}", flush=True)
            pts, cols = self._load_point_cloud(str(load_path))
            print(f"[PCFD] load.read points={len(pts)} seconds={time.perf_counter()-started:.2f}", flush=True)

            # 直接上传 PLY + 同名 .dist 时，预处理发生在 register_dataset 之前。
            # 未提供 .dist 的普通 PLY 继续走旧路径，保证历史导入行为不变。
            dist_file = Path(distance_path).resolve() if distance_path else load_path.with_suffix('.dist')
            dist_exists = dist_file.exists()
            print(f"[PCFD] load.dist status={('found' if dist_exists else 'not_found')} path={dist_file}", flush=True)
            if dist_exists and self.pointcloud_service is not None:
                dist = read_dist(dist_file, pts, dataset_metadata or {})
                print(
                    f"[PCFD] load.dist loaded source={dist.source} "
                    f"points={len(pts)} origins={len(dist.scan_origins)} "
                    f"warnings={dist.warnings}", flush=True)

                # 加载阶段仅执行距离分层下采样，不执行自适应去噪；
                # 去噪由用户手动点击按钮触发。
                elevations = None
                if len(dist.scan_origins):
                    from algorithms.geometry import estimate_elevation_angles
                    elevations = estimate_elevation_angles(pts, dist.scan_origins)

                proc_pts, proc_cols, proxy_offsets, proxy_indices, proc_ranges = stratified_proxy_build(
                    pts, cols,
                    dist.ranges_m,
                    scan_origin=dist.scan_origins if len(dist.scan_origins) else None,
                    elevations=elevations)

                source_id = (dataset_metadata or {}).get(
                    'source_id', f"{project_uuid or 'local'}:source:{load_path.stem}")

                # metadata 中记录实际使用的 shells 参数
                shells_used = ((10., .10), (20., .08), (35., .06),
                                  (50., .05), (80., .045), (100., .04))
                metadata = dict(dataset_metadata or {})
                metadata.update({
                    'source_id': source_id,
                    'source_raw_count': int(len(pts)),
                    'proxy_source_offsets': proxy_offsets.tolist(),
                    'proxy_source_indices': proxy_indices.tolist(),
                    'ranges': proc_ranges.tolist(),
                    'scan_origins': dist.scan_origins.tolist(),
                    'distance_source': dist.source,
                    'distance_warnings': dist.warnings,
                    'preprocess': {
                    'input_count': int(len(pts)),
                        'output_count': int(len(proc_pts)),
                        'shells': [[float(h), float(v)] for h, v in shells_used],
                    },
                    'adaptive_detection': {
                        'enabled': True, 'range_coeff': .0012,
                        'normal_relax_deg_per_m': .15,
                        'normal_angle_max_deg': 15., 'irls_iters': 2,
                    },
                })

                # === 先注册 source asset，再注册 dataset ===
                self.pointcloud_service.register_source_asset(
                    source_id, pts, cols,
                    {'ply_path': str(load_path), 'dist_path': str(dist_file)})

                dataset_id = f"{project_uuid or 'local'}:{load_path.name}"
                dataset = self.pointcloud_service.register_dataset(
                    dataset_id, proc_pts, proc_cols, metadata=metadata)

                pts_ds, cols_ds = dataset.proxy_points, dataset.proxy_colors
                print(f"[PCFD] load.range_adaptive dist={dist_file.name} "
                      f"source={len(pts)} proxy={len(pts_ds)}",
                      flush=True)

            elif self.pointcloud_service is not None:
                dataset_id = f"{project_uuid or 'local'}:{load_path.name}"
                dataset = self.pointcloud_service.register_dataset(
                    dataset_id, pts, cols, metadata=dataset_metadata)
                pts_ds, cols_ds = dataset.proxy_points, dataset.proxy_colors
                print(f"[PCFD] load.index_ready proxy={len(pts_ds)} raw={len(pts)} "
                      f"seconds={time.perf_counter()-started:.2f}", flush=True)
            else:
                pts_ds, cols_ds = voxel_downsample(pts, cols, voxel_size=voxel_size)

            name = asset.original_name if asset else src.name
            self.render_service.show_point_cloud(name=name, points=pts_ds, colors=cols_ds)
            print(f"[PCFD] load.render_done displayed={len(pts_ds)} "
                  f"seconds={time.perf_counter()-started:.2f}", flush=True)

            data = self.viewport.get_cloud_data(name) if hasattr(self.viewport, "get_cloud_data") else None
            if data is not None and self.pointcloud_service is not None:
                data["dataset_id"] = dataset_id
                data["domain"] = "proxy"
                data["index_space"] = "proxy_global"
                data["is_processing_cloud"] = True
                data["proxy_ids"] = np.arange(len(pts_ds), dtype=np.int32)
                print(f"[PCFD] cloud.bound cloud={name} dataset={dataset_id} "
                      f"proxy={len(pts_ds)}", flush=True)

            # 更新 pcfd 索引资产
            try:
                if project_uuid:
                    rel = os.path.relpath(str(load_path),
                                          start=str(Storage.resolve_project_root(project_uuid)))
                    if not rel.startswith('..') and not os.path.isabs(rel):
                        Storage.append_pcfd_asset_for_uuid(project_uuid, 'generated_ply',
                                                           rel.replace('\\', '/'))
                    else:
                        Storage.append_pcfd_asset_for_uuid(project_uuid, 'raw_pointclouds',
                                                           str(load_path))
            except Exception:
                pass
        elif kind == FileKind.raw_image:
            img = self._load_image(str(load_path))
            name = asset.original_name if asset else src.name
            self.render_service.show_image(name=name, image=img)
            try:
                if project_uuid:
                    Storage.append_pcfd_asset_for_uuid(project_uuid, 'raw_images', str(load_path))
            except Exception:
                pass
        else:
            # Should not happen because we only support those two kinds
            raise RuntimeError(f"无法处理的文件类型: {kind}")

        return asset

    def import_fls_directory(self, dir_path: str, project_uuid: Optional[str]) -> dict:
        # TODO(性能/响应性): import_fls_directory：转换器调用及逐个 PLY 读取/处理。
        """
        通过基于子进程的转换器将 FARO FLS 目录转换为 PLY 格式进行导入，随后通过
        相同的 upload_files 管道保存并渲染生成的 PLY 文件，以确保一致性。
        """
        trace('fls.import.begin', path=dir_path, project_uuid=project_uuid)
        log_event(project_uuid, 'fls.import.begin', path=dir_path)
        Storage.ensure_base_dirs()
        src = Path(dir_path).resolve()
        if not src.exists() or not src.is_dir():
            msg = f"FLS 目录不存在: {src}"
            trace('fls.import.invalid_path', path=src)
            log_event(project_uuid, 'fls.import.failed', reason=msg)
            return {"success": False, "message": msg, "ply_paths": [], "output_dir": ""}

        if project_uuid:
            project_root = Storage.project_root(project_uuid)
            out_dir = project_root / Storage.CACHE_DIRNAME
        else:
            out_dir = Storage.DATA_DIR / "tmp_fls_import"
        out_dir.mkdir(parents=True, exist_ok=True)

        project_name = src.name

        # 调用转换器(pybind)将 FLS 转为 PLY
        try:
            result = convert_fls_to_ply(
                fls_folder=str(src), output_dir=str(out_dir),
                project_name=project_name
            )
        except Exception as e:
            msg = f"FLS 转换器调用失败: {e}"
            trace('fls.import.convert_failed', error=e)
            log_event(project_uuid, 'fls.import.failed', reason=msg)
            return {"success": False, "message": msg, "ply_paths": [], "output_dir": str(out_dir)}

        # Collect PLY and station metadata
        ply_paths = []
        stations_meta = None

        search_dir = Path(getattr(result, 'output_dir', str(out_dir)) or out_dir)
        for p in search_dir.rglob("*.ply"):
            ply_paths.append(str(p))
            meta_path = p.with_suffix('').with_suffix('.stations.json')
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    stations_meta = json.load(f)

        if not ply_paths:
            trace('fls.import.no_ply', output=search_dir)
            log_event(project_uuid, 'fls.import.failed', reason='未生成 PLY', output=str(search_dir))
            return {"success": False, "message": "未生成 PLY"}

        # 导入并注册
        success_count = 0
        imported_metadata = []
        for p in ply_paths:
            try:
                # FLS 批量结果必须逐站写入项目 FileAsset；否则视口虽有多个
                # cloud，项目站点域无法发现第二站点，重开项目也会丢失。
                asset = None
                if project_uuid:
                    asset = FileRepo.import_file(
                        project_uuid=project_uuid,
                        src_path=p,
                        kind=FileKind.raw_pointcloud,
                        copy_into_project=False,
                    )
                source_pts, source_cols = self._load_point_cloud(p)
                scan_meta = next((s for s in getattr(result, 'scans', [])
                                  if str(getattr(s, 'ply_path', '')) == str(Path(p).resolve())), None)

                dist_path = Path(p).with_suffix('.dist')
                dist_exists = dist_path.exists()
                trace('fls.import.dist', status=('found' if dist_exists else 'not_found'), path=dist_path)

                dist = read_dist(dist_path if dist_exists else None, source_pts, scan_meta)
                trace('fls.import.dist_loaded', source=dist.source,
                      points=len(source_pts), origins=len(dist.scan_origins),
                      warnings=dist.warnings)

                # FLS 加载阶段仅做距离分层下采样，不自动去噪
                elevations = None
                if len(dist.scan_origins):
                    from algorithms.geometry import estimate_elevation_angles
                    elevations = estimate_elevation_angles(source_pts, dist.scan_origins)

                # 使用修复后的 stratified_downsample
                proc_pts, proc_cols, proxy_offsets, proxy_indices, proc_ranges = stratified_proxy_build(
                    source_pts, source_cols,
                    dist.ranges_m,
                    scan_origin=dist.scan_origins if len(dist.scan_origins) else None,
                    elevations=elevations)

                source_id = f"{project_uuid or 'local'}:source:{Path(p).stem}"

                #  metadata 中记录实际 shells 参数
                shells_used = ((10., .10), (20., .08), (35., .06),
                              (50., .05), (80., .045), (100., .04))
                metadata = {
                    'source_id': source_id,
                    'source_raw_count': int(len(source_pts)),
                    'proxy_source_offsets': proxy_offsets.tolist(),
                    'proxy_source_indices': proxy_indices.tolist(),
                    'ranges': proc_ranges.tolist(),
                    'scan_origins': dist.scan_origins.tolist(),
                    'distance_source': dist.source,
                    'distance_warnings': dist.warnings,
                    'preprocess': {
                        'input_count': int(len(source_pts)),
                        'output_count': int(len(proc_pts)),
                        'shells': [[float(h), float(v)] for h, v in shells_used],
                    },
                    'adaptive_detection': {
                        'enabled': True, 'range_coeff': .0012,
                        'normal_relax_deg_per_m': .15,
                        'normal_angle_max_deg': 15., 'irls_iters': 2},
                }

                # ===  先注册 source asset ===
                if self.pointcloud_service is not None:
                    self.pointcloud_service.register_source_asset(
                        source_id, source_pts, source_cols,
                        {'ply_path': str(Path(p).resolve()), 'dist_path': str(dist_path)})

                dataset_id = f"{project_uuid or 'local'}:{Path(p).name}"

                if self.pointcloud_service is not None:
                    self.pointcloud_service.register_dataset(
                        dataset_id, proc_pts, proc_cols, metadata=metadata)
                    dataset = self.pointcloud_service.get_dataset(dataset_id)
                    trace('fls.import.dataset_bound', cloud=Path(p).name,
                          dataset=dataset_id, proxy=len(dataset.proxy_points))
                else:
                    proc_pts, proc_cols = voxel_downsample(proc_pts, proc_cols, voxel_size=0.05)

                imported_metadata.append(metadata)
                try:
                    if project_uuid:
                        Storage.append_pcfd_asset_for_uuid(project_uuid, 'fls_folders', str(src))
                except Exception:
                    pass
                success_count += 1

            except Exception as e:
                trace('fls.import.ply_failed', path=p, error=e)
                log_event(project_uuid, 'fls.import.ply_failed', path=p, error=str(e))

        trace('fls.import.done', uploaded=success_count, total=len(ply_paths), output=search_dir)
        log_event(project_uuid, 'fls.import.done', uploaded=success_count,
                  total=len(ply_paths), output=str(search_dir))
        return {
            "success": success_count > 0,
            "message": getattr(result, 'message', ''),
            "output_dir": str(search_dir),
            "ply_paths": ply_paths,
            "uploaded": success_count,
            "metadata": imported_metadata,
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
        try:
            pts = np.asarray(pcd.points, dtype=np.float32).reshape(-1, 3)
            pts = pts if pts.flags.c_contiguous else np.ascontiguousarray(pts)
            cols = None
            if pcd.has_colors():
                candidate = np.asarray(pcd.colors, dtype=np.float32).reshape(-1, 3)
                if len(candidate) == len(pts):
                    cols = candidate if candidate.flags.c_contiguous else np.ascontiguousarray(candidate)
            return pts, cols
        finally:
            del pcd

    def _load_image(self, path: str) -> np.ndarray:
        img = o3d.io.read_image(path)
        arr = np.asarray(img)
        if arr.ndim == 2:               # grayscale -> RGB
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = arr.astype(np.float32, copy=False)
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