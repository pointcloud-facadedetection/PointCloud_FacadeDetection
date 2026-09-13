"""站点源点云导出业务层。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

from algorithms.preprocess import voxel_downsample
from services.dal.pointcloud_station_repo import PointCloudStationRepo
from utils.ply_fast_reader import read_ply_fast
from utils.ply_writer import write_point_cloud_ply_atomic

#: 业务规定的导出下采样体素边长（米）。
DEFAULT_VOXEL_SIZE = 0.2


class ModelExportCancelled(RuntimeError):
    """调用方在导出过程中请求中止；不等于导出失败。"""

#: 进度回调签名：``(percent: int, text: str) -> None``
ProgressCallback = Callable[[int, str], None]


class ModelExportService:
    """把某站点的源点云下采样后导出为 PLY 文件。"""

    def __init__(self, project_uuid: Optional[str] = None):
        self.project_uuid = project_uuid

    def set_project(self, project_uuid):
        self.project_uuid = project_uuid

    # ------------------------------------------------------------------
    # 站点查询
    # ------------------------------------------------------------------
    def list_stations(self, project_uuid: Optional[str] = None):
        """列出可导出的站点行（供 UI 的站点选择对话框使用）。"""
        uuid_value = project_uuid or self.project_uuid
        if not uuid_value:
            return []
        return PointCloudStationRepo.list(uuid_value)

    def get_station(self, station_id, project_uuid: Optional[str] = None):
        """按主键取站点行；不存在时返回 None。"""
        for station in self.list_stations(project_uuid):
            if int(station.id) == int(station_id):
                return station
        return None

    def default_output_name(self, station) -> str:
        """生成默认文件名：``<站点名>_export.ply``。"""
        name = str(getattr(station, 'display_name', '') or 'station').strip()
        safe = ''.join(
            ch if ch not in '\\/:*?"<>|' else '_' for ch in name) or 'station'
        return f'{safe}_voxel{DEFAULT_VOXEL_SIZE:g}.ply'

    # ------------------------------------------------------------------
    # 源点云读取
    # ------------------------------------------------------------------
    def load_source_points(self, station):
        """读取站点源点云，返回 ``(points, colors)``；已全局化，不再施加变换。"""
        path = Path(str(station.source_path))
        if not path.exists():
            raise FileNotFoundError(f'站点源文件不存在：{path}')
        if path.suffix.lower() == '.e57':
            raise RuntimeError('站点源点云必须是运行时 PLY；请重新导入原始 E57')

        # 二进制 PLY 走 memmap 免解析快读；不满足条件时回退 Open3D，
        # 两条路径对同一资产逐点一致（与 PointCloudStationService._load 同语义）。
        fast = read_ply_fast(path)
        if fast is not None:
            return fast
        import open3d as o3d
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points, dtype=np.float64)
        colors = (np.asarray(cloud.colors, dtype=np.float64)
                  if cloud.has_colors() else None)
        return points, colors

    # ------------------------------------------------------------------
    # 导出主流程
    # ------------------------------------------------------------------
    def export_station_ply(self, station_id, target_path,
                           voxel_size: float = DEFAULT_VOXEL_SIZE,
                           project_uuid: Optional[str] = None,
                           progress_cb: Optional[ProgressCallback] = None,
                           cancel_check: Optional[Callable[[], bool]] = None):
        """导出指定站点的下采样点云。

        参数
        ----
        station_id : int
            站点主键（``PointCloudStation`` 的 id）。
        target_path : str | Path
            目标 PLY 路径；父目录会自动创建。
        voxel_size : float
            体素边长（米），默认 0.2。
        project_uuid : str | None
            不传则用构造时注入的项目。
        progress_cb : callable | None
            ``(percent, text)``；百分比为真实进度（按阶段推进）。
        cancel_check : callable | None
            返回 True 表示调用方已请求中止；在阶段边界与写盘前检查，
            命中即抛 :class:`ModelExportCancelled`。写盘走原子替换，
            因此中止不会留下半截目标文件。

        返回
        ----
        dict
            ``{path, source_points, exported_points, voxel_size,
               bounds_min, bounds_max, station_id, station_name}``
        """
        def check_cancelled():
            if cancel_check is not None and cancel_check():
                raise ModelExportCancelled('导出已取消')

        def report(percent, text):
            if progress_cb is not None:
                try:
                    progress_cb(int(percent), str(text))
                except Exception:
                    pass

        target = Path(target_path)
        if target.suffix.lower() != '.ply':
            target = target.with_suffix('.ply')

        check_cancelled()
        report(2, '正在定位站点源点云...')
        station = self.get_station(station_id, project_uuid)
        if station is None:
            raise ValueError(f'未找到站点：{station_id}')
        if getattr(station, 'last_error', None):
            raise RuntimeError(f'站点资产无效：{station.last_error}')

        check_cancelled()
        report(10, '正在读取站点源点云...')
        points, colors = self.load_source_points(station)
        if points is None or len(points) == 0:
            raise ValueError(f'站点「{station.display_name}」源点云为空')
        source_count = int(len(points))
        report(55, f'源点云 {source_count:,} 点，正在下采样...')

        if voxel_size is None or float(voxel_size) <= 0:
            raise ValueError(f'voxel_size 必须大于 0，当前为 {voxel_size}')
        voxel = float(voxel_size)

        # 下采样按 5% → 85% 线性映射，让长任务在 5 秒粒度的刷新下
        # 仍能看出推进；点数极大时这里是唯一耗时环节。
        check_cancelled()
        down_points, down_colors = voxel_downsample(
            points, colors, voxel_size=voxel)
        # 下采样是单次库调用、内部无回报点：中止延迟由它决定。
        # 写盘前再查一次，避免用户已取消却仍落下一个完整文件。
        check_cancelled()
        report(85, f'已下采样至 {len(down_points):,} 点，正在写入 PLY...')

        if len(down_points) == 0:
            raise ValueError(
                f'下采样后点云为空（voxel={voxel:g}m 过大），请调小体素')

        result = write_point_cloud_ply_atomic(
            target, down_points, down_colors)
        report(100, f'导出完成：{result["point_count"]:,} 点')

        payload = dict(result)
        payload.update({
            'station_id': int(station.id),
            'station_name': str(getattr(station, 'display_name', '') or ''),
            'source_points': source_count,
            'exported_points': int(result['point_count']),
            'voxel_size': voxel,
        })
        return payload
