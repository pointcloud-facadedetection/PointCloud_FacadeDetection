from __future__ import annotations

from typing import Iterable, Optional
from datetime import datetime
import json
import numpy as np

from sqlalchemy import select

from db.connection import project_session
from models import Facade, QualityMetric, Project, ResultScene


def _sqlite_scalar_text(value):
    """Convert metadata to a value accepted by SQLite TEXT columns."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)

class ResultsRepo:
    @staticmethod
    def persist_quality_artifact(results_dir, facade_id: int, quality: dict) -> str | None:
        """Persist only the raw-index vector needed to replay a heatmap."""
        indices = quality.get('__global_indices') if isinstance(quality, dict) else None
        if indices is None or len(indices) == 0 or not results_dir:
            return None
        from pathlib import Path
        root = Path(results_dir)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f'facade_{int(facade_id):03d}_quality_domain.npz'
        np.savez_compressed(path, global_indices=np.asarray(indices, dtype=np.int64))
        # 将项目相对的构建产物名称存储在 SQLite 中
        return path.name

    @staticmethod
    def load_quality_artifact(path) -> np.ndarray:
        if not path:
            return np.empty(0, dtype=np.int64)
        try:
            with np.load(path, allow_pickle=False) as data:
                return np.asarray(data['global_indices'], dtype=np.int64)
        except Exception:
            return np.empty(0, dtype=np.int64)

    @staticmethod
    def commit_quality_success(project_uuid: str, facade_id: int, quality: dict,
                                *, display_no=None, facade_data=None, color=None,
                                dataset_revision=None, quality_artifact_path=None) -> None:
        """Atomically persist a successful report without persisting point clouds."""
        if not isinstance(quality, dict) or not quality.get('ok', True):
            raise ValueError('只能持久化成功的质量结果')

        def serializable(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (np.floating, np.integer)):
                return value.item()
            if isinstance(value, dict):
                return {str(k): serializable(v) for k, v in value.items()
                        if not str(k).startswith('__') or str(k) == '__global_indices'}
            if isinstance(value, (list, tuple)):
                return [serializable(v) for v in value]
            return value

        def scalar_int(value, name):
            """Reject container values before they reach a scalar SQL bind."""
            if value is None:
                return None
            value = serializable(value)
            if isinstance(value, (dict, list, tuple)):
                raise ValueError(f'{name} 必须是标量，实际为 {type(value).__name__}')
            return int(value)

        report = serializable(quality)
        report.pop('__export_context', None)
        # 在 SQLite 中保留质量域。
        if quality_artifact_path and not report.get('__global_indices'):
            report['quality_artifact_path'] = str(quality_artifact_path)
        # 在开启事务之前进行验证，以防止格式错误的算法输出
        json.dumps(report, ensure_ascii=False)
        with project_session(project_uuid) as s:
            # 只有明确标记为数据库 ID 的 ID 才能指向现有行。
            facade = None
            expected_station = None
            if isinstance(facade_data, dict):
                expected_station = scalar_int(facade_data.get('station_id'), 'station_id')
            if isinstance(facade_data, dict) and facade_data.get('facade_db_id'):
                facade = s.execute(select(Facade).where(
                    Facade.id == int(facade_data['facade_db_id']),
                    Facade.project_id == select(Project.id).where(Project.uuid == project_uuid).scalar_subquery(),
                    Facade.is_deleted == 0,
                )).scalar_one_or_none()

            if facade is None and isinstance(facade_data, dict):
                project = s.execute(select(Project).where(
                    Project.uuid == project_uuid)).scalar_one_or_none()
                scene = s.execute(select(ResultScene).where(
                    ResultScene.project_id == project.id,
                    ResultScene.is_active.is_(True),
                )).scalar_one_or_none() if project is not None else None
                if project is not None:
                    if scene is None:
                        scene = ResultScene(
                            project_id=project.id,
                            name='Scene 1',
                            is_active=True,
                        )
                        s.add(scene)
                        s.flush()
                    geometry = {key: facade_data.get(key) for key in (
                        'plane_model', 'normal', 'center', 'inlier_indices',
                        'proxy_indices', 'measurement_indices', 'voxel_ids',
                        'cloud_name', '__index_space')
                        if facade_data.get(key) is not None}
                    point_count = int(facade_data.get('point_count') or
                                      len(facade_data.get('proxy_indices') or
                                          facade_data.get('inlier_indices') or []))
                    facade = Facade(
                        project_id=project.id, scene_id=scene.id,
                        label=f'Facade {int(display_no or 1)}',
                        display_no=int(display_no or 1),
                        point_count=point_count,
                        raw_point_count=int(facade_data.get('raw_point_count') or point_count),
                        plane_json=geometry,
                        bbox_json=facade_data.get('bbox_2d'),
                        area=float(facade_data.get('area', 0.0) or 0.0),
                        orientation=facade_data.get('type_label') or facade_data.get('type'),
                        station_id=expected_station,
                        dataset_id=_sqlite_scalar_text(facade_data.get('dataset_id')),
                        dataset_fingerprint=_sqlite_scalar_text(facade_data.get('dataset_fingerprint')),
                    )
                    s.add(facade)
                    s.flush()
            if facade is None:
                raise ValueError(
                    f'立面不存在: facade_id={facade_id}, display_no={display_no}')
            if isinstance(facade_data, dict):
                if expected_station is not None and int(facade.station_id or -1) != expected_station:
                    raise ValueError(
                        '立面与当前站点不匹配，拒绝保存质量结果 '
                        f'(facade_db_id={facade.id}, db_station_id={facade.station_id}, '
                        f'expected_station_id={expected_station})')
                if expected_station is not None:
                    facade.station_id = expected_station
                if facade_data.get('dataset_id') is not None:
                    facade.dataset_id = _sqlite_scalar_text(facade_data.get('dataset_id'))
                if facade_data.get('dataset_fingerprint') is not None:
                    facade.dataset_fingerprint = _sqlite_scalar_text(facade_data.get('dataset_fingerprint'))
            facade.quality_report_json = report
            if display_no is not None:
                facade.display_no = int(display_no)
            facade.quality_status = 'complete'
            facade.quality_completed_at = datetime.now()
            if dataset_revision is not None:
                facade.dataset_revision = _sqlite_scalar_text(dataset_revision)
            if color is not None:
                facade.color_json = serializable(color)
            # 在重复评估时替换指标，而不是累积行。
            for metric in list(facade.metrics):
                s.delete(metric)
            overall = report.get('overall') or {}
            thresholds = report.get('thresholds')
            values = {
                'flatness_std': (overall.get('flatness_std'), 'mm'),
                'flatness_max': (overall.get('flatness_max'), 'mm'),
                'verticality': (overall.get('verticality'), 'deg'),
            }
            for name, (value, unit) in values.items():
                if value is not None:
                    s.add(QualityMetric(facade_id=facade.id, metric_name=name,
                                        value=float(value), unit=unit,
                                        thresholds_json=thresholds))
            s.flush()

    @staticmethod
    def update_facade_review_status(project_uuid: str, facade_id: int, status: str) -> None:
        """Persist the operator's review state without creating a new result."""
        if status not in {'pending', 'incomplete', 'complete'}:
            raise ValueError(f'unsupported facade review status: {status}')
        with project_session(project_uuid) as s:
            facade = s.execute(
                select(Facade).where(Facade.id == int(facade_id), Facade.is_deleted == 0)
            ).scalar_one_or_none()
            if facade is None:
                raise ValueError(f'立面不存在: {facade_id}')
            payload = dict(facade.plane_json or {})
            payload['review_status'] = status
            facade.plane_json = payload
            s.flush()

    @staticmethod
    def save_detected_facades(project_uuid: str, items: Iterable[dict]) -> list[Facade]:
        """将一个检测批次及其基本指标持久化到当前场景中。

        将此事务保存在存储库中，可避免应用程序服务依赖于 SQLAlchemy 会话的详细信息。
        """
        with project_session(project_uuid) as s:
            project = s.execute(
                select(Project).where(Project.uuid == project_uuid)
            ).scalar_one_or_none()
            if project is None:
                raise RuntimeError("项目不存在")
            scene = s.execute(
                select(ResultScene).where(
                    ResultScene.project_id == project.id,
                    ResultScene.is_active.is_(True),
                )
            ).scalar_one_or_none()
            if scene is None:
                scene = ResultScene(project_id=project.id, name="Scene 1", is_active=True)
                s.add(scene)
                s.flush()

            items = list(items)
            station_ids = {int(item['station_id']) for item in items
                           if item.get('station_id') is not None}
            if not items:
                return []
            if len(station_ids) != 1:
                raise ValueError('检测结果必须全部属于同一个站点')
            station_id = next(iter(station_ids))
            dataset_ids = {str(item['dataset_id']) for item in items
                           if item.get('dataset_id')}
            dataset_id = next(iter(dataset_ids), None)
            # 新的检测结果将替换当前场景之前的有效数据集。
            old_rows = s.execute(select(Facade).where(
                Facade.scene_id == scene.id, Facade.station_id == station_id,
                Facade.is_deleted == 0)).scalars().all()
            for old in old_rows:
                s.delete(old)
            s.flush()
            saved = []
            for display_no, item in enumerate(items, 1):
                point_count = int(item.get('point_count') or
                                  len(item.get('proxy_indices') or item.get('inlier_indices') or []))
                raw_point_count = int(item.get('raw_point_count') or point_count)
                item['display_no'] = display_no
                facade = Facade(
                    project_id=project.id,
                    scene_id=scene.id,
                    label=f"Facade {display_no}",
                    display_no=display_no,
                    point_count=point_count,
                    raw_point_count=raw_point_count,
                    # 保留完整的索引空间有效载荷。
                    plane_json={key: item.get(key) for key in (
                        "plane_model", "normal", "center", "inlier_indices",
                        "proxy_indices", "measurement_indices", "voxel_ids",
                        "review_status",
                        "cloud_name", "__index_space",
                    ) if item.get(key) is not None} | {
                        'point_count': point_count,
                        'raw_point_count': raw_point_count,
                    },
                    bbox_json=item.get("bbox_2d"),
                    area=float(item.get("area", 0.0)),
                    orientation=item.get("type_label") or item.get("type"),
                    station_id=station_id,
                    dataset_id=_sqlite_scalar_text(item.get('dataset_id') or dataset_id),
                    dataset_fingerprint=_sqlite_scalar_text(item.get('dataset_fingerprint')),
                )
                s.add(facade)
                s.flush()
                item['facade_db_id'] = facade.id
                saved.append(facade)
                metrics = (
                    ("flatness_std", float(item.get("flatness", 0.0)) * 1000.0, "mm"),
                    ("flatness_mean", float(item.get("flatness_mean", 0.0)) * 1000.0, "mm"),
                    ("flatness_max", float(item.get("flatness_max", 0.0)) * 1000.0, "mm"),
                    ("verticality", float(item.get("verticality", 0.0)), "deg"),
                    ("horizontality", float(item.get("horizontality", 0.0)), "deg"),
                )
                for name, value, unit in metrics:
                    s.add(QualityMetric(
                        facade_id=facade.id, metric_name=name, value=value, unit=unit
                    ))
            s.flush()
            return saved

    @staticmethod
    def save_facades(project_uuid: str, scene_id: int, items: Iterable[dict]) -> list[Facade]:
        """items: dicts with keys label, plane_json, bbox_json, area, orientation"""
        saved: list[Facade] = []
        with project_session(project_uuid) as s:
            proj = s.execute(select(Project).where(Project.uuid == project_uuid)).scalar_one()
            for d in items:
                f = Facade(
                    project_id=proj.id,
                    scene_id=scene_id,
                    label=d.get("label", "facade"),
                    plane_json=d.get("plane_json") or {
                        key: d.get(key) for key in (
                            "plane_model", "normal", "center", "inlier_indices",
                            "proxy_indices", "measurement_indices", "voxel_ids",
                            "review_status",
                            "cloud_name", "__index_space",
                        ) if d.get(key) is not None
                    },
                    bbox_json=d.get("bbox_json"),
                    area=d.get("area"),
                    orientation=d.get("orientation"),
                )
                s.add(f)
                s.flush()
                saved.append(f)
        return saved

    @staticmethod
    def save_quality(project_uuid: str, facade_id: int, metrics: dict[str, dict]) -> None:
        """metrics: name -> {value: float, unit: str|None, pass_flag: int|None, thresholds: dict|None}"""
        with project_session(project_uuid) as s:
            for name, payload in metrics.items():
                value = payload.get("value")
                unit = payload.get("unit")
                pass_flag = payload.get("pass_flag")
                thresholds = payload.get("thresholds")
                s.add(QualityMetric(
                    facade_id=facade_id,
                    metric_name=name,
                    value=value,
                    unit=unit,
                    pass_flag=pass_flag,
                    thresholds_json=thresholds,
                ))
            s.flush()

    @staticmethod
    def get_facade_list(project_uuid: str, scene_id: int) -> list[Facade]:
        with project_session(project_uuid) as s:
            q = s.execute(select(Facade).where((Facade.scene_id == scene_id) & (Facade.is_deleted == 0)))
            return q.scalars().all()
