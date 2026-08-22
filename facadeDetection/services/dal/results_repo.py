from __future__ import annotations

from typing import Iterable, Optional

from sqlalchemy import select

from db.connection import project_session
from models import Facade, Heatmap, QualityMetric, Project, ResultScene


class ResultsRepo:
    @staticmethod
    def save_detected_facades(project_uuid: str, items: Iterable[dict]) -> None:
        """Persist a detection batch and its basic metrics in the active scene.

        Keeping this transaction in the repository prevents the application
        service from depending on SQLAlchemy session details.
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

            for item in items:
                facade = Facade(
                    project_id=project.id,
                    scene_id=scene.id,
                    label=f"Facade {int(item.get('id', 0))}",
                    # Keep the complete index-space payload.  Quality evaluation
                    # after reopening depends on these fields, not just on the
                    # plane preview/summary.
                    plane_json={key: item.get(key) for key in (
                        "plane_model", "normal", "center", "inlier_indices",
                        "proxy_indices", "measurement_indices", "voxel_ids",
                        "cloud_name", "__index_space",
                    ) if item.get(key) is not None},
                    bbox_json=item.get("bbox_2d"),
                    area=float(item.get("area", 0.0)),
                    orientation=item.get("type_label") or item.get("type"),
                )
                s.add(facade)
                s.flush()
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
    def bind_heatmap(project_uuid: str, facade_id: int, file_id: int, vmin: float | None, vmax: float | None, cmap: str | None) -> Heatmap:
        with project_session(project_uuid) as s:
            h = Heatmap(facade_id=facade_id, file_id=file_id, vmin=vmin, vmax=vmax, cmap=cmap)
            s.add(h)
            s.flush()
            return h

    @staticmethod
    def get_facade_list(project_uuid: str, scene_id: int) -> list[Facade]:
        with project_session(project_uuid) as s:
            q = s.execute(select(Facade).where((Facade.scene_id == scene_id) & (Facade.is_deleted == 0)))
            return q.scalars().all()
