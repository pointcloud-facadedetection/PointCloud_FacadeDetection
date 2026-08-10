from __future__ import annotations

from typing import Iterable, Optional

from sqlalchemy import select

from db.connection import project_session
from models import Facade, Heatmap, QualityMetric, Project


class ResultsRepo:
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
                    plane_json=d.get("plane_json"),
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
