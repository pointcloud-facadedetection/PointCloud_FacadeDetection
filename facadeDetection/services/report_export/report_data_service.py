"""Build a read-only, serializable snapshot for report preview/export."""
from __future__ import annotations

from pathlib import Path
import re


class ReportDataService:
    """Adapt runtime and historical facade dictionaries without changing them."""

    @staticmethod
    def build(project, facades=None, project_root=None) -> dict:
        rows = []
        for index, source in enumerate(facades or [], 1):
            item = dict(source or {})
            quality = item.get("quality_report")
            if not ReportDataService._has_written_quality(item, quality):
                continue
            number = item.get("display_no") or item.get("facade_no")
            if number is None:
                match = re.search(r"(\d+)", str(item.get("type_label") or item.get("label") or ""))
                number = int(match.group(1)) + 1 if match else index
            try:
                number = int(number)
            except (TypeError, ValueError):
                number = index
            item["report_no"] = number
            item["quality"] = quality
            item["images"] = ReportDataService._images(project_root, number, item["quality"])
            rows.append(item)
        rows.sort(key=lambda value: (value["report_no"], str(value.get("id", ""))))
        return {
            "project": {
                "name": getattr(project, "name", "") if project else "未选择项目",
                "org_unit": getattr(project, "org_unit", None) if project else None,
                "address": getattr(project, "address", None) if project else None,
                "building_floor": getattr(project, "building_floor", None) if project else None,
                "remarks": getattr(project, "remarks", None) if project else None,
                "project_id": getattr(project, "project_id", None) if project else None,
            },
            "facades": rows,
        }

    @staticmethod
    def _has_written_quality(facade, quality):
        """A report row exists only after a successful persisted measurement."""
        if facade.get("quality_status") != "complete" or not isinstance(quality, dict):
            return False
        if quality.get("ok") is False:
            return False
        overall = quality.get("overall") or {}
        try:
            valid = int(overall.get("quality_valid_window_count", 0) or 0)
        except (TypeError, ValueError):
            valid = 0
        windows = quality.get("windows")
        return valid > 0 and isinstance(windows, list) and bool(windows)

    @staticmethod
    def _images(project_root, number, quality):
        paths = []
        context = quality.get("__export_context") if isinstance(quality, dict) else {}
        if isinstance(context, dict):
            artifacts = context.get("heatmaps") or {}
            for mode in ("flatness", "verticality"):
                artifact = artifacts.get(mode) if isinstance(artifacts, dict) else None
                value = ((artifact.get("overlay") or artifact.get("heatmap"))
                         if isinstance(artifact, dict) else None)
                if value and Path(value).is_file():
                    paths.append({"mode": mode, "title": artifact.get("title", mode),
                                  "path": str(Path(value))})
            for key in ("heatmap", "overlay", "flatness_image", "verticality_image"):
                value = context.get(key)
                if value and Path(value).is_file() and not any(
                        (item.get("path") if isinstance(item, dict) else item) == str(Path(value))
                        for item in paths):
                    paths.append(str(Path(value)))
        # Persisted reports keep portable artifact metadata at the quality
        # root because __export_context intentionally excludes large arrays.
        artifacts = quality.get("heatmap_artifacts") if isinstance(quality, dict) else {}
        if isinstance(artifacts, dict):
            for mode in ("flatness", "verticality"):
                artifact = artifacts.get(mode)
                value = ((artifact.get("overlay") or artifact.get("heatmap"))
                         if isinstance(artifact, dict) else None)
                if value and Path(value).is_file() and not any(
                        (item.get("path") if isinstance(item, dict) else item) == str(Path(value))
                        for item in paths):
                    paths.append({"mode": mode, "title": artifact.get("title", mode),
                                  "path": str(Path(value))})
        if project_root:
            folder = Path(project_root) / "results" / f"facade_{number:03d}"
            for mode in ("flatness", "verticality"):
                path = folder / f"facade_{number:03d}_{mode}_overlay.png"
                if not path.is_file():
                    path = folder / f"facade_{number:03d}_{mode}_heatmap.png"
                if path.is_file() and not any((item.get("path") if isinstance(item, dict) else item) == str(path)
                                              for item in paths):
                    paths.append({"mode": mode, "title": f"{'平整度' if mode == 'flatness' else '垂直度'}热力图",
                                  "path": str(path)})
            # Legacy artifacts remain readable for reports created by older builds.
            for name in ("defect_overlay.png", f"facade_{number:03d}_defect_heatmap.png"):
                path = folder / name
                if path.is_file() and not any((item.get("path") if isinstance(item, dict) else item) == str(path)
                                              for item in paths):
                    paths.append(str(path))
        # Keep the two measurement modes side by side in the PDF. If only one
        # mode has been exported, do not fill the second slot with a legacy
        # duplicate image.
        return paths[:2]