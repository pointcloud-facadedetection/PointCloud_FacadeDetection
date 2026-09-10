"""Build a read-only, serializable snapshot for report preview/export."""
from __future__ import annotations

from pathlib import Path
import re
import math
import numpy as np


class ReportDataService:
    """Adapt runtime and historical facade dictionaries without changing them."""

    # 4 组展示模式（用于图片检索）
    _DISPLAY_MODES = [
        'ruler_flatness_area',
        'ruler_verticality_area',
        'global_plane_flatness_area',
        'global_plane_verticality_area',
    ]

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
            ReportDataService._ensure_quality_averages(quality)
            item["images"] = ReportDataService._images(project_root, number, quality)
            # 新增：结构化墙体数据
            item["wall_data"] = ReportDataService._build_wall_data(quality)
            # A facade is owned by exactly one imported PLY station.  Preserve
            # this identity in the report snapshot so incremental imports do
            # not merge walls with the same display number.
            station_id = item.get("station_id")
            item["building_key"] = str(station_id) if station_id is not None else "unbound"
            item["building_label"] = item.get("building_label") or (
                f"楼栋 {station_id}" if station_id is not None else "未绑定楼栋")
            item["wall_key"] = str(item.get("id") or item.get("display_no") or number)
            item["wall_label"] = item.get("wall_label") or f"墙面 {number}"
            # Keep the source identity beside the measured result.  Different
            # PLY imports may reuse facade numbers, so report consumers must
            # never infer the source from ``report_no`` alone.
            item["ply_id"] = item.get("ply_id") or item.get("cloud_name") or station_id
            item["ply_path"] = item.get("ply_path") or item.get("source_path")
            rows.append(item)
        rows.sort(key=lambda value: (value["report_no"], str(value.get("id", ""))))
        rows.sort(key=lambda value: (value["building_key"], value["report_no"],
                                     value["wall_key"]))
        buildings = []
        for key in sorted({row["building_key"] for row in rows}):
            buildings.append({
                "key": key,
                "label": next(row["building_label"] for row in rows
                               if row["building_key"] == key),
                "walls": [row for row in rows if row["building_key"] == key],
            })
        return {
            "project": {
                "name": getattr(project, "name", "") if project else "未选择项目",
                "org_unit": getattr(project, "org_unit", None) if project else None,
                "address": getattr(project, "address", None) if project else None,
                "building_floor": getattr(project, "building_floor", None) if project else None,
                "remarks": getattr(project, "remarks", None) if project else None,
                "project_id": getattr(project, "project_id", None) if project else None,
                # PDF 报告元信息
                "construction_unit": getattr(project, "construction_unit", None) if project else None,
                "construction_unit_executor": getattr(project, "construction_unit_executor", None) if project else None,
                "inspection_unit": getattr(project, "inspection_unit", None) if project else None,
                "supervision_unit": getattr(project, "supervision_unit", None) if project else None,
                "client_unit": getattr(project, "client_unit", None) if project else None,
                "report_no": getattr(project, "report_no", None) if project else None,
                "inspection_date": getattr(project, "inspection_date", None) if project else None,
                "report_date": getattr(project, "report_date", None) if project else None,
            },
            "facades": rows,
            "buildings": buildings,
            # 全局摘要
            "summary": ReportDataService._calc_summary(rows),
        }

    # ------------------------------------------------------------------
    # 墙体数据结构化（模板大表所需）
    # ------------------------------------------------------------------
    @staticmethod
    def _build_wall_data(quality: dict) -> dict:
        """Extract wall metrics matching the template table format."""
        comparison = quality.get("quality_comparison", {}) if quality else {}
        methods = comparison.get("methods", {})
        thresholds = quality.get("thresholds", {}) if quality else {}
        profile = quality.get("profile_snapshot", {}) if quality else {}
        profile = profile if isinstance(profile, dict) else {}
        standard_name = (profile.get("standard_name") or profile.get("wall_type")
                         or "未指定")
        version = profile.get("version") or ""
        material = profile.get("material") or standard_name

        def _extract(method_data: dict, metric: str) -> dict:
            method_data = method_data if isinstance(method_data, dict) else {}
            data = method_data.get(metric, {})
            data = data if isinstance(data, dict) else {}
            rates = data.get("rates", {})
            rates = rates if isinstance(rates, dict) else {}
            windows = data.get("windows", [])
            windows = windows if isinstance(windows, list) else []
            value_key = "flatness_gap_mm" if metric == "flatness" else "verticality_deviation_mm"
            vals = []
            for window in windows:
                try:
                    value = float(window.get(value_key, np.nan))
                except (TypeError, ValueError, AttributeError):
                    continue
                if np.isfinite(value):
                    vals.append(value)
            total_area = float(rates.get("total_area_m2", 0.0))
            pass_area = float(rates.get("pass_area_m2", 0.0))
            detection_area = float(rates.get("detection_area_m2", total_area))
            total_points = int(rates.get("total_points", 0) or 0)
            pass_points = int(rates.get("pass_points", 0) or 0)
            fail_points = int(rates.get("fail_points", max(total_points - pass_points, 0)) or 0)
            return {
                "total_area_m2": total_area,
                "pass_area_m2": pass_area,
                "fail_area_m2": float(rates.get("fail_area_m2", max(total_area - pass_area, 0.0))),
                "detection_area_m2": detection_area,
                "area_rate": float(rates.get("area_rate", 0.0)),
                "total_points": total_points,
                "pass_points": pass_points,
                "fail_points": fail_points,
                "point_rate": float(rates.get("point_rate", 0.0)),
                "max_mm": max(vals) if vals else None,
                "avg_mm": (sum(vals) / len(vals)) if vals else None,
                "threshold_mm": float(profile.get(
                    f"{metric}_limit_mm",
                    thresholds.get(
                        f"{metric}_limit_mm",
                        (method_data.get("parameters", {}) or {}).get(
                            f"{metric}_limit_mm", 4.0),
                    ),
                )),
                "material": material,
                "standard_name": standard_name,
                "version": version,
                "flatness_limit_mm": float(profile.get(
                    "flatness_limit_mm", thresholds.get("flatness_limit_mm", 4.0))),
                "verticality_limit_mm": float(profile.get(
                    "verticality_limit_mm", thresholds.get("verticality_limit_mm", 4.0))),
                "window_count": len(windows),
                "pass_window_count": sum(
                    1 for w in windows if w.get(
                        "flatness_pass" if metric == "flatness" else "verticality_pass", False
                    )
                ),
            }

        ruler = methods.get("ruler", {})
        global_plane = methods.get("global_plane", {})

        data = {
            "ruler_flatness": _extract(ruler, "flatness"),
            "ruler_verticality": _extract(ruler, "verticality"),
            "global_plane_flatness": _extract(global_plane, "flatness"),
            "global_plane_verticality": _extract(global_plane, "verticality"),
        }

        # 整体合格率 = (平整度面积合格率 + 垂直度面积合格率) / 2
        for algo in ("ruler", "global_plane"):
            prefix = f"{algo}_"
            fr = data[f"{prefix}flatness"]["area_rate"]
            vr = data[f"{prefix}verticality"]["area_rate"]
            data[f"{prefix}overall_rate"] = (
                (fr + vr) / 2.0 if fr is not None and vr is not None else None
            )

        return data

    # ------------------------------------------------------------------
    # 全局摘要
    # ------------------------------------------------------------------
    @staticmethod
    def _calc_summary(rows: list[dict]) -> dict:
        total_facades = len(rows)
        rates = []
        total_area = 0.0
        total_points = 0
        for f in rows:
            wd = f.get("wall_data", {})
            for algo in ("ruler_", "global_plane_"):
                rate = wd.get(f"{algo}overall_rate")
                if rate is not None:
                    rates.append(rate)
            # 使用 detection_area_m2（窗口覆盖并集）作为检测面积口径，
            # 回退到 total_area_m2（点云投影面积）以兼容旧数据。
            rf = wd.get("ruler_flatness", {})
            area = float(rf.get("detection_area_m2", 0.0))
            if area <= 0.0:
                area = float(rf.get("total_area_m2", 0.0))
            total_area += area
            total_points += int(rf.get("total_points", 0))

        avg_rate = (sum(rates) / len(rates) * 100) if rates else None
        return {
            "total_facades": total_facades,
            "avg_pass_rate": f"{avg_rate:.1f}%" if avg_rate is not None else "--",
            "total_area": round(total_area, 2),
            "total_points": total_points,
        }

    # ------------------------------------------------------------------
    # 图片检索（重构后适配 triplet 文件命名）
    # ------------------------------------------------------------------
    @staticmethod
    def _images(project_root, number, quality):
        paths = []
        context = quality.get("__export_context") if isinstance(quality, dict) else {}
        if isinstance(context, dict):
            artifacts = context.get("heatmaps") or {}
            for mode in ReportDataService._DISPLAY_MODES:
                artifact = artifacts.get(mode) if isinstance(artifacts, dict) else None
                if isinstance(artifact, dict):
                    for key in ("overlay", "heatmap_grid", "photo", "report"):
                        value = artifact.get(key)
                        if value and Path(value).is_file():
                            paths.append({
                                "mode": mode,
                                "key": key,
                                "title": artifact.get("title", mode),
                                "path": str(Path(value)),
                            })

        # Fallback: scan filesystem by known naming convention
        search_roots = []
        if project_root:
            search_roots.append(Path(project_root) / "results" / f"facade_{number:03d}")
        if isinstance(context, dict) and context.get("results_dir"):
            search_roots.append(Path(context["results_dir"]) / f"facade_{number:03d}")

        for folder in search_roots:
            if not folder.is_dir():
                continue
            for mode in ReportDataService._DISPLAY_MODES:
                prefix = f"facade_{number:03d}_{mode}"
                for suffix, key in (
                    ("_overlay.png", "overlay"),
                    ("_heatmap_grid.png", "heatmap_grid"),
                    ("_photo_overlay.png", "photo"),
                    ("_report.png", "report"),
                ):
                    path = folder / (prefix + suffix)
                    if path.is_file() and not any(
                        (item.get("path") if isinstance(item, dict) else item) == str(path)
                        for item in paths
                    ):
                        metric = "平整度" if "flatness" in mode else "垂直度"
                        algo = "靠尺法" if mode.startswith("ruler") else "全局平面法"
                        paths.append({
                            "mode": mode,
                            "key": key,
                            "title": f"{algo}{metric}",
                            "path": str(path),
                        })

        return paths[:12]

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
    def _ensure_quality_averages(quality):
        """Fill statistics for legacy results without mutating persisted data."""
        overall = quality.setdefault("overall", {})
        windows = quality.get("windows") or []

        if "flatness_avg_gap_mm" not in overall:
            values = []
            for row in windows:
                if not row.get("coverage_valid", False):
                    continue
                try:
                    value = float(row.get("flatness_gap_mm"))
                    if math.isfinite(value):
                        values.append(value)
                except (TypeError, ValueError):
                    pass
            overall["flatness_avg_gap_mm"] = sum(values) / len(values) if values else math.nan

        if "verticality_avg_deviation_mm" not in overall:
            values = []
            for row in windows:
                try:
                    value = float(row.get("verticality_deviation_mm"))
                    if math.isfinite(value):
                        values.append(value)
                except (TypeError, ValueError):
                    pass
            overall["verticality_avg_deviation_mm"] = sum(values) / len(values) if values else math.nan