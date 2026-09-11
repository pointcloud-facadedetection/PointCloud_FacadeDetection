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
    def build(project, facades=None, project_root=None, facades_by_station=None) -> dict:
        # 优先使用 facades_by_station（多站点增量拓展模式）
        if facades_by_station is not None:
            return ReportDataService._build_by_station(project, facades_by_station, project_root)
        # 兼容旧模式：扁平 facades 列表
        project_name = getattr(project, "name", "") if project else ""
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
            item["wall_data"] = ReportDataService._build_wall_data(quality)
            station_id = item.get("station_id")
            item["building_key"] = str(station_id) if station_id is not None else "unbound"
            item["building_label"] = item.get("building_label") or (
                f"楼栋 {station_id}" if station_id is not None else "未绑定楼栋")
            item["wall_key"] = str(item.get("id") or item.get("display_no") or number)
            item["wall_label"] = item.get("wall_label") or f"墙面 {number}"
            item["pc_id"] = item.get("pc_id") or item.get("cloud_name") or station_id
            item["ply_path"] = item.get("ply_path") or item.get("source_path")
            item["project_name"] = project_name
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
            "project": ReportDataService._project_meta(project),
            "facades": rows,
            "buildings": buildings,
            "summary": ReportDataService._calc_summary(rows),
        }

    @staticmethod
    def _project_meta(project) -> dict:
        """Extract project metadata for report cover."""
        return {
            "name": getattr(project, "name", "") if project else "未选择项目",
            "org_unit": getattr(project, "org_unit", None) if project else None,
            "address": getattr(project, "address", None) if project else None,
            "building_floor": getattr(project, "building_floor", None) if project else None,
            "remarks": getattr(project, "remarks", None) if project else None,
            "project_id": getattr(project, "project_id", None) if project else None,
            "construction_unit": getattr(project, "construction_unit", None) if project else None,
            "construction_unit_executor": getattr(project, "construction_unit_executor", None) if project else None,
            "inspection_unit": getattr(project, "inspection_unit", None) if project else None,
            "supervision_unit": getattr(project, "supervision_unit", None) if project else None,
            "client_unit": getattr(project, "client_unit", None) if project else None,
            "report_no": getattr(project, "report_no", None) if project else None,
            "inspection_date": getattr(project, "inspection_date", None) if project else None,
            "report_date": getattr(project, "report_date", None) if project else None,
        }

    @staticmethod
    def _build_by_station(project, facades_by_station: dict[int, list[dict]], project_root=None) -> dict:
        """按站点（点云文件）分组构建报告数据，实现项目-多点云站点-立面墙体的强绑定。
        全新层级结构：项目基础信息 → 【点云文件名1-墙面1、墙面2...】 → 【点云文件名2-墙面1、墙面2...】
        """
        from pathlib import Path
        project_name = getattr(project, "name", "") if project else ""
        all_rows = []
        buildings = []
        for station_id in sorted(facades_by_station.keys()):
            station_facades = facades_by_station[station_id]
            # 从第一个 facade 中提取点云文件名作为 building 标签
            ply_name = None
            for f in station_facades:
                ply_path = f.get('ply_path') or f.get('source_path')
                if ply_path:
                    ply_name = Path(ply_path).name
                    break
            building_label = ply_name or f"点云站点 {station_id}"
            building_key = str(station_id)
            station_rows = []
            for index, source in enumerate(station_facades, 1):
                item = dict(source or {})
                quality = item.get("quality_report")
                if not ReportDataService._has_written_quality(item, quality):
                    continue
                number = item.get("display_no") or item.get("facade_no") or index
                try:
                    number = int(number)
                except (TypeError, ValueError):
                    number = index
                item["report_no"] = number
                item["quality"] = quality
                ReportDataService._ensure_quality_averages(quality)
                item["images"] = ReportDataService._images(project_root, number, quality)
                item["wall_data"] = ReportDataService._build_wall_data(quality)
                item["building_key"] = building_key
                item["building_label"] = building_label
                item["wall_key"] = str(item.get("id") or item.get("display_no") or number)
                item["wall_label"] = item.get("wall_label") or f"墙面 {number}"
                item["pc_id"] = item.get("pc_id") or item.get("cloud_name") or station_id
                item["ply_path"] = item.get("ply_path") or item.get("source_path") or building_label
                item["project_name"] = project_name
                station_rows.append(item)
            # 站内立面按 display_no 排序
            station_rows.sort(key=lambda v: (int(v.get("report_no") or 0), str(v.get("id", ""))))
            all_rows.extend(station_rows)
            if station_rows:
                buildings.append({
                    "key": building_key,
                    "label": building_label,
                    "walls": station_rows,
                })
        return {
            "project": ReportDataService._project_meta(project),
            "facades": all_rows,
            "buildings": buildings,
            "summary": ReportDataService._calc_summary(all_rows),
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

        # 双指标面积参考率：按两个指标各自的物理面积加权，禁止直接平均
        # 百分比（不同指标的有效面积可能不同）。这不是“同时合格面积率”，
        # 因为当前结果契约未保留平整度/垂直度联合掩码，故名称明确为参考率。
        for algo in ("ruler", "global_plane"):
            prefix = f"{algo}_"
            flat = data[f"{prefix}flatness"]
            vert = data[f"{prefix}verticality"]
            total = flat["total_area_m2"] + vert["total_area_m2"]
            passed = flat["pass_area_m2"] + vert["pass_area_m2"]
            data[f"{prefix}overall_rate"] = passed / total if total > 0 else None

        return data

    # ------------------------------------------------------------------
    # 全局摘要
    # ------------------------------------------------------------------
    @staticmethod
    def _calc_summary(rows: list[dict]) -> dict:
        total_facades = len(rows)
        pass_area = 0.0
        rate_area = 0.0
        total_area = 0.0
        total_points = 0
        for f in rows:
            wd = f.get("wall_data", {})
            # 汇总必须使用面积分子/分母，不得平均各墙面百分比。
            for metric in ("ruler_flatness", "ruler_verticality",
                           "global_plane_flatness", "global_plane_verticality"):
                metric_data = wd.get(metric, {})
                total = float(metric_data.get("total_area_m2", 0.0) or 0.0)
                passed = float(metric_data.get("pass_area_m2", 0.0) or 0.0)
                if total > 0:
                    rate_area += total
                    pass_area += min(max(passed, 0.0), total)
            # 使用 detection_area_m2（窗口覆盖并集）作为检测面积口径，
            # 回退到 total_area_m2（点云投影面积）以兼容旧数据。
            rf = wd.get("ruler_flatness", {})
            area = float(rf.get("detection_area_m2", 0.0))
            if area <= 0.0:
                area = float(rf.get("total_area_m2", 0.0))
            total_area += area
            total_points += int(rf.get("total_points", 0))

        avg_rate = (pass_area / rate_area * 100) if rate_area > 0 else None
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