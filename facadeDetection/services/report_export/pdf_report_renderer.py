"""Shared HTML template and native Qt PDF renderer for commercial reports."""
from __future__ import annotations

from html import escape
from pathlib import Path
from datetime import datetime
from PySide6.QtGui import QTextDocument
from PySide6.QtPrintSupport import QPrinter
from PySide6.QtGui import QPageSize


def _text(value, fallback="--"):
    return escape(str(value)) if value not in (None, "") else fallback


def _num(value, unit="", decimals=2):
    try:
        text = f"{float(value):.{decimals}f}".rstrip("0").rstrip(".")
        return f"{text} {unit}".strip()
    except (TypeError, ValueError):
        return "--"


def _pct(value):
    try:
        number = float(value)
        if abs(number) <= 1:
            number *= 100
        return f"{number:.1f}%"
    except (TypeError, ValueError):
        return "--"


class PdfReportRenderer:
    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    @staticmethod
    def html(snapshot: dict) -> str:
        project = snapshot.get("project") or {}
        summary = snapshot.get("summary") or {}
        facades = snapshot.get("facades", [])

        cover = PdfReportRenderer._cover(project, summary)
        body = PdfReportRenderer._body(facades, snapshot.get("buildings"))

        style = (
            "@page { size:A4; margin:12mm 14mm 14mm; }"
            "body { font-family:'Microsoft YaHei','SimSun',sans-serif; color:#1f2937; font-size:9pt; line-height:1.4; margin:0; }"
            "h1 { color:#163a63; font-size:18pt; margin:0 0 3mm; border-bottom:2.5px solid #2f75b5; padding-bottom:3mm; font-weight:700; text-align:center; }"
            "h2 { color:#163a63; background:#eaf2fb; padding:5px 10px; margin:0 0 6px; border-left:4px solid #2f75b5; font-size:12pt; }"
            "h3 { color:#365b7d; font-size:10pt; margin:8px 0 4px; font-weight:600; border-bottom:1px solid #e2e8f0; padding-bottom:2px; }"
            ".cover-company { text-align:center; font-size:20pt; font-weight:700; color:#1e293b; margin-bottom:6px; }"
            ".cover-title { text-align:center; font-size:18pt; font-weight:700; color:#163a63; margin-bottom:16px; border-bottom:2px solid #2f75b5; padding-bottom:8px; }"
            ".cover-section-title { font-size:13pt; font-weight:700; color:#2f75b5; margin:12px 0 8px; text-align:center; }"
            "table.summary-cards { width: 60%; margin: 0 auto 16px; border-collapse: separate; border-spacing: 10px; }"
            "td.summary-card { background: #f8fafc; border: 1px solid #e2e8f0; padding: 14px 8px; text-align: center; vertical-align: middle; width: 50%; }"
            "td.sc-blue { border-top: 3px solid #3b82f6; }"
            "td.sc-green { border-top: 3px solid #22c55e; }"
            "td.sc-orange { border-top: 3px solid #f59e0b; }"
            "td.sc-cyan { border-top: 3px solid #06b6d4; }"
            ".sc-number { font-size: 18pt; font-weight: 700; color: #163a63; line-height: 1.2; }"
            ".sc-label { font-size: 8pt; color: #64748b; margin-top: 4px; }"
            "table.cover-info { width: 100%; border-collapse: collapse; font-size: 10pt; margin-top: 8px; }"
            "table.cover-info td { padding: 5px 8px; vertical-align: top; }"
            ".ci-label { width: 25%; color: #64748b; text-align: right; font-size: 9.5pt; }"
            ".ci-value { width: 75%; color: #1e293b; font-weight: 500; text-align: left; }"
            "table.wall-data { width: 100%; border-collapse: collapse; font-size: 8.5pt; margin: 8px 0; }"
            "table.wall-data td { border: 1px solid #cbd5e1; padding: 5px 6px; text-align: center; vertical-align: middle; }"
            "table.wall-data td:first-child { background: #f1f5f9; font-weight: 600; color: #475569; }"
            "table.wall-data td:last-child { background: #f8fafc; font-weight: 600; color: #163a63; }"
            ".wd-header { background: #e2e8f0 !important; font-weight: 700; color: #1e293b; }"
            ".analysis-text { font-size: 8pt; color: #334155; line-height: 1.6; margin: 8px 0 12px; padding: 8px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px; }"
            "table.triplet { width: 100%; border-collapse: separate; border-spacing: 8px; margin: 8px 0; page-break-inside: avoid; page-break-before: avoid; page-break-after: avoid; }"
            "table.triplet td { width: 33.3%; vertical-align: top; text-align: center; border: 1px solid #e2e8f0; padding: 4px; background: #fff; }"
            "table.triplet img { max-width: 100%; max-height: 92mm; width: auto; height: auto; object-fit: contain; display: block; margin: 0 auto; }"
            ".triplet-title { font-size: 8pt; color: #365b7d; font-weight: 600; margin-bottom: 4px; text-align: center; }"
            ".triplet-placeholder { width: 100%; height: 120px; background: #f1f5f9; color: #94a3b8; font-size: 8pt; display: flex; align-items: center; justify-content: center; }"
            ".report-meta { font-size: 8pt; color: #94a3b8; margin-bottom: 8px; text-align: right; }"
            ".muted { color: #94a3b8; font-size: 8.5pt; text-align: center; padding: 10px; }"
            "section { page-break-inside: avoid; margin-bottom: 10px; }"
             ".building { page-break-before: always; page-break-after: avoid; }"
            ".facade-header { margin-bottom: 6px; }"
        )

        return (
            "<!doctype html><html><head><meta charset='utf-8'><style>"
            + style
            + "</style></head><body>"
            + cover
            + body
            + "</body></html>"
        )

    # ------------------------------------------------------------------
    # Cover page
    # ------------------------------------------------------------------
    @staticmethod
    def _cover(project: dict, summary: dict) -> str:
        inspection_date = project.get("inspection_date")
        report_date = project.get("report_date")
        idate = (
            inspection_date.strftime("%Y年%m月%d日")
            if hasattr(inspection_date, "strftime")
            else _text(inspection_date)
        )
        rdate = (
            report_date.strftime("%Y年%m月%d日")
            if hasattr(report_date, "strftime")
            else _text(report_date)
        )

        info_rows = ""
        pairs = [
            ("项目名称", project.get("name")),
            ("报告编号", project.get("report_no")),
            ("建设单位", project.get("construction_unit")),
            ("施工单位", project.get("construction_unit_executor")),
            ("检测单位", project.get("inspection_unit")),
            ("监理单位", project.get("supervision_unit")),
            ("测量日期", idate),
            ("施工阶段", "外立面工程"),
            ("报告日期", rdate),
        ]
        for label, value in pairs:
            info_rows += (
                f"<tr><td class='ci-label'>{escape(label)}</td>"
                f"<td class='ci-value'>{_text(value)}</td></tr>"
            )

        total_facades = summary.get("total_facades", 0)
        avg_pass_rate = summary.get("avg_pass_rate", "--")
        total_area = summary.get("total_area", 0)
        total_points = summary.get("total_points", 0)

        return (
            f"<div class='cover-company'>{_text(project.get('inspection_unit'), 'XX有限公司')}</div>"
            f"<div class='cover-title'>外立面激光测量检测报告</div>"
            f"<div class='cover-section-title'>项目检测摘要</div>"
            f"<table class='summary-cards'>"
            f"<tr>"
            f"<td class='summary-card sc-blue'><div class='sc-number'>{total_facades}</div><div class='sc-label'>检测立面</div></td>"
            f"<td class='summary-card sc-green'><div class='sc-number' style='color:#15803d;'>{avg_pass_rate}</div><div class='sc-label'>平均合格率</div></td>"
            f"</tr>"
            f"<tr>"
            f"<td class='summary-card sc-orange'><div class='sc-number' style='color:#b45309;'>{total_area}</div><div class='sc-label'>检测面积</div></td>"
            f"<td class='summary-card sc-cyan'><div class='sc-number' style='color:#0369a1;'>{total_points}</div><div class='sc-label'>检测点数</div></td>"
            f"</tr>"
            f"</table>"
            f"<table class='cover-info'>{info_rows}</table>"
            f"<div style='page-break-after:always;'></div>"
        )

    # ------------------------------------------------------------------
    # Body (facades)
    # ------------------------------------------------------------------
    @staticmethod
    def _body(facades: list[dict], buildings: list[dict] | None = None) -> str:
        if not facades:
            return '<p class="muted">暂无立面检测结果</p>'

        sections = []
        building_groups = buildings or [{"label": "楼栋 1", "walls": facades}]
        for building in building_groups:
            walls = building.get("walls") or []
            if not walls:
                continue
            building_index = len([part for part in sections if "class='building'" in part])
            building_class = "building" if building_index else "building first-building"
            sections.append(f"<div class='{building_class}'><h1>{_text(building.get('label'), '楼栋')}</h1></div>")
            for facade in walls:
                wd = facade.get("wall_data") or {}
                number = facade.get("report_no", facade.get("id", 0))

                table_html = PdfReportRenderer._wall_data_table(wd)
                analysis_html = PdfReportRenderer._analysis_text(wd)
                images = facade.get("images", [])
                triplets_html = PdfReportRenderer._triplets(images, number)

                sections.append(
                    f"<section>"
                    f"<h2>{_text(facade.get('wall_label'), f'墙面 {number}')} — "
                    f"PLY { _text(facade.get('ply_id') or facade.get('station_id')) }</h2>"
                    f"{table_html}"
                    f"<div class='analysis-text'>{analysis_html}</div>"
                    f"{triplets_html}"
                    f"</section>"
                )

        return "".join(sections)

    # ------------------------------------------------------------------
    # Wall data table (5 rows x 5 cols)
    # ------------------------------------------------------------------
    @staticmethod
    def _wall_data_table(wd: dict) -> str:
        rf = wd.get("ruler_flatness", {})
        rv = wd.get("ruler_verticality", {})
        gf = wd.get("global_plane_flatness", {})
        gv = wd.get("global_plane_verticality", {})

        # ---- 构建同时包含平整度与垂直度的检测标准 ----
        flat_name = rf.get("standard_name") or "未指定标准"
        flat_ver = rf.get("version") or ""
        flat_limit = rf.get("threshold_mm", 4.0)

        vert_name = rv.get("standard_name") or flat_name
        vert_ver = rv.get("version") or flat_ver
        vert_limit = rv.get("threshold_mm", 4.0)

        std = f"{flat_name} {flat_ver} [平整度 {flat_limit:.1f}mm / 垂直度 {vert_limit:.1f}mm]".strip()
        
        def cell(v):
            return _num(v, "m²") if isinstance(v, (int, float)) and v > 10 else _num(v, "", 1)

        def pct(v):
            return _pct(v) if v is not None else "--"

        rows = [
            f"<tr><td>面层材质</td><td>{_text(rf.get('material'))}</td><td class='wd-header'>检测标准</td><td>{escape(std)}</td></tr>",
            f"<tr><td class='wd-header'>总测量面积(模拟下尺)</td><td class='wd-header'>合格面积(模拟下尺)</td><td class='wd-header'>不合格面积(模拟下尺)</td><td class='wd-header'>合格率</td></tr>"
            f"<tr><td>{cell(rf.get('total_area_m2'))}</td><td>{cell(rf.get('pass_area_m2'))}</td><td>{cell(rf.get('fail_area_m2'))}</td><td>{pct(rf.get('area_rate'))}</td></tr>",
            f"<tr><td class='wd-header'>总测量点数(模拟下尺)</td><td class='wd-header'>合格点数(模拟下尺)</td><td class='wd-header'>不合格点数(模拟下尺)</td><td class='wd-header'>合格率</td></tr>"
            f"<tr><td>{rf.get('total_points', '--')}</td><td>{rf.get('pass_points', '--')}</td><td>{rf.get('fail_points', '--')}</td><td>{pct(rf.get('point_rate'))}</td></tr>",
            f"<tr><td class='wd-header'>总测量面积(模拟墙面)</td><td class='wd-header'>合格面积(模拟墙面)</td><td class='wd-header'>不合格面积(模拟墙面)</td><td class='wd-header'>合格率</td></tr>"
            f"<tr><td>{cell(gf.get('total_area_m2'))}</td><td>{cell(gf.get('pass_area_m2'))}</td><td>{cell(gf.get('fail_area_m2'))}</td><td>{pct(gf.get('area_rate'))}</td></tr>",
            f"<tr><td class='wd-header'>总测量点数(模拟墙面)</td><td class='wd-header'>合格点数(模拟墙面)</td><td class='wd-header'>不合格点数(模拟墙面)</td><td class='wd-header'>合格率</td></tr>"
            f"<tr><td>{gf.get('total_points', '--')}</td><td>{gf.get('pass_points', '--')}</td><td>{gf.get('fail_points', '--')}</td><td>{pct(gf.get('point_rate'))}</td></tr>",
        ]

        return f"<table class='wall-data'>{''.join(rows)}</table>"

    # ------------------------------------------------------------------
    # Analysis text paragraph
    # ------------------------------------------------------------------
    @staticmethod
    def _analysis_text(wd: dict) -> str:
        rf = wd.get("ruler_flatness", {})
        rv = wd.get("ruler_verticality", {})
        gf = wd.get("global_plane_flatness", {})
        gv = wd.get("global_plane_verticality", {})

        def line(label, data, metric_label):
            rate = data.get("area_rate")
            area = data.get("pass_area_m2")
            pts = data.get("pass_points")
            avg = data.get("avg_mm")
            maxv = data.get("max_mm")
            rate_str = f"{rate*100:.1f}%" if rate is not None else "--"
            area_str = f"{area:.2f}m²" if area is not None else "--"
            pts_str = str(pts) if pts is not None else "--"
            avg_str = f"{avg:.2f}mm" if avg is not None else "--"
            max_str = f"{maxv:.2f}mm" if maxv is not None else "--"
            extreme_label = "最大间隙" if "平整度" in label else "最大偏差"
            avg_label = "平均间隙" if "平整度" in label else "平均偏差"
            return (
                f"{label}【{rate_str}】+合格面积【{area_str}】+合格点数【{pts_str}】"
                f"+{avg_label}【{avg_str}】+{extreme_label}【{max_str}】"
            )

        r_overall = wd.get("ruler_overall_rate")
        g_overall = wd.get("global_plane_overall_rate")
        r_overall_str = f"{r_overall*100:.1f}%" if r_overall is not None else "--"
        g_overall_str = f"{g_overall*100:.1f}%" if g_overall is not None else "--"

        text = (
            f"具体分析：\n"
            f"此墙整体合格率：【{r_overall_str}】(模拟下尺)；【{g_overall_str}】(模拟墙面)\n"
            f"{line('平整度面积合格率(模拟下尺)', rf, '平整度')}\n"
            f"{line('平整度面积合格率(模拟墙面)', gf, '平整度')}\n"
            f"{line('垂直度面积合格率(模拟下尺)', rv, '垂直度')}\n"
            f"{line('垂直度面积合格率(模拟墙面)', gv, '垂直度')}\n"
        )
        return escape(text).replace("\n", "<br/>")

    # ------------------------------------------------------------------
    # Image triplets (4 groups x 3 images)
    # ------------------------------------------------------------------
    @staticmethod
    def _triplets(images: list, facade_no: int) -> str:
        groups = [
            ("ruler_flatness_area", "平整度面积合格率(模拟下尺)"),
            ("ruler_verticality_area", "垂直度面积合格率(模拟下尺)"),
            ("global_plane_flatness_area", "平整度面积合格率(模拟墙面)"),
            ("global_plane_verticality_area", "垂直度面积合格率(模拟墙面)"),
        ]

        parts = []
        for mode, title in groups:
            group_images = {
                img.get("key"): img for img in images if img.get("mode") == mode
            }

            def img_tag(key, default_title):
                img = group_images.get(key)
                if img and img.get("path"):
                    return f"<img src='{Path(img['path']).as_uri()}' alt='{escape(default_title)}'/>"
                return f"<div class='triplet-placeholder'>{escape(default_title)}<br/>（未生成）</div>"

            parts.append(
                f"<h3>{escape(title)}</h3>"
                f"<table class='triplet'>"
                f"<tr>"
                f"<td><div class='triplet-title'>点云立面叠加原始热力映射</div>{img_tag('overlay', 'Overlay')}</td>"
                f"<td><div class='triplet-title'>独立热力图 + 0.5m网格</div>{img_tag('heatmap_grid', 'Heatmap Grid')}</td>"
                f"<td><div class='triplet-title'>2D照片 + 热力对齐叠加</div>{img_tag('photo', '2D Photo Overlay（未接入）')}</td>"
                f"</tr>"
                f"</table>"
            )

        return "".join(parts)

    # ------------------------------------------------------------------
    # PDF output
    # ------------------------------------------------------------------
    @staticmethod
    def write_pdf(html: str, path) -> None:
        output = Path(path).expanduser().resolve()
        if output.suffix.lower() != ".pdf":
            output = output.with_suffix(".pdf")
        output.parent.mkdir(parents=True, exist_ok=True)
        document = QTextDocument()
        document.setHtml(html)
        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
        printer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
        printer.setOutputFileName(str(output))
        document.print_(printer)
        if not output.is_file() or output.stat().st_size == 0:
            raise OSError(f"PDF 文件生成失败：{output}")