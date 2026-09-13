"""Shared HTML template and native Qt PDF renderer for commercial reports."""
from __future__ import annotations
from html import escape
from pathlib import Path
from datetime import datetime
from PySide6.QtGui import QTextDocument, QImage
from PySide6.QtPrintSupport import QPrinter
from PySide6.QtGui import QPageSize
import numpy as np

# ── 图片尺寸限制 ──
_TARGET_IMG_MAX_W = 200
_TARGET_IMG_MAX_H = 300
# 色条单独限制：更宽以防止畸变
_COLORBAR_MAX_W = 80
_COLORBAR_MAX_H = 300


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


def _auto_trim(img: QImage, threshold: int = 255) -> QImage:
    """裁掉图片四周透明或接近纯白的空白边距。

    统一策略：把"透明"或"三通道都接近纯白"判为背景。
    """
    try:
        w, h = img.width(), img.height()
        if w <= 0 or h <= 0:
            return img

        if img.hasAlphaChannel():
            conv = img.convertToFormat(QImage.Format.Format_RGBA8888)
            bpl = conv.bytesPerLine()
            buf = conv.constBits()
            arr = np.frombuffer(buf, np.uint8, h * bpl).reshape(h, bpl)
            rgba = arr[:, :w * 4].reshape(h, w, 4)
            transparent = rgba[:, :, 3] < 16
            near_white = (
                (rgba[:, :, 0] > threshold)
                & (rgba[:, :, 1] > threshold)
                & (rgba[:, :, 2] > threshold)
            )
            content = ~(transparent | near_white)
        else:
            conv = img.convertToFormat(QImage.Format.Format_RGB888)
            bpl = conv.bytesPerLine()
            buf = conv.constBits()
            arr = np.frombuffer(buf, np.uint8, h * bpl).reshape(h, bpl)
            rgb = arr[:, :w * 3].reshape(h, w, 3)
            content = ~(
                (rgb[:, :, 0] > threshold)
                & (rgb[:, :, 1] > threshold)
                & (rgb[:, :, 2] > threshold)
            )

        cols = np.where(content.any(axis=0))[0]
        rows = np.where(content.any(axis=1))[0]
        if len(cols) == 0 or len(rows) == 0:
            return img

        x0 = max(0, int(cols[0]) - 2)
        x1 = min(w - 1, int(cols[-1]) + 2)
        y0 = max(0, int(rows[0]) - 2)
        y1 = min(h - 1, int(rows[-1]) + 2)
        nw, nh = x1 - x0 + 1, y1 - y0 + 1
        if nw <= 0 or nh <= 0:
            return img
        return img.copy(x0, y0, nw, nh)
    except Exception:
        return img


def _fit_image_size(path, max_w=_TARGET_IMG_MAX_W, max_h=_TARGET_IMG_MAX_H):
    """读取图片 → 裁白边 → 等比缩放到 [max_w, max_h] 内（允许放大）。 """
    try:
        img = QImage(str(path))
        if img.isNull():
            return None, None
        img = _auto_trim(img)          # 先裁空白
        w, h = img.width(), img.height()
        if w <= 0 or h <= 0:
            return None, None
        scale = min(max_w / float(w), max_h / float(h), 1.0)
        return max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    except Exception:
        return None, None


def _is_colorbar(path) -> bool:
    """启发式判断：宽高比视为色条/legend 图。  """
    try:
        img = QImage(str(path))
        if img.isNull():
            return False
        w, h = img.width(), img.height()
        if w <= 0 or h <= 0:
            return False
        return (w / float(h)) < 0.10
    except Exception:
        return False


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
        style = PdfReportRenderer._style()
        return (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<style>{style}</style></head><body>"
            + cover
            + body
            + "</body></html>"
        )

    # ------------------------------------------------------------------
    # CSS
    # ------------------------------------------------------------------
    @staticmethod
    def _style() -> str:
        return (
            "@page { size:A4; margin:10mm 12mm 10mm; }"
            "body { font-family:'Microsoft YaHei','SimSun',sans-serif;"
            " color:#1f2937; font-size:9pt; line-height:1.35; margin:0; }"
            "h1 { color:#163a63; font-size:16pt; margin:0 0 3mm;"
            " border-bottom:2.5px solid #2f75b5; padding-bottom:2mm;"
            " font-weight:700; text-align:center; }"
            "h2 { color:#163a63; background:#eaf2fb; padding:4px 8px;"
            " margin:0 0 5px; border-left:4px solid #2f75b5;"
            " font-size:11pt; page-break-after:avoid; }"
            "h3 { color:#365b7d; font-size:9pt; margin:2px 0 3px;"
            " font-weight:600; border-bottom:1px solid #e2e8f0;"
            " padding-bottom:1px; page-break-after:avoid; text-align:center; }"

            # ── 封面 ──
            ".cover-wrap { text-align:center; page-break-inside:avoid; padding-top:14mm; }"
            ".cover-title { font-size:17pt; font-weight:700; color:#163a63;"
            " margin:0 auto 10px; padding-bottom:6px;"
            " border-bottom:2px solid #2f75b5; width:80%; line-height:1.5; }"
            ".cover-section-title { font-size:12pt; font-weight:700;"
            " color:#2f75b5; margin:8px 0 6px; text-align:center; }"
            "table.summary-cards { width:70%; margin:0 auto 10px;"
            " border-collapse:separate; border-spacing:6px; }"
            "td.summary-card { background:#f8fafc; border:1px solid #e2e8f0;"
            " padding:8px 4px; text-align:center; vertical-align:middle; }"
            "td.sc-blue  { border-top:3px solid #3b82f6; }"
            "td.sc-green { border-top:3px solid #22c55e; }"
            "td.sc-orange{ border-top:3px solid #f59e0b; }"
            "td.sc-cyan  { border-top:3px solid #06b6d4; }"
            ".sc-number { font-size:15pt; font-weight:700; color:#163a63;"
            " line-height:1.2; }"
            ".sc-label { font-size:7.5pt; color:#64748b; margin-top:2px; }"
            "table.cover-info { width:75%; margin:0 auto;"
            " border-collapse:collapse; font-size:9.5pt; }"
            "table.cover-info td { padding:3px 6px;"
            " vertical-align:middle; text-align:center; }"
            ".ci-label { width:35%; color:#64748b; font-size:9pt; }"
            ".ci-value { width:65%; color:#1e293b; font-weight:500; }"

            # ── 数据表 ──
            "table.wall-data { width:100%; border-collapse:collapse;"
            " font-size:8.5pt; margin:5px 0; }"
            "table.wall-data td { border:1px solid #cbd5e1; padding:3px 5px;"
            " text-align:center; vertical-align:middle; }"
            "table.wall-data td:first-child { background:#f1f5f9;"
            " font-weight:600; color:#475569; }"
            "table.wall-data td:last-child { background:#f8fafc;"
            " font-weight:600; color:#163a63; }"
            ".wd-header { background:#e2e8f0 !important;"
            " font-weight:700; color:#1e293b; }"
            ".analysis-text { font-size:7.5pt; color:#334155;"
            " line-height:1.45; margin:4px 0 6px; padding:5px 7px;"
            " background:#f8fafc; border:1px solid #e2e8f0;"
            " border-radius:4px; page-break-inside:avoid; }"

            # ── 三图组：关键修复 ──
            ".triplet-page { page-break-before:always; }"
            ".triplet-group { page-break-inside:avoid; margin:0 0 6px;"
            " text-align:center; }"
            ".triplet-group:last-child { margin-bottom:0; }"
            # 不用 table-layout:fixed，让内容自然分布
            "table.triplet { width:100%; border-collapse:separate;"
            " border-spacing:2px; margin:0; page-break-inside:avoid; }"
            "table.triplet td { vertical-align:top; text-align:center;"
            " border:1px solid #e2e8f0; padding:2px; background:#fff; }"
            "table.triplet img { display:block; margin:0 auto 1px; }"
            ".triplet-title { font-size:7.5pt; color:#365b7d;"
            " font-weight:600; margin-top:1px; text-align:center; }"
            ".triplet-placeholder { background:#f1f5f9; color:#94a3b8;"
            " font-size:7.5pt; line-height:1.3; text-align:center;"
            " padding:60px 4px; margin:0 auto; }"
            ".triplet-group h3 { margin:1px 0 3px; }"

            ".report-meta { font-size:8pt; color:#94a3b8;"
            " margin-bottom:6px; text-align:right; }"
            ".muted { color:#94a3b8; font-size:8.5pt;"
            " text-align:center; padding:10px; }"
            "section { margin-bottom:0; }"
            ".building { page-break-before:always; }"
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
            ("项目地址", project.get("address")),
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
            "<div class='cover-wrap'>"
            "<div class='cover-spacer'></div>"
            "<div class='cover-title'>"
            "深圳瑞捷技术股份有限公司<br/>外立面激光测量检测报告"
            "</div>"
            "<div class='cover-section-title'>项目检测摘要</div>"
            "<table class='summary-cards'>"
            "<tr>"
            "<td class='summary-card sc-blue'>"
            f"<div class='sc-number'>{total_facades}</div>"
            "<div class='sc-label'>检测立面</div></td>"
            "<td class='summary-card sc-green'>"
            f"<div class='sc-number'>{avg_pass_rate}</div>"
            "<div class='sc-label'>双指标面积参考率</div></td>"
            "</tr><tr>"
            "<td class='summary-card sc-orange'>"
            f"<div class='sc-number'>{total_area}</div>"
            "<div class='sc-label'>检测面积</div></td>"
            "<td class='summary-card sc-cyan'>"
            f"<div class='sc-number'>{total_points}</div>"
            "<div class='sc-label'>检测点数</div></td>"
            "</tr></table>"
            "<table class='cover-info'>"
            f"{info_rows}"
            "</table>"
            "</div>"
        )

    # ------------------------------------------------------------------
    # Body (facades)
    # ------------------------------------------------------------------
    @staticmethod
    def _body(facades: list[dict], buildings: list[dict] | None = None) -> str:
        if not facades:
            return "<p class='muted'>暂无立面检测结果</p>"
        sections = []
        building_groups = buildings or [{"label": "楼栋 1", "walls": facades}]
        for building in building_groups:
            walls = building.get("walls") or []
            if not walls:
                continue

            building_label = _text(building.get("label"), "点云站点")

            sections.append(
                f"<div class='building'>"
                f"<h1>{building_label}</h1>"
            )

            for facade in walls:
                wd = facade.get("wall_data") or {}
                number = facade.get("report_no", facade.get("id", 0))
                table_html = PdfReportRenderer._wall_data_table(wd)
                analysis_html = PdfReportRenderer._analysis_text(wd)
                images = facade.get("images", [])
                triplets_html = PdfReportRenderer._triplets(images, number)
                project_name = _text(facade.get("project_name"), "")
                station_label = _text(building.get("label"), "")
                wall_label = _text(facade.get("wall_label"), f"墙面 {number}")
                chapter_title = (
                    f"{project_name}-{station_label}-{wall_label}"
                    if project_name
                    else f"{station_label}-{wall_label}"
                )
                sections.append(
                    f"<div class='wall-head'>"
                    f"<h2>{chapter_title}</h2>"
                    f"{table_html}"
                    f"{analysis_html}"
                    f"</div>"
                    f"{triplets_html}"
                )

            sections.append("</div>")  # 关闭 building div

        return "".join(sections)

    # ------------------------------------------------------------------
    # Wall data table (5 rows x 4 cols)
    # ------------------------------------------------------------------
    @staticmethod
    def _wall_data_table(wd: dict) -> str:
        rf = wd.get("ruler_flatness", {})
        rv = wd.get("ruler_verticality", {})
        gf = wd.get("global_plane_flatness", {})
        gv = wd.get("global_plane_verticality", {})

        flat_name = rf.get("standard_name") or "未指定标准"
        flat_ver = rf.get("version") or ""
        flat_limit = rf.get("threshold_mm", 4.0)
        vert_name = rv.get("standard_name") or flat_name
        vert_ver = rv.get("version") or flat_ver
        vert_limit = rv.get("threshold_mm", 4.0)
        std = (
            f"{flat_name} {flat_ver}"
            f" [平整度 {flat_limit:.1f}mm / 垂直度 {vert_limit:.1f}mm]"
        ).strip()

        def cell(v):
            if isinstance(v, (int, float)) and v > 10:
                return _num(v, "m²")
            return _num(v, "", 1)

        def pct(v):
            return _pct(v) if v is not None else "--"

        # 平整度和垂直度是两个独立指标，不能把两个百分比的算术平均
        # 伪装成“整体合格率”。表格中的合格率单元格明确同时列出两者。
        def paired_rate(flat_data, vert_data, key):
            return (f"平整度 {_pct(flat_data.get(key))} / "
                    f"垂直度 {_pct(vert_data.get(key))}")

        ruler_area = paired_rate(rf, rv, "area_rate")
        ruler_point = paired_rate(rf, rv, "point_rate")
        global_area = paired_rate(gf, gv, "area_rate")
        global_point = paired_rate(gf, gv, "point_rate")

        rows = [
            # 行1：材质 + 标准
            f"<tr>"
            f"<td>面层材质</td><td>{_text(rf.get('material'))}</td>"
            f"<td>检测标准</td><td>{escape(std)}</td>"
            f"</tr>",
            # 行2：模拟下尺·面积
            f"<tr>"
            f"<td>总测量面积<br/>(模拟下尺)</td>"
            f"<td>合格面积<br/>(模拟下尺)</td>"
            f"<td>不合格面积<br/>(模拟下尺)</td>"
            f"<td>合格率</td>"
            f"</tr>",
            f"<tr>"
            f"<td>{cell(rf.get('total_area_m2'))}</td>"
            f"<td>{cell(rf.get('pass_area_m2'))}</td>"
            f"<td>{cell(rf.get('fail_area_m2'))}</td>"
            f"<td>{ruler_area}</td>"
            f"</tr>",
            # 行3：模拟下尺·点数
            f"<tr>"
            f"<td>总测量点数<br/>(模拟下尺)</td>"
            f"<td>合格点数<br/>(模拟下尺)</td>"
            f"<td>不合格点数<br/>(模拟下尺)</td>"
            f"<td>合格率</td>"
            f"</tr>",
            f"<tr>"
            f"<td>{rf.get('total_points', '--')}</td>"
            f"<td>{rf.get('pass_points', '--')}</td>"
            f"<td>{rf.get('fail_points', '--')}</td>"
            f"<td>{ruler_point}</td>"
            f"</tr>",
            # 行4：模拟墙面·面积
            f"<tr>"
            f"<td>总测量面积<br/>(模拟墙面)</td>"
            f"<td>合格面积<br/>(模拟墙面)</td>"
            f"<td>不合格面积<br/>(模拟墙面)</td>"
            f"<td>合格率</td>"
            f"</tr>",
            f"<tr>"
            f"<td>{cell(gf.get('total_area_m2'))}</td>"
            f"<td>{cell(gf.get('pass_area_m2'))}</td>"
            f"<td>{cell(gf.get('fail_area_m2'))}</td>"
            f"<td>{global_area}</td>"
            f"</tr>",
            # 行5：模拟墙面·点数
            f"<tr>"
            f"<td>总测量点数<br/>(模拟墙面)</td>"
            f"<td>合格点数<br/>(模拟墙面)</td>"
            f"<td>不合格点数<br/>(模拟墙面)</td>"
            f"<td>合格率</td>"
            f"</tr>",
            f"<tr>"
            f"<td>{gf.get('total_points', '--')}</td>"
            f"<td>{gf.get('pass_points', '--')}</td>"
            f"<td>{gf.get('fail_points', '--')}</td>"
            f"<td>{global_point}</td>"
            f"</tr>",
        ]
        return f"<table class='wall-data'>{''.join(rows)}</table>"

    # ------------------------------------------------------------------
    # Analysis text
    # ------------------------------------------------------------------
    @staticmethod
    def _analysis_text(wd: dict) -> str:
        rf = wd.get("ruler_flatness", {})
        rv = wd.get("ruler_verticality", {})
        gf = wd.get("global_plane_flatness", {})
        gv = wd.get("global_plane_verticality", {})

        def line(label, data):
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
            extreme = "最大间隙" if "平整度" in label else "最大偏差"
            avg_lbl = "平均间隙" if "平整度" in label else "平均偏差"
            return (
                f"{label}【{rate_str}】+合格面积【{area_str}】"
                f"+合格点数【{pts_str}】+{avg_lbl}【{avg_str}】"
                f"+{extreme}【{max_str}】"
            )

        text = (
            f"具体分析：\n"
            f"此墙双指标面积参考率（非联合同时合格率）："
            f"【{_pct(wd.get('ruler_overall_rate'))}】(模拟下尺)；"
            f"【{_pct(wd.get('global_plane_overall_rate'))}】(模拟墙面)\n"
            f"{line('平整度面积合格率(模拟下尺)', rf)}\n"
            f"{line('平整度面积合格率(模拟墙面)', gf)}\n"
            f"{line('垂直度面积合格率(模拟下尺)', rv)}\n"
            f"{line('垂直度面积合格率(模拟墙面)', gv)}"
        )
        return (
            f"<div class='analysis-text'>"
            f"{escape(text).replace(chr(10), '<br/>')}"
            f"</div>"
        )

    # ------------------------------------------------------------------
    # Image triplets — 核心修复
    # ------------------------------------------------------------------
    @staticmethod
    def _triplets(images: list, facade_no: int) -> str:
        """输出 4 组三图，按 2 组/页 组织。"""
        groups = [
            ("ruler_flatness_area", "平整度面积合格率(模拟下尺)"),
            ("ruler_verticality_area", "垂直度面积合格率(模拟下尺)"),
            ("global_plane_flatness_area", "平整度面积合格率(模拟墙面)"),
            ("global_plane_verticality_area", "垂直度面积合格率(模拟墙面)"),
        ]

        def render_group(mode: str, title: str) -> str:
            group_images = {
                img.get("key"): img
                for img in images
                if img.get("mode") == mode
            }

            def img_tag(key, default_title):
                img = group_images.get(key)
                if not (img and img.get("path")):
                    return (
                        "<div class='triplet-placeholder'"
                        " style='width:150px;height:280px;'>"
                        f"{escape(default_title)}<br/>（未生成）"
                        "</div>"
                    )
                path = img["path"]
                # 色条用更宽的约束，防止畸变
                if _is_colorbar(path):
                    w, h = _fit_image_size(
                        path,
                        max_w=_COLORBAR_MAX_W,
                        max_h=_COLORBAR_MAX_H,
                    )
                else:
                    w, h = _fit_image_size(path)
                if w is None:
                    return (
                        "<div class='triplet-placeholder'"
                        " style='width:150px;height:280px;'>"
                        f"{escape(default_title)}<br/>（读取失败）"
                        "</div>"
                    )
                return (
                    f"<img src='{Path(path).as_uri()}'"
                    f" width='{w}' height='{h}'"
                    f" alt='{escape(default_title)}'/>"
                )

            return (
                "<div class='triplet-group'>"
                f"<h3>{escape(title)}</h3>"
                "<table class='triplet'>"
                "<tr>"
                f"<td>{img_tag('overlay', 'Overlay')}"
                "<div class='triplet-title'>点云立面热力映射图</div></td>"
                f"<td>{img_tag('heatmap_grid', 'Heatmap Grid')}"
                "<div class='triplet-title'>独立热力网格图</div></td>"
                f"<td>{img_tag('photo', '2D Photo Overlay')}"
                "<div class='triplet-title'>2D现场热力映射图</div></td>"
                "</tr></table>"
                "</div>"
            )

        # 每页 2 组
        page_1 = render_group(*groups[0]) + render_group(*groups[1])
        page_2 = render_group(*groups[2]) + render_group(*groups[3])
        return (
            f"<div class='triplet-page'>{page_1}</div>"
            f"<div class='triplet-page'>{page_2}</div>"
        )

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