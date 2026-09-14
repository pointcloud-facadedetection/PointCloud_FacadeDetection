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
_TARGET_IMG_MAX_H = 280
# 色条单独限制
_COLORBAR_MAX_W = 96
_COLORBAR_MAX_H = 280
# 色条与相邻热力图之间的最小间隙
_COLORBAR_GAP_PX = 12

# A4 版心宽度
_A4_CONTENT_W_PX = 688
# 2 组图/页 的硬约束下，单组图片可用高度。
_TRIPLET_ROW_MAX_H = 280
# 表格边框与单元格内边距在三列上占用的横向像素。
_TRIPLET_BORDER_PX = 24


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
        footer = PdfReportRenderer._footer(project, summary)
        style = PdfReportRenderer._style()
        return (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<style>{style}</style></head><body>"
            + cover
            + body
            + footer
            + "</body></html>"
        )

    # ------------------------------------------------------------------
    # Footer page: 结论与声明
    # ------------------------------------------------------------------
    @staticmethod
    def _footer(project: dict, summary: dict) -> str:
        """报告结尾的结论摘要、声明与签章表。

        只使用 snapshot 中已存在的字段，缺字段按占位符输出，
        不引入任何新的数据来源，保证与既有业务数据结构完全兼容。
        """
        report_date = project.get("report_date")
        rdate = (
            report_date.strftime("%Y年%m月%d日")
            if hasattr(report_date, "strftime")
            else _text(report_date)
        )
        total_facades = summary.get("total_facades", 0)
        avg_pass_rate = summary.get("avg_pass_rate", "--")
        total_area = summary.get("total_area", 0)
        total_points = summary.get("total_points", 0)
        conclusion = (
            "本项目共检测外立面 "
            f"{escape(str(total_facades))} 面，"
            f"累计检测面积 {escape(str(total_area))}，"
            f"累计检测点数 {escape(str(total_points))}，"
            f"双指标面积参考合格率 {escape(str(avg_pass_rate))}。"
            "平整度与垂直度按现行标准分别判定，"
            "具体分部数据见各立面章节。"
        )
        return (
            "<div class='report-footer'>"
            "<h1>结论与声明</h1>"
            "<h2>检测结论</h2>"
            f"<div class='statement'>{conclusion}</div>"
            "<h2>报告声明</h2>"
            "<div class='statement'>"
            "1. 本报告依据三维激光扫描点云自动分析结果编制，"
            "检测数据仅对本次受检部位负责。<br/>"
            "2. 报告中平整度、垂直度分别独立判定，"
            "所列合格率均为单指标统计值，不作为联合验收结论。<br/>"
            "3. 报告中各项数据均以毫米（mm）为单位。<br/>"
            "</div>"
            "</div>"
        )

    # ------------------------------------------------------------------
    # CSS
    # ------------------------------------------------------------------
    @staticmethod
    def _style() -> str:
        return (
            # 页边距预留装订边与页脚空间，符合 A4 工业检测报告常规版式。
            "@page { size:A4; margin:14mm 13mm 16mm 15mm; }"
            "body { font-family:'Microsoft YaHei','SimSun',sans-serif;"
            " color:#1f2937; font-size:9.5pt; line-height:1.42; margin:0; }"
            "p { margin:0 0 4px; }"
            "h1 { color:#163a63; font-size:16pt; margin:0 0 4mm;"
            " border-bottom:2.5px solid #2f75b5; padding-bottom:2mm;"
            " font-weight:700; text-align:center; }"
            # 二级标题：左侧色条 + 淡底，页内不跨页断裂。
            "h2 { color:#163a63; background:#eaf2fb; padding:5px 8px;"
            " margin:6px 0 6px; border-left:4px solid #2f75b5;"
            " font-size:11pt; page-break-after:avoid; }"
            "h3 { color:#365b7d; font-size:9.5pt; margin:3px 0 4px;"
            " font-weight:600; border-bottom:1px solid #e2e8f0;"
            " padding-bottom:2px; page-break-after:avoid; text-align:center; }"

            # ── 封面 ──
            ".cover-wrap { text-align:center; page-break-inside:avoid;"
            " padding:6mm 0 0; }"
            # 顶部机构标识带：工业报告的"抬头"，比裸标题更规范。
            ".cover-org-band { border-bottom:3px solid #1d4f91;"
            " padding-bottom:4mm; margin-bottom:8mm; }"
            ".cover-org { font-size:13pt; font-weight:700; color:#1d4f91;"
            " letter-spacing:2px; }"
            ".cover-org-en { font-size:7.5pt; color:#64748b;"
            " letter-spacing:1px; margin-top:2px; }"
            ".cover-title { font-size:22pt; font-weight:700; color:#163a63;"
            " margin:0 auto 4mm; padding-bottom:3mm;"
            " border-bottom:2px solid #2f75b5; width:86%; line-height:1.4;"
            " letter-spacing:4px; }"
            ".cover-doc-code { font-size:9.5pt; color:#475569;"
            " margin-bottom:10mm; letter-spacing:1px; }"
            ".cover-section-title { font-size:11.5pt; font-weight:700;"
            " color:#2f75b5; margin:7mm 0 4mm; text-align:center; }"
            "table.summary-cards { width:74%; margin:0 auto 8mm;"
            " border-collapse:separate; border-spacing:7px; }"
            "td.summary-card { background:#f8fafc; border:1px solid #dbe3ed;"
            " padding:10px 4px; text-align:center; vertical-align:middle; }"
            "td.sc-blue  { border-top:3px solid #3b82f6; }"
            "td.sc-green { border-top:3px solid #22c55e; }"
            "td.sc-orange{ border-top:3px solid #f59e0b; }"
            "td.sc-cyan  { border-top:3px solid #06b6d4; }"
            ".sc-number { font-size:16pt; font-weight:700; color:#163a63;"
            " line-height:1.2; }"
            ".sc-label { font-size:7.5pt; color:#64748b; margin-top:3px; }"
            # 封面信息表加完整边框，形成"项目信息卡"，比悬浮文字更正式。
            "table.cover-info { width:80%; margin:0 auto;"
            " border-collapse:collapse; font-size:9.5pt;"
            " border:1px solid #cbd5e1; }"
            "table.cover-info td { padding:4px 8px;"
            " vertical-align:middle; text-align:center;"
            " border:1px solid #e2e8f0; }"
            ".ci-label { width:32%; color:#475569; font-size:9pt;"
            " background:#f1f5f9; font-weight:600; }"
            ".ci-value { width:68%; color:#1e293b; font-weight:500; }"
            # 封面底部签章区：标准化报告的必要收尾元素。
            ".cover-signature { width:80%; margin:12mm auto 0;"
            " font-size:9pt; color:#334155; text-align:left;"
            " border-top:1px solid #cbd5e1; padding-top:5px;"
            " line-height:1.7; }"
            ".cover-note { color:#64748b; font-size:8pt; }"

            # ── 数据表 ──
            "table.wall-data { width:100%; border-collapse:collapse;"
            " font-size:8.5pt; margin:6px 0 8px; }"
            "table.wall-data td { border:1px solid #cbd5e1; padding:4px 5px;"
            " text-align:center; vertical-align:middle; }"
            "table.wall-data td:first-child { background:#f1f5f9;"
            " font-weight:600; color:#475569; }"
            "table.wall-data td:last-child { background:#f8fafc;"
            " font-weight:600; color:#163a63; }"
            # 指标分组表头：深蓝底白字，一眼区分"模拟下尺 / 模拟墙面"。
            ".wd-header { background:#1d4f91 !important;"
            " font-weight:700; color:#ffffff !important;"
            " font-size:8.5pt; padding:5px 4px !important; }"
            # 分组小标题行：淡蓝底居中，作为数据块之间的分节标识。
            ".wd-group { background:#e8f0fb !important;"
            " font-weight:700; color:#1d4f91 !important;"
            " letter-spacing:1px; }"
            ".analysis-text { font-size:8pt; color:#334155;"
            " line-height:1.5; margin:4px 0 7px; padding:6px 8px;"
            " background:#f8fafc; border:1px solid #e2e8f0;"
            " border-left:3px solid #2f75b5;"
            " border-radius:3px; page-break-inside:avoid; }"

            # ── 三图组 ──
            ".triplet-page { page-break-before:always; }"
            ".triplet-group { page-break-inside:avoid; margin:0 0 6px;"
            " text-align:center; }"
            ".triplet-group:last-child { margin-bottom:0; }"
            # 用 table-layout:fixed + 显式列宽，让三列严格按 _column_widths
            # 的预算铺满版心，不受单张图片宽高比影响而互相挤压。
            "table.triplet { width:100%; table-layout:fixed;"
            " border-collapse:separate; border-spacing:2px; margin:0;"
            " page-break-inside:avoid; }"
            "table.triplet td { vertical-align:top; text-align:center;"
            " border:1px solid #e2e8f0; padding:2px; background:#fff;"
            " overflow:hidden; }"
            "table.triplet img { display:block; margin:0 auto 1px;"
            " max-width:100%; }"
            # 色条列：左侧留出可见间隙，避免色条紧贴相邻热力图
            "table.triplet td.triplet-bar { padding-left:12px;"
            " padding-right:6px; }"
            ".triplet-title { font-size:7.5pt; color:#365b7d;"
            " font-weight:600; margin-top:1px; text-align:center;"
            " white-space:nowrap; overflow:hidden;"
            " text-overflow:ellipsis; }"
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
            # 报告结尾的结论与声明区，补齐标准化报告的固定结构。
            ".report-footer { page-break-before:always; }"
            ".report-footer table.sign-table { width:100%;"
            " border-collapse:collapse; font-size:9pt; margin-top:8px; }"
            ".report-footer table.sign-table td {"
            " border:1px solid #cbd5e1; padding:8px;"
            " text-align:left; vertical-align:top; height:58px; }"
            ".report-footer table.sign-table td.sign-role {"
            " width:20%; background:#f1f5f9; font-weight:600;"
            " color:#475569; text-align:center; vertical-align:middle; }"
            ".report-footer .statement { font-size:8.5pt; color:#334155;"
            " line-height:1.6; margin-top:8px; padding:8px 10px;"
            " background:#f8fafc; border:1px solid #e2e8f0; }"
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

        org = project.get("inspection_unit") or "深圳瑞捷技术股份有限公司"
        report_no = project.get("report_no")

        return (
            "<div class='cover-wrap'>"
            # 顶部机构抬头带：标准化报告的固定视觉锚点。
            "<div class='cover-org-band'>"
            f"<div class='cover-org'>{escape(str(org))}</div>"
            "<div class='cover-org-en'>FACADE LASER SCANNING INSPECTION</div>"
            "</div>"
            "<div class='cover-title'>外立面激光测量检测报告</div>"
            f"<div class='cover-doc-code'>报告编号：{_text(report_no)}</div>"
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
            # 封面签章区：说明报告用途与生效条件，收尾更规范。
            "<div class='cover-signature'>"
            f"<div>检测机构：{_text(org)}</div>"
            "<div>报告说明：本报告数据来源于三维激光扫描点云自动分析，"
            "检测结果仅对本次受检部位有效。</div>"
            "<div>签发日期："
            f"{rdate if rdate != '--' else datetime.now().strftime('%Y年%m月%d日')}"
            "</div>"
            "</div>"
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
            # 行1：材质 + 标准（表头样式，突出关键判定依据）
            f"<tr class='wd-header'>"
            f"<td>面层材质</td><td>{_text(rf.get('material'))}</td>"
            f"<td>检测标准</td><td>{escape(std)}</td>"
            f"</tr>",
            # 行2：模拟下尺·面积
            f"<tr class='wd-group'><td colspan='4'>一、模拟下尺（靠尺法）</td></tr>",
            f"<tr class='wd-header'>"
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
            f"<tr class='wd-header'>"
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
            f"<tr class='wd-group'><td colspan='4'>二、模拟墙面（面域法）</td></tr>",
            f"<tr class='wd-header'>"
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
            f"<tr class='wd-header'>"
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
    # Image triplets
    # ------------------------------------------------------------------
    @staticmethod
    def _column_widths(cells: list) -> list:
        """按"铺满 A4 版心"反推每列图片的目标宽度。
        """
        n = len(cells) or 1
        n_bars = sum(1 for _, is_bar in cells if is_bar)
        n_normals = max(1, n - n_bars)
        # 色条列加宽后自身占位；其余列分摊剩余宽度。
        bar_budget = n_bars * (_COLORBAR_MAX_W + _COLORBAR_GAP_PX)
        normal_budget = max(
            40, (_A4_CONTENT_W_PX - _TRIPLET_BORDER_PX - bar_budget) // n_normals)
        sizes = []
        for _, is_bar in cells:
            if is_bar:
                sizes.append((_COLORBAR_MAX_W, _TRIPLET_ROW_MAX_H))
            else:
                sizes.append((normal_budget, _TRIPLET_ROW_MAX_H))
        return sizes

    @staticmethod
    def _triplets(images: list, facade_no: int) -> str:
        """输出 4 组三图，按 2 组/页 组织。

        版面契约：每组一页两行，行内图片按 heatmap 实际宽度自适应铺满
        A4 版心宽度（最多 3 图 + 边框），色条单独成列并留间隙。
        """
        groups = [
            ("ruler_flatness_area", "平整度面积合格率(模拟下尺)"),
            ("ruler_verticality_area", "垂直度面积合格率(模拟下尺)"),
            ("global_plane_flatness_area", "平整度面积合格率(模拟墙面)"),
            ("global_plane_verticality_area", "垂直度面积合格率(模拟墙面)"),
        ]

        # 列定义保持"两组图同一页"的高度契约不变，仅让横向宽度自适应。
        columns = (
            ('overlay', 'Overlay', '点云立面热力映射图'),
            ('heatmap_grid', 'Heatmap Grid', '独立热力网格图'),
            ('photo', '2D Photo Overlay', '2D现场热力映射图'),
        )

        def render_group(mode: str, title: str) -> str:
            group_images = {
                img.get("key"): img
                for img in images
                if img.get("mode") == mode
            }

            # 先探测每列是否为色条，再按版心宽度分配各列上限。
            cell_paths = []
            for key, _, _ in columns:
                img = group_images.get(key)
                path = img.get("path") if img else None
                cell_paths.append((path, bool(path) and _is_colorbar(path)))
            limits = PdfReportRenderer._column_widths(cell_paths)

            cells_html = []
            for (key, default_title, caption), (path, is_bar), (max_w, max_h) in zip(
                columns, cell_paths, limits
            ):
                if not path:
                    cells_html.append(
                        "<td>"
                        "<div class='triplet-placeholder'"
                        " style='width:150px;height:280px;'>"
                        f"{escape(default_title)}<br/>（未生成）"
                        "</div>"
                        f"<div class='triplet-title'>{escape(caption)}</div>"
                        "</td>"
                    )
                    continue
                w, h = _fit_image_size(path, max_w=max_w, max_h=max_h)
                if w is None:
                    cells_html.append(
                        "<td>"
                        "<div class='triplet-placeholder'"
                        " style='width:150px;height:280px;'>"
                        f"{escape(default_title)}<br/>（读取失败）"
                        "</div>"
                        f"<div class='triplet-title'>{escape(caption)}</div>"
                        "</td>"
                    )
                    continue
                # 色条列额外加左内边距，与相邻热力图拉开可见间隙。
                td_cls = " class='triplet-bar'" if is_bar else ""
                cells_html.append(
                    f"<td{td_cls}>"
                    f"<img src='{Path(path).as_uri()}'"
                    f" width='{w}' height='{h}'"
                    f" alt='{escape(default_title)}'/>"
                    f"<div class='triplet-title'>{escape(caption)}</div>"
                    "</td>"
                )

            # table-layout:fixed 下必须给出显式列宽，否则三列等宽会把
            # 按版心预算放大的热力图裁掉。列宽与 _column_widths 的预算一致，
            # 色条列按其自身宽度收窄，剩余宽度全部分给热力图列。
            colgroup = "<colgroup>" + "".join(
                f"<col width='{w}'/>" for w, _ in limits
            ) + "</colgroup>"
            return (
                "<div class='triplet-group'>"
                f"<h3>{escape(title)}</h3>"
                "<table class='triplet'>"
                + colgroup
                + "<tr>" + "".join(cells_html) + "</tr>"
                "</table>"
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