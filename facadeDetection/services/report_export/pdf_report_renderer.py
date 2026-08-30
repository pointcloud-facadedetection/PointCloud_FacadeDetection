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


class PdfReportRenderer:
    @staticmethod
    def html(snapshot: dict) -> str:
        project = snapshot.get("project") or {}
        info = [("项目名称", project.get("name")), ("所属单位", project.get("org_unit")),
                ("项目地址", project.get("address")), ("楼栋号信息", project.get("building_floor")),
                ("备注", project.get("remarks"))]
        info_html = "".join(f"<tr><th>{escape(label)}</th><td>{_text(value)}</td></tr>" for label, value in info)
        sections = []
        for facade in snapshot.get("facades", []):
            quality = facade.get("quality") or {}
            overall = quality.get("overall") or {}
            profile = quality.get("profile_snapshot") or {}
            width, height = PdfReportRenderer._dimensions(facade)
            thresholds = quality.get("thresholds") or {}
            metrics = [("立面编号", facade.get("report_no")),
                       ("立面面积", PdfReportRenderer._number(facade.get("area"), "m²")),
                       ("检测标准", f"{profile.get('standard_name') or '未指定'} {profile.get('version') or ''}".strip()),
                       ("平整度阈值", PdfReportRenderer._number(thresholds.get("flatness_limit_mm"), "mm")),
                       ("垂直度阈值", PdfReportRenderer._number(thresholds.get("verticality_limit_mm"), "mm")),
                       ("平整度合格率", PdfReportRenderer._percent(overall.get("flatness_pass_rate"))),
                       ("垂直度合格率", PdfReportRenderer._percent(overall.get("verticality_pass_rate"))),
                       ("质量状态", "检测完成" if quality else "未检测")]
            detail_rows = []
            parameters = quality.get("parameters") or {}
            key_metrics = [("质量有效窗口数", overall.get("quality_valid_window_count"), "个"),
                           ("检测窗口总数", overall.get("window_count") or overall.get("total_window_count"), "个"),
                           ("平整度最大间隙", overall.get("flatness_max_gap_mm"), "mm"),
                           ("平整度原始最大间隙", overall.get("flatness_raw_max_gap_mm"), "mm"),
                           ("垂直度最大偏差", overall.get("verticality_deviation_mm"), "mm"),
                           ("检测靠尺长度", parameters.get("ruler_length_m"), "m")]
            for label, value, unit in key_metrics:
                if value is not None:
                    detail_rows.append(f"<tr><th>{escape(label)}</th><td>{_text(PdfReportRenderer._number(value, unit))}</td></tr>")
            metric_html = "".join(f"<tr><th>{escape(label)}</th><td>{_text(value)}</td></tr>" for label, value in metrics)
            image_parts = []
            for image in facade.get("images", []):
                if isinstance(image, dict):
                    path, title = image.get("path"), image.get("title") or "热力图"
                else:
                    path, title = image, "热力图"
                if path:
                    image_parts.append(f"<figure><figcaption>{escape(str(title))}</figcaption>"
                                       f"<img src='{Path(path).as_uri()}' /></figure>")
            images = "".join(image_parts)
            image_markup = images or "<p class='muted'>暂无平整度/垂直度热力图结果</p>"
            detail_html = "<table>" + "".join(detail_rows) + "</table>" if detail_rows else "<p class='muted'>暂无详细检测数据</p>"
            sections.append(f"<section><h2>立面 {facade['report_no']}</h2><table class='facade-overview'>{metric_html}</table>"
                            f"<h3>关键检测数据</h3>{detail_html}<h3>可视化结果</h3>"
                            f"<div class='images'>{image_markup}</div></section>")
        return f"""<!doctype html><html><head><meta charset='utf-8'><style>
        @page {{ size: A4; margin: 12mm 13mm 12mm; }} body {{ font-family: 'Microsoft YaHei','SimSun'; color:#1f2937; font-size:9pt; line-height:1.25; }}
        h1 {{ color:#163a63; font-size:19pt; margin:0 0 5mm; border-bottom:2px solid #2f75b5; padding-bottom:4px; }} h2 {{ color:#163a63; background:#eaf2fb; padding:5px 7px; margin:7px 0 5px; border-left:4px solid #2f75b5; }} h3 {{ color:#365b7d; font-size:10pt; margin:5px 0 3px; }}
        table {{ width:100%; border-collapse:collapse; margin:3px 0 7px; }} th,td {{ border:1px solid #cbd5e1; padding:3px 5px; text-align:left; vertical-align:top; }} th {{ background:#f1f5f9; width:22%; }} pre {{ white-space:pre-wrap; font-family:'Microsoft YaHei','SimSun'; font-size:8pt; margin:0; }} section {{ page-break-inside:auto; }} .images {{ display:flex; gap:8px; align-items:flex-start; }} figure {{ margin:0; width:49%; }} figcaption {{ color:#365b7d; font-weight:bold; margin-bottom:2px; }} img {{ display:block; width:100%; max-height:105mm; object-fit:contain; }} .muted {{ color:#64748b; }}
        </style></head><body><h1>建筑外立面质量检测报告</h1><p>报告生成时间：{datetime.now():%Y-%m-%d %H:%M}</p>
        <h2>一、项目基础信息</h2><table>{info_html}</table><h2>二、建筑立面检测结果</h2>{''.join(sections) or '<p class="muted">暂无立面检测结果</p>'}</body></html>"""

    @staticmethod
    def _percent(value):
        try:
            number = float(value)
            if abs(number) <= 1:
                number *= 100
            return f"{number:.1f}%"
        except (TypeError, ValueError):
            return "--"

    @staticmethod
    def _number(value, unit=""):
        try:
            text = f"{float(value):.2f}".rstrip("0").rstrip(".")
            return f"{text} {unit}".strip()
        except (TypeError, ValueError):
            return "--"

    @staticmethod
    def _dimensions(facade):
        bbox = facade.get("bbox") or facade.get("bbox_2d") or {}
        if isinstance(bbox, dict):
            width, height = bbox.get("width"), bbox.get("height")
        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 2:
            width, height = bbox[0], bbox[1]
        else:
            width = height = None
        return (f"{float(width):.2f}" if width is not None else "--",
                f"{float(height):.2f}" if height is not None else "--")

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