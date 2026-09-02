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

        # ============================================
        # 项目信息 - 3列2行内嵌table
        # ============================================
        info_items = [
            ("项目名称", project.get("name")),
            ("所属单位", project.get("org_unit")),
            ("项目地址", project.get("address")),
            ("楼栋号信息", project.get("building_floor")),
            ("备注", project.get("remarks")),
            ("报告编号", project.get("project_id")),
        ]
        info_rows = []
        for i in range(0, len(info_items), 3):
            row_cells = ""
            for j in range(3):
                idx = i + j
                if idx < len(info_items):
                    label, value = info_items[idx]
                    row_cells += (
                        f"<td class='info-cell'>"
                        f"<div class='info-label'>{escape(label)}</div>"
                        f"<div class='info-value'>{_text(value)}</div></td>"
                    )
                else:
                    row_cells += "<td class='info-cell'></td>"
            info_rows.append(f"<tr>{row_cells}</tr>")
        info_html = f"<table class='info-grid'>{''.join(info_rows)}</table>"

        # ============================================
        # 统计摘要 - 2列2行内嵌table
        # ============================================
        total_facades = len(snapshot.get("facades", []))
        avg_pass_rate = PdfReportRenderer._calc_avg_pass_rate(snapshot)
        total_windows = PdfReportRenderer._calc_total_windows(snapshot)
        total_points = PdfReportRenderer._calc_total_points(snapshot)
        
        summary_html = f"""
        <table class='summary-panel'>
            <tr>
                <td class='summary-card sc-blue'>
                    <div class='summary-number'>{total_facades}</div>
                    <div class='summary-label'>检测立面</div>
                </td>
                <td class='summary-card sc-green'>
                    <div class='summary-number' style='color:#15803d;'>{avg_pass_rate}</div>
                    <div class='summary-label'>平均合格率</div>
                </td>
            </tr>
            <tr>
                <td class='summary-card sc-orange'>
                    <div class='summary-number' style='color:#b45309;'>{total_windows}</div>
                    <div class='summary-label'>有效窗口</div>
                </td>
                <td class='summary-card sc-cyan'>
                    <div class='summary-number' style='color:#0369a1;'>{total_points}</div>
                    <div class='summary-label'>立面点数</div>
                </td>
            </tr>
        </table>
        """

        # ============================================
        # 修复: 外层table将两者左右并排
        # ============================================
        top_section_html = f"""
        <table class='top-layout'>
            <tr>
                <td class='top-left'>
                    <h3 style='margin-top:0;'>项目基础信息</h3>
                    {info_html}
                </td>
                <td class='top-right'>
                    <h3 style='margin-top:0;'>项目检测摘要</h3>
                    {summary_html}
                </td>
            </tr>
        </table>
        """

        sections = []
        for facade in snapshot.get("facades", []):
            quality = facade.get("quality") or {}
            overall = quality.get("overall") or {}
            profile = quality.get("profile_snapshot") or {}
            width, height = PdfReportRenderer._dimensions(facade)
            thresholds = quality.get("thresholds") or {}
            
            overview_metrics = [
                ("立面编号", facade.get("report_no")),
                ("立面面积", PdfReportRenderer._number(facade.get("area"), "m²")),
                ("检测标准", f"{profile.get('standard_name') or '未指定'} {profile.get('version') or ''}".strip()),
                ("平整度阈值", PdfReportRenderer._number(thresholds.get("flatness_limit_mm"), "mm")),
                ("垂直度阈值", PdfReportRenderer._number(thresholds.get("verticality_limit_mm"), "mm")),
                ("平整度合格率", PdfReportRenderer._percent(overall.get("flatness_pass_rate"))),
                ("垂直度合格率", PdfReportRenderer._percent(overall.get("verticality_pass_rate"))),
                ("质量状态", "检测完成" if quality else "未检测")
            ]
            overview_html = "".join(
                f"<tr><td class='ov-label'>{escape(label)}</td>"
                f"<td class='ov-value'>{_text(value)}</td></tr>"
                for label, value in overview_metrics)
            
            parameters = quality.get("parameters") or {}
            key_metrics = [
                ("质量有效窗口数", overall.get("quality_valid_window_count"), "个"),
                ("检测窗口总数", overall.get("window_count") or overall.get("total_window_count"), "个"),
                ("平整度最大间隙", overall.get("flatness_max_gap_mm"), "mm"),
                ("平整度窗口平均间隙", overall.get("flatness_avg_gap_mm"), "mm"),
                ("平整度原始最大间隙", overall.get("flatness_raw_max_gap_mm"), "mm"),
                ("垂直度最大偏差", overall.get("verticality_deviation_mm"), "mm"),
                ("垂直度窗口平均偏差", overall.get("verticality_avg_deviation_mm"), "mm"),
                ("检测靠尺长度", parameters.get("ruler_length_m"), "m")
            ]
            detail_rows = []
            for label, value, unit in key_metrics:
                if value is not None:
                    val_text = PdfReportRenderer._number(value, unit)
                    is_alert = False
                    try:
                        if "最大间隙" in label and float(value) > 50:
                            is_alert = True
                        elif "最大偏差" in label and float(value) > 50:
                            is_alert = True
                    except (TypeError, ValueError):
                        pass
                    alert_class = "alert-value" if is_alert else ""
                    detail_rows.append(
                        f"<tr><td class='dt-label'>{escape(label)}</td>"
                        f"<td class='dt-value {alert_class}'>{_text(val_text)}</td></tr>")
            detail_html = "".join(detail_rows) if detail_rows else "<tr><td colspan='2' class='muted'>暂无详细检测数据</td></tr>"

            image_parts = []
            for image in facade.get("images", []):
                if isinstance(image, dict):
                    path, title = image.get("path"), image.get("title") or "热力图"
                    mode = image.get("mode", "")
                else:
                    path, title = image, "热力图"
                    mode = ""
                if path:
                    limit_val = thresholds.get(f"{mode}_limit_mm", 4.0) if mode else 4.0
                    max_val = overall.get(f"{mode}_max_gap_mm", limit_val) if mode else limit_val
                    
                    image_parts.append(
                        f"<td class='image-cell'>"
                        f"<div class='image-card'>"
                        f"<div class='image-title'>{escape(str(title))}</div>"
                        f"<div class='image-frame'>"
                        f"<img src='{Path(path).as_uri()}' /></div>"
                        f"<div class='image-legend'>"
                        f"<div class='legend-bar'></div>"
                        f"<div class='legend-labels'>"
                        f"<span>合格 &lt;{float(limit_val):.1f}mm</span>"
                        f"<span>严重 &gt;{float(max_val):.1f}mm</span></div></div></div></td>"
                    )
            
            images_html = "".join(image_parts)
            image_markup = (
                f"<table class='image-gallery'><tr>{images_html}</tr></table>"
                if images_html 
                else "<p class='muted'>暂无平整度/垂直度热力图结果</p>"
            )

            sections.append(f"""
            <section>
                <div class='facade-header'>
                    <h2>立面 {facade['report_no']}</h2>
                    <span class='facade-dim'>{width} &times; {height}</span>
                </div>
                <table class='data-columns'>
                    <tr>
                        <td class='column-left'>
                            <h3>立面概览</h3>
                            <table class='overview-table'>{overview_html}</table>
                        </td>
                        <td class='column-right'>
                            <h3>关键检测数据</h3>
                            <table class='detail-table'>{detail_html}</table>
                        </td>
                    </tr>
                </table>
                <h3>可视化结果</h3>
                {image_markup}
            </section>
            """)

        return f"""<!doctype html><html><head><meta charset='utf-8'><style>
        @page {{ size:A4; margin:12mm 14mm 14mm; }}
        
        body {{ 
            font-family:'Microsoft YaHei','SimSun',sans-serif; 
            color:#1f2937; 
            font-size:9pt; 
            line-height:1.4; 
            margin:0;
        }}
        
        h1 {{ 
            color:#163a63; 
            font-size:18pt; 
            margin:0 0 3mm; 
            border-bottom:2.5px solid #2f75b5; 
            padding-bottom:3mm; 
            font-weight:700;
        }}
        h2 {{ 
            color:#163a63; 
            background:#eaf2fb; 
            padding:5px 10px; 
            margin:0 0 6px; 
            border-left:4px solid #2f75b5; 
            font-size:12pt;
        }}
        h3 {{ 
            color:#365b7d; 
            font-size:10pt; 
            margin:8px 0 4px; 
            font-weight:600;
            border-bottom:1px solid #e2e8f0;
            padding-bottom:2px;
        }}
        
        /* ============================================
           修复: 顶层左右并排布局
           ============================================ */
        table.top-layout {{
            width: 100%;
            border-collapse: collapse;
            margin: 0 0 10px;
        }}
        td.top-left {{
            width: 58%;
            padding-right: 12px;
            vertical-align: top;
        }}
        td.top-right {{
            width: 42%;
            padding-left: 12px;
            vertical-align: top;
            border-left: 2px solid #e2e8f0;
        }}
        
        /* ============================================
           项目信息 - 3列2行
           ============================================ */
        table.info-grid {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 6px;
        }}
        td.info-cell {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 10px 16px;
            width: 33.33%;
            vertical-align: top;
        }}
        .info-label {{
            font-size: 8pt;
            color: #64748b;
            display: block;
            margin-bottom: 2px;
        }}
        .info-value {{
            font-size: 10pt;
            color: #1e293b;
            font-weight: 500;
            display: block;
        }}
        
        /* ============================================
           统计摘要 - 2列2行（右侧紧凑布局）
           ============================================ */
        table.summary-panel {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 12px;
        }}
        td.summary-card {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 16px 8px;
            text-align: center;
            vertical-align: middle;
            width: 100%;
        }}
        /* 不同卡片顶部彩色条 */
        td.sc-blue {{ border-top: 3px solid #3b82f6; }}
        td.sc-green {{ border-top: 3px solid #22c55e; }}
        td.sc-orange {{ border-top: 3px solid #f59e0b; }}
        td.sc-cyan {{ border-top: 3px solid #06b6d4; }}
        
        .summary-number {{
            font-size: 16pt;
            font-weight: 700;
            color: #163a63;
            line-height: 1.2;
        }}
        .summary-label {{
            font-size: 7.5pt;
            color: #64748b;
            margin-top: 6px;
        }}
        
        /* ============================================
           立面区域
           ============================================ */
        section {{ 
            page-break-inside: avoid; 
            margin-bottom: 8px;
            border: 1px solid #e2e8f0;
            padding: 8px;
        }}
        .facade-header {{
            margin-bottom: 6px;
        }}
        .facade-header h2 {{
            display: inline;
        }}
        .facade-dim {{
            font-size: 8pt;
            color: #94a3b8;
            margin-left: 10px;
        }}
        
        /* ============================================
           左右分栏
           ============================================ */
        table.data-columns {{
            width: 100%;
            border-collapse: collapse;
            margin: 6px 0 10px;
        }}
        td.column-left {{
            width: 50%;
            padding-right: 8px;
            border-right: 1px solid #e2e8f0;
            vertical-align: top;
        }}
        td.column-right {{
            width: 50%;
            padding-left: 8px;
            vertical-align: top;
        }}
        
        /* ============================================
           表格样式
           ============================================ */
        table.overview-table, table.detail-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 8.5pt;
        }}
        .overview-table td {{
            padding: 3px 5px;
            border-bottom: 1px solid #f1f5f9;
        }}
        .ov-label {{
            color: #64748b;
            width: 40%;
            font-size: 8pt;
        }}
        .ov-value {{
            color: #1e293b;
            font-weight: 500;
            text-align: right;
        }}
        .detail-table td {{
            padding: 3px 5px;
            border-bottom: 1px solid #f1f5f9;
        }}
        .dt-label {{
            color: #64748b;
            width: 55%;
            font-size: 8pt;
        }}
        .dt-value {{
            color: #1e293b;
            font-weight: 500;
            text-align: right;
        }}
        .alert-value {{
            color: #dc2626 !important;
            font-weight: 700;
        }}
        
        /* ============================================
           图片画廊
           ============================================ */
        table.image-gallery {{
            width: 100%;
            table-layout: fixed;
            border-collapse: separate;
            border-spacing: 10px;
            margin: 8px 0;
        }}
        td.image-cell {{
            width: 50%;
            max-width: 50%;
            overflow: hidden;
            vertical-align: top;
        }}
        .image-card {{
            background: #fff;
            border: 1px solid #e2e8f0;
            padding: 6px;
        }}
        .image-title {{
            font-size: 8pt;
            color: #365b7d;
            font-weight: 600;
            margin-bottom: 4px;
            text-align: center;
        }}
        .image-frame {{
            background: #f5f7fa;
            text-align: center;
            padding: 2px;
            width: 80%;
            height: 50mm;
            overflow: hidden;
            page-break-inside: avoid;
        }}
        .image-frame img {{
            display: block;
            width: 100%;
            max-width: 100%;
            height: 48mm;
        }}
        .image-legend {{
            margin-top: 2px;
            padding-top: 2px;
            border-top: 1px solid #f1f5f9;
        }}
        .legend-bar {{
            height: 6px;
            background: linear-gradient(to right, #1677c8 0%, #16b8c4 25%, #f2d12e 50%, #f28c28 75%, #c91f2b 100%);
            margin-bottom: 3px;
        }}
        .legend-labels {{
            font-size: 6.5pt;
            color: #94a3b8;
        }}
        .legend-labels span {{
            display: inline-block;
            width: 49%;
        }}
        .legend-labels span:last-child {{
            text-align: right;
        }}
        
        .muted {{ 
            color: #94a3b8; 
            font-size: 8.5pt;
            text-align: center;
            padding: 10px;
        }}
        .report-meta {{
            font-size: 8pt;
            color: #94a3b8;
            margin-bottom: 8px;
            text-align: right;
        }}
        </style></head><body>
        <h1>建筑外立面质量检测报告</h1>
        <div class='report-meta'>报告生成时间：{datetime.now():%Y-%m-%d %H:%M}</div>
        
        {top_section_html}
        
        <h2>建筑立面检测结果</h2>
        {''.join(sections) or '<p class=\"muted\">暂无立面检测结果</p>'}
        </body></html>"""

    @staticmethod
    def _calc_avg_pass_rate(snapshot):
        facades = snapshot.get("facades", [])
        rates = []
        for f in facades:
            q = f.get("quality") or {}
            o = q.get("overall") or {}
            for key in ("flatness_pass_rate", "verticality_pass_rate"):
                val = o.get(key)
                if val is not None:
                    try:
                        rates.append(float(val))
                    except (TypeError, ValueError):
                        pass
        if not rates:
            return "--"
        avg = sum(rates) / len(rates)
        if avg <= 1:
            avg *= 100
        return f"{avg:.1f}%"

    @staticmethod
    def _calc_total_windows(snapshot):
        facades = snapshot.get("facades", [])
        total = 0
        for f in facades:
            q = f.get("quality") or {}
            o = q.get("overall") or {}
            val = o.get("quality_valid_window_count")
            if val is not None:
                try:
                    total += int(val)
                except (TypeError, ValueError):
                    pass
        return str(total) if total > 0 else "--"

    @staticmethod
    def _calc_total_points(snapshot):
        facades = snapshot.get("facades", [])
        total = 0
        for f in facades:
            q = f.get("quality") or {}
            o = q.get("overall") or {}
            val = o.get("point_count")
            if val is not None:
                try:
                    total += int(val)
                except (TypeError, ValueError):
                    pass
        return str(total) if total > 0 else "--"

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