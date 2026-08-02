"""报告预览/导出页面的业务入口。"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PdfDocument:
    """WebView2 加载 PDF 所需的数据。"""

    name: str
    uri: str


class ReportExportService:
    """校验并整理报告页面需要加载的 PDF。"""

    def prepare_pdf(self, pdf_path):
        path = Path(pdf_path).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() != '.pdf':
            raise ValueError('请选择有效的 PDF 文件。')
        return PdfDocument(name=path.name, uri=path.as_uri())
