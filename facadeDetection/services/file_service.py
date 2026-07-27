"""文件选择后的前端业务入口。

这里仅负责整理 UI 传入的文件路径。实际解析、转换和算法处理由后续
文件/算法接口接入，避免把业务实现写进主窗口。
"""

from pathlib import Path


class FileService:
    """处理上传文件列表并为后续文件接口保留清晰边界。"""

    def __init__(self):
        self.selected_file_paths = []
        self.extracted_file_paths = []

    def upload_files(self, file_paths):
        normalized_paths = [
            str(Path(file_path).expanduser().resolve())
            for file_path in file_paths
            if file_path
        ]
        self.selected_file_paths = normalized_paths
        print('upload_files triggered', flush=True)
        return self.extract_files(normalized_paths)

    def extract_files(self, file_paths):
        self.extracted_file_paths = list(file_paths)
        print('extract_files triggered', flush=True)
        return list(self.extracted_file_paths)
