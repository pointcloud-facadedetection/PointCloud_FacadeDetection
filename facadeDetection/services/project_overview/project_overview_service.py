"""项目概览页面的业务入口。

UI 只调用本页面 Service；后续文件解析和数据库持久化可以在这里接入，
无需修改主窗口中的按钮和项目卡片代码。
"""

from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4


@dataclass
class ProjectSummary:
    """项目列表卡片需要的最小数据。"""

    project_id: str
    name: str
    directory_path: str
    file_paths: list[str] = field(default_factory=list)


class ProjectOverviewService:
    """处理项目概览页的文件选择和项目列表业务。"""

    def __init__(self):
        self.selected_file_paths = []
        self.extracted_file_paths = []
        self._projects = {}

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

    def list_projects(self):
        return list(self._projects.values())

    def get_project(self, project_id):
        return self._projects.get(project_id)

    def create_project(self, name, directory_path):
        return self._upsert_project(name=name, directory_path=directory_path)

    def open_project(self, directory_path):
        path = Path(directory_path).expanduser().resolve()
        return self._upsert_project(
            name=path.name or '未命名项目',
            directory_path=str(path),
        )

    def register_upload(self, file_paths, current_project=None):
        normalized_paths = [
            str(Path(path).expanduser().resolve())
            for path in file_paths
        ]
        if not normalized_paths:
            raise ValueError('至少需要选择一个文件。')

        if current_project is None:
            first_path = Path(normalized_paths[0])
            project = self._upsert_project(
                name=first_path.stem or '未命名项目',
                directory_path=str(first_path.parent),
            )
        else:
            project = current_project

        project.file_paths = list(
            dict.fromkeys(project.file_paths + normalized_paths)
        )
        self._projects[project.project_id] = project
        return project

    def remove_project(self, project_id):
        return self._projects.pop(project_id, None)

    def rename_project(self, project_id, new_name):
        """修改项目显示名称，并返回修改后的项目。"""
        project = self.get_project(project_id)
        if project is None:
            raise ValueError('项目不存在或已被删除。')

        normalized_name = (new_name or '').strip()
        if not normalized_name:
            raise ValueError('项目名称不能为空。')

        # 这里只更新当前项目列表中的名称；不会重命名用户的本地目录。
        project.name = normalized_name
        return project

    def _upsert_project(self, name, directory_path):
        normalized_name = (name or '').strip() or '未命名项目'
        normalized_directory = str(
            Path(directory_path).expanduser().resolve()
        )

        for project in self._projects.values():
            if project.directory_path == normalized_directory:
                project.name = normalized_name
                return project

        project = ProjectSummary(
            project_id=str(uuid4()),
            name=normalized_name,
            directory_path=normalized_directory,
        )
        self._projects[project.project_id] = project
        return project
