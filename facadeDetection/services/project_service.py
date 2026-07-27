"""项目概览页使用的前端项目 Service。

当前实现保存本次运行期间的项目摘要，接口形式与后续数据库 Service
保持解耦。数据库工程师接入持久化后，UI 只需替换本 Service 的实现。
"""

from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4


@dataclass
class ProjectSummary:
    """项目列表卡片需要的最小数据集合。"""

    project_id: str
    name: str
    directory_path: str
    file_paths: list[str] = field(default_factory=list)


class ProjectService:
    """提供项目列表、新建、导入和选择所需的前端接口。"""

    def __init__(self):
        self._projects = {}

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
        normalized_paths = [str(Path(path).expanduser().resolve()) for path in file_paths]
        if current_project is None:
            first_path = Path(normalized_paths[0])
            project = self._upsert_project(
                name=first_path.stem or '未命名项目',
                directory_path=str(first_path.parent),
            )
        else:
            project = current_project

        project.file_paths = list(dict.fromkeys(project.file_paths + normalized_paths))
        self._projects[project.project_id] = project
        return project

    def remove_project(self, project_id):
        return self._projects.pop(project_id, None)

    def _upsert_project(self, name, directory_path):
        normalized_name = (name or '').strip() or '未命名项目'
        normalized_directory = str(Path(directory_path).expanduser().resolve())

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
