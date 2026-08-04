from __future__ import annotations

from services.dal.project_repo import ProjectRepo


class ProjectService:
    """
    项目管理相关service
    """

    @staticmethod
    def create_project(name: str, org_unit: str | None = None,
                       address: str | None = None, remarks: str | None = None) -> dict:
        """Create a project via repository and return basic info dict."""
        return ProjectRepo.create_project(name=name, org_unit=org_unit, address=address, remarks=remarks)
