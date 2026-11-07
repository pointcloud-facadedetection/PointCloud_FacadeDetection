from __future__ import annotations

import json

from PySide6.QtWidgets import QLabel

from ui.dialogs.project_create_dialog import ProjectCreateDialog


class InspectionParametersDialog(ProjectCreateDialog):
    def __init__(self, project, facade_no=None, parent=None):
        super().__init__(parent=parent, project=project)
        self.setWindowTitle(
            f'立面 {facade_no} · 楼宇参数' if facade_no is not None
            else '项目默认 · 楼宇参数')
        for index, name in enumerate(('基础信息', '楼宇信息', '报告信息', '检测参数', '文件导入')):
            self.tabs.setTabVisible(index, name in ('楼宇信息', '检测参数'))
        self.tabs.setCurrentIndex(1)
        title = self.findChild(QLabel, 'formDialogTitle')
        if title is not None:
            title.setText('楼宇与质量检测参数')
        subtitle = self.findChild(QLabel, 'formDialogSubtitle')
        if subtitle is not None:
            subtitle.setText('保存后更新项目默认参数；未单独设置的立面自动沿用最近一次设置。')
        self._on_tab_changed(self.tabs.currentIndex())

    def inspection_values(self) -> dict:
        return json.loads(self.values()['inspection_params_json'])
