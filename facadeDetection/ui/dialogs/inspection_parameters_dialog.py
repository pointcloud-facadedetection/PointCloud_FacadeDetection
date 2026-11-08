from __future__ import annotations

import json

from PySide6.QtWidgets import QLabel

from ui.dialogs.project_create_dialog import ProjectCreateDialog


class InspectionParametersDialog(ProjectCreateDialog):
    def __init__(self, project, facade_no=None, parent=None):
        super().__init__(parent=parent, project=project)
        self.setWindowTitle(
            f'立面 {facade_no} · 检测参数' if facade_no is not None
            else '项目默认 · 楼宇参数')
        visible_names = {'检测参数'}
        if facade_no is None:
            visible_names.add('楼宇信息')
        for index, name in enumerate(('基础信息', '楼宇信息', '报告信息', '检测参数', '文件导入')):
            self.tabs.setTabVisible(index, name in visible_names)
        self.tabs.setCurrentIndex(3 if facade_no is not None else 1)
        title = self.findChild(QLabel, 'formDialogTitle')
        if title is not None:
            title.setText('当前立面检测参数' if facade_no is not None
                          else '楼宇与质量检测参数')
        subtitle = self.findChild(QLabel, 'formDialogSubtitle')
        if subtitle is not None:
            subtitle.setText('楼宇信息由项目统一维护；立面只微调检测参数。')
        self._on_tab_changed(self.tabs.currentIndex())

    def inspection_values(self) -> dict:
        return json.loads(self.values()['inspection_params_json'])
