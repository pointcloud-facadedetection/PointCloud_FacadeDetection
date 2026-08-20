from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QTextEdit,
    QVBoxLayout,
    QLabel,
    QMessageBox,
)


class ProjectCreateDialog(QDialog):
    """
    简单的“创建项目”表单对话框：
    - 必填：项目名称
    - 选填：单位、地址、备注
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("创建项目")
        self.setModal(True)
        # 给真实工程中的长名称、单位和地址留出足够输入空间。
        self.setMinimumSize(560, 360)
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        title = QLabel("请输入项目信息")
        title.setStyleSheet("font-size:16px; font-weight:600; color:#333;")
        lay.addWidget(title)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)

        self.edt_name = QLineEdit()
        self.edt_name.setPlaceholderText("必填")
        form.addRow("项目名称：", self.edt_name)

        self.edt_org = QLineEdit()
        self.edt_org.setPlaceholderText("选填")
        form.addRow("单位：", self.edt_org)

        self.edt_address = QLineEdit()
        self.edt_address.setPlaceholderText("选填")
        form.addRow("地址：", self.edt_address)

        self.edt_remarks = QTextEdit()
        self.edt_remarks.setPlaceholderText("选填")
        self.edt_remarks.setFixedHeight(80)
        form.addRow("备注：", self.edt_remarks)

        lay.addLayout(form)

        self.btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self.btns.accepted.connect(self._on_accept)
        self.btns.rejected.connect(self.reject)
        lay.addWidget(self.btns)

        self.resize(640, 420)
        self.edt_name.setFocus()

    def _on_accept(self):
        if not self.edt_name.text().strip():
            QMessageBox.warning(self, "提示", "项目名称为必填项！")
            self.edt_name.setFocus()
            return
        self.accept()

    def values(self) -> dict:
        return {
            "name": self.edt_name.text().strip(),
            "org_unit": self.edt_org.text().strip() or None,
            "address": self.edt_address.text().strip() or None,
            "remarks": self.edt_remarks.toPlainText().strip() or None,
        }
