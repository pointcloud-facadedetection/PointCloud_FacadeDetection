"""Main application shell for the first-version UI prototype."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QStackedWidget, QVBoxLayout, QWidget

from services.button_service import ButtonService
from ui.dialogs.new_project_dialog import NewProjectDialog
from ui.pages.home_page import HomePage
from ui.pages.report_page import ReportPage
from ui.pages.workbench_page import WorkbenchPage
from ui.styles import APP_STYLESHEET


class MainWindow(QMainWindow):
    """Own the Home → Workbench → Report navigation and placeholder services."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('mainWindow')
        self.setWindowTitle('外立面激光检测工作台')
        self.setMinimumSize(1180, 760)
        self.resize(1500, 920)
        self.button_service = ButtonService()
        self.new_project_dialog = None
        self.current_project_id = None
        self._setup_ui()
        self._connect_pages()
        self.show_home()

    def _setup_ui(self):
        root = QWidget()
        root.setObjectName('appRoot')
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.page_stack = QStackedWidget()
        self.page_stack.setObjectName('mainPageStack')
        self.home_page = HomePage()
        self.workbench_page = WorkbenchPage()
        self.report_page = ReportPage()
        self.page_stack.addWidget(self.home_page)
        self.page_stack.addWidget(self.workbench_page)
        self.page_stack.addWidget(self.report_page)
        layout.addWidget(self.page_stack)
        self.setCentralWidget(root)
        self.setStyleSheet(APP_STYLESHEET)

    def _connect_pages(self):
        self.home_page.new_project_requested.connect(self.open_new_project)
        self.home_page.enter_project_requested.connect(self.open_project)
        self.home_page.report_requested.connect(self.open_project_report)
        self.home_page.action_requested.connect(self.button_service.trigger)

        self.workbench_page.back_requested.connect(self.show_home)
        self.workbench_page.report_requested.connect(self.show_report)
        self.workbench_page.action_requested.connect(self.button_service.trigger)

        self.report_page.back_requested.connect(self.show_workbench)
        self.report_page.action_requested.connect(self.button_service.trigger)

    @property
    def current_page_name(self) -> str:
        current = self.page_stack.currentWidget()
        if current is self.home_page:
            return 'home'
        if current is self.workbench_page:
            return 'workbench'
        return 'report'

    def show_home(self):
        self.page_stack.setCurrentWidget(self.home_page)

    def show_workbench(self):
        self.page_stack.setCurrentWidget(self.workbench_page)

    def show_report(self):
        self.page_stack.setCurrentWidget(self.report_page)

    def open_new_project(self):
        if self.new_project_dialog is not None and self.new_project_dialog.isVisible():
            self.new_project_dialog.raise_()
            self.new_project_dialog.activateWindow()
            return

        dialog = NewProjectDialog(self)
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        dialog.project_created.connect(self._create_project)
        dialog.close_button.clicked.connect(
            lambda: self.button_service.trigger('close_new_project', '关闭')
        )
        dialog.cancel_button.clicked.connect(
            lambda: self.button_service.trigger('cancel_new_project', '取消')
        )
        dialog.create_button.clicked.connect(
            lambda: self.button_service.trigger(
                'create_project_and_enter', '创建项目并进入工作台'
            )
        )
        dialog.finished.connect(self._clear_new_project_dialog)
        self.new_project_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _create_project(self, project_data: dict):
        card = self.home_page.add_project(project_data)
        self._load_project(card.project_id)
        self.show_workbench()

    def _load_project(self, project_id: str):
        project_data = self.home_page.get_project_data(project_id)
        self.current_project_id = project_id
        self.workbench_page.set_project_data(project_data)
        self.report_page.set_project_data(project_data)

    def open_project(self, project_id: str):
        self._load_project(project_id)
        self.show_workbench()

    def open_project_report(self, project_id: str):
        self._load_project(project_id)
        self.show_report()

    def _clear_new_project_dialog(self, _result: int):
        dialog = self.sender()
        if dialog is self.new_project_dialog:
            self.new_project_dialog = None

    def closeEvent(self, event):
        if self.new_project_dialog is not None:
            self.new_project_dialog.close()
        super().closeEvent(event)
