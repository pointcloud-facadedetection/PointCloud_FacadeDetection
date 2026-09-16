"""C 组：结构回归扫描（防重构回潮）。"""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

DIALOG_CLASSES = ('QMessageBox', 'QColorDialog', 'QFileDialog', 'QInputDialog')
PRIVATE_SERVICE_FIELDS = ('_index_service', '_last_facade_results',
                          'from services.dal')


def _python_files(dirname):
    return sorted((PACKAGE_ROOT / dirname).rglob('*.py'))


def test_ui_layer_avoids_private_service_fields():
    """ui/ 下不得再引用 service 私有字段或直连 DAL。"""
    offenders = []
    for path in _python_files('ui'):
        text = path.read_text(encoding='utf-8')
        for token in PRIVATE_SERVICE_FIELDS:
            if token in text:
                offenders.append(f'{path}: {token}')
    assert offenders == []


def test_services_layer_has_no_dialog_classes():
    """services/ 下不得 import/使用任何 QtWidgets 对话框类。"""
    offenders = []
    for path in _python_files('services'):
        text = path.read_text(encoding='utf-8')
        for token in DIALOG_CLASSES:
            if token in text:
                offenders.append(f'{path}: {token}')
    assert offenders == []


def test_main_window_mro_has_only_page_mixins(qapp):
    """MainWindow 继承列表只剩页面/外壳 mixin，controller 已全部对象化。"""
    from ui.main_window import MainWindow
    names = [c.__name__ for c in MainWindow.__mro__]
    mixins = [n for n in names if n.endswith('Mixin')]
    assert mixins == ['OverviewPageMixin', 'OperationPageMixin',
                      'InspectionReviewPageMixin', 'ReportPageMixin',
                      'ScaffoldPageMixin']


def test_main_window_exposes_controllers(qapp):
    """三个 controller 类可从各自模块导入且为 QObject 子类。"""
    from PySide6.QtCore import QObject
    from ui.controllers.registration import RegistrationController
    from ui.controllers.facade_quality import FacadeQualityController
    from ui.controllers.project_lifecycle import ProjectLifecycleController
    for cls in (RegistrationController, FacadeQualityController,
                ProjectLifecycleController):
        assert issubclass(cls, QObject)
