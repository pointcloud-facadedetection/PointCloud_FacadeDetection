"""主窗口 Header 按钮对应的临时 service 接口。"""

from typing import Iterable, Optional

from services.file_service import FileService
from services.pointcloud_service import PointCloudService
from services.project_service import ProjectService


class ButtonService:
    """接收 UI 按钮事件，接入上传与视口渲染逻辑。"""

    def __init__(self, viewport=None, db=None, render_service=None):
        # 由 MainWindow 注入的依赖
        self.viewport = viewport
        self.db = db  # session factory (callable)
        self.render_service = render_service
        self.active_project_uuid: Optional[str] = None
        self.on_project_created = None  # Optional[Callable[[dict], None]] set by UI

        # 懒加载 FileService（在首次使用前或依赖齐备时创建）
        self.file_service: Optional[FileService] = None
        if self.viewport is not None and self.db is not None and self.render_service is not None:
            self.file_service = FileService(self.viewport, self.db, self.render_service)

        # 点云业务 service
        self.pointcloud_service = PointCloudService(self.viewport, self.render_service)
        # 统一由 FileService 提供 FLS 导入能力

        # 记录上传/解压路径
        self.selected_file_paths = []
        self.extracted_file_paths = []

    @staticmethod
    def _notify(action_name):
        print(f'{action_name} triggered', flush=True)

    # ---- Context setters ----
    def set_active_project_uuid(self, project_uuid: Optional[str]):
        """由 UI 注入当前激活项目 UUID。为 None 时表示未选择项目。"""
        self.active_project_uuid = project_uuid

    # 算法工程师可在以下 btn_* 方法体中接入对应的业务/算法接口。
    def btn_upload(self, file_paths: Iterable[str]):
        """
        处理文件上传按钮的点击事件：
        - 通过 FileService.upload_files 统一处理（自动识别点云/图片，
          若存在激活项目则通过 FileRepo 记录到项目并触发渲染）
        参数:
            file_paths: 用户选择的文件路径列表
        """
        # 将传入的文件路径列表转换为列表形式并保存
        self.selected_file_paths = list(file_paths)
        self._notify('btn_upload')
        for path in self.selected_file_paths:
            try:
                if self.file_service is None:
                    self.file_service = FileService(self.viewport, self.db, self.render_service)
                # 统一调用 upload_files，由 Service 内部根据 project_uuid 决定持久化策略
                self.file_service.upload_files(
                    project_uuid=self.active_project_uuid,
                    file_path=path,
                    copy_into_project=False,
                )
            except Exception as e:
                # 容错：继续处理其他文件，并输出错误
                print(f'处理文件失败: {path} -> {e}', flush=True)

        # 若有后续“解压/预处理”等步骤，可在此衔接
        self.extract_files(self.selected_file_paths)

        # 上传完成后可选重置视图，便于用户查看
        try:
            if hasattr(self.viewport, 'auto_range'):
                self.viewport.auto_range()
        except Exception:
            pass

    # 新增：导入 FLS 目录（由 UI 调用）
    def import_fls_directory(self, dir_path: str):
        self._notify('import_fls_directory')
        try:
            if self.file_service is None:
                self.file_service = FileService(self.viewport, self.db, self.render_service)
            res = self.file_service.import_fls_directory(dir_path, self.active_project_uuid)
        except Exception as e:
            print(f'FLS 导入流程失败: {e}', flush=True)
            return

        if not res.get('success'):
            print(f"FLS 转换未成功: {res.get('message')}", flush=True)
            return

        # 上传完成后可选重置视图，便于用户查看
        try:
            if hasattr(self.viewport, 'auto_range'):
                self.viewport.auto_range()
        except Exception:
            pass

    def extract_files(self, file_paths):
        self.extracted_file_paths = list(file_paths)
        self._notify('extract_files')

    def btn_reset_view(self):
        self._notify('btn_reset_view')
        try:
            if hasattr(self.viewport, 'reset_view'):
                self.viewport.reset_view()
            elif hasattr(self.viewport, 'auto_range'):
                self.viewport.auto_range()
        except Exception:
            pass

    def btn_change_color(self):
        self._notify('btn_change_color')

    def btn_denoise(self, _checked=False, method: str = 'radius', voxel_size: float = 0.05, **kwargs):
        """
        点云去噪：
        - 选择当前活动点云（若无则取最后一个）
        - 使用 algorithms.preprocess.denoise 进行半径/统计去噪
        - 将结果更新回视口以即时显示
        参数:
            method: 'radius' 或 'statistical'
            voxel_size: 体素/尺度参数，用于推导默认半径等
            **kwargs: 可选参数（radius/min_neighbors/nb_neighbors/std_ratio）
        """
        self._notify('btn_denoise')
        try:
            if not hasattr(self, "pointcloud_service") or self.pointcloud_service is None:
                self.pointcloud_service = PointCloudService(self.viewport, self.render_service)
            self.pointcloud_service.denoise(method=method, voxel_size=voxel_size, **kwargs)
        except Exception as e:
            print(f'去噪失败: {e}', flush=True)

    def btn_registration(self):
        self._notify('btn_registration')

    def btn_facade_detection(self):
        self._notify('btn_facade_detection')

    def btn_quality_inspection(self):
        self._notify('btn_quality_inspection')

    def btn_box_segmentation(self):
        self._notify('btn_box_segmentation')

    def btn_calculate_detail(self):
        self._notify('btn_calculate_detail')

    def btn_align_2d_3d(self):
        # 设计文档暂称“2D_align_3D”，这里使用合法的 Python 方法名。
        self._notify('btn_align_2d_3d')

    # ---------------- Project actions ----------------
    def btn_create_project(self):
        """触发“创建项目”对话框，收集数据，并调用 ProjectService 完成创建。"""
        from PySide6.QtWidgets import QMessageBox, QDialog
        try:
            from ui.dialogs.project_create_dialog import ProjectCreateDialog
        except Exception as e:
            print(f"加载创建项目对话框失败: {e}", flush=True)
            return

        # 打开表单
        parent = getattr(self.viewport, 'get_widget', None)
        parent_widget = parent() if callable(parent) else None
        dlg = ProjectCreateDialog(parent_widget)
        result_code = dlg.exec()
        try:
            accepted_code = int(QDialog.DialogCode.Accepted)
        except Exception:
            accepted_code = getattr(QDialog, 'Accepted', 1)
        if result_code == accepted_code:
            payload = dlg.values()
            try:
                info = ProjectService.create_project(
                    name=payload.get('name', ''),
                    org_unit=payload.get('org_unit'),
                    address=payload.get('address'),
                    remarks=payload.get('remarks'),
                )
            except Exception as e:
                QMessageBox.critical(parent_widget, '创建项目失败', f'错误：{e}')
                return

            # 更新本服务上下文
            self.active_project_uuid = info.get('project_uuid')
            # 通知 UI 更新
            try:
                if callable(self.on_project_created):
                    self.on_project_created(info)
            except Exception:
                pass
