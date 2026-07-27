"""主窗口 Header 按钮对应的临时 service 接口。"""


class ButtonService:
    """接收 UI 按钮事件，当前仅验证接口触发链。"""

    def __init__(self):
        self.selected_file_paths = []
        self.extracted_file_paths = []

    @staticmethod
    def _notify(action_name):
        print(f'{action_name} triggered', flush=True)

    # 算法工程师可在以下 btn_* 方法体中接入对应的业务/算法接口。
    def btn_upload(self, file_paths):
        self.selected_file_paths = list(file_paths)
        self._notify('btn_upload')
        self.extract_files(self.selected_file_paths)

    def extract_files(self, file_paths):
        self.extracted_file_paths = list(file_paths)
        self._notify('extract_files')

    def btn_reset_view(self):
        self._notify('btn_reset_view')

    def btn_change_color(self):
        self._notify('btn_change_color')

    def btn_denoise(self):
        self._notify('btn_denoise')

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
