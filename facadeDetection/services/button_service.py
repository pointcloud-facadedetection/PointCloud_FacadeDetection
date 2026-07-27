"""主窗口 Header 按钮对应的临时 service 接口。"""


class ButtonService:
    """接收 UI 按钮事件，当前仅验证接口触发链。"""

    def __init__(self):
        self.selected_file_paths = []
        self.extracted_file_paths = []

    @staticmethod
    def _notify(action_name):
        print(f'{action_name} triggered', flush=True)

    def upload_files(self, file_paths):
        self.selected_file_paths = list(file_paths)
        self._notify('upload_files')
        self.extract_files(self.selected_file_paths)

    def extract_files(self, file_paths):
        self.extracted_file_paths = list(file_paths)
        self._notify('extract_files')

    def reset(self):
        self._notify('reset')

    def change_colors(self):
        self._notify('change_colors')

    def denoise(self):
        self._notify('denoise')

    def registration(self):
        self._notify('registration')

    def facade_detection(self):
        self._notify('facade_detection')

    def compute_quality(self):
        self._notify('compute_quality')

    def segmentation(self):
        self._notify('segmentation')

    def compute_detail(self):
        self._notify('compute_detail')

    def align_2d_3d(self):
        # 设计文档暂称“2D_align_3D”，这里使用合法的 Python 方法名。
        self._notify('align_2d_3d')
