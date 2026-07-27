"""项目操作页面的点云功能入口。"""


class PointCloudService:
    """将 UI 按钮事件转交给视口或后续算法接口。"""

    def __init__(self, viewport):
        self._viewport = viewport

    @staticmethod
    def _notify(action_name):
        print(f'{action_name} triggered', flush=True)

    def reset_view(self):
        self._notify('reset_view')
        self._viewport.reset_view()

    def change_color(self):
        self._notify('change_color')

    def denoise(self):
        self._notify('denoise')

    def registration(self):
        self._notify('registration')

    def facade_detection(self):
        self._notify('facade_detection')

    def quality_inspection(self):
        self._notify('quality_inspection')

    def box_segmentation(self):
        self._notify('box_segmentation')

    def calculate_detail(self):
        self._notify('calculate_detail')

    def align_2d_3d(self):
        self._notify('align_2d_3d')
