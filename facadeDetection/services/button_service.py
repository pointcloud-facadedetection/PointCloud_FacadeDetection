"""主窗口 Header 按钮对应的 service。"""


class ButtonService:
    """接收 UI 按钮事件，并在控制台输出触发结果。"""

    def upload_file(self):
        print('upload_file被点击了', flush=True)

    def point_cloud_denoise(self):
        print('point_cloud_denoise被点击了', flush=True)

    def facade_detection(self):
        print('facade_detection被点击了', flush=True)
