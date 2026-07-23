"""Silent service endpoint for prototype button clicks."""


class ButtonService:
    """Record UI actions internally without writing noise to the terminal."""

    def __init__(self):
        self.last_action = None
        self.action_history = []

    def trigger(self, action_name: str, _label: str = ''):
        self.last_action = action_name
        self.action_history.append(action_name)

    # Keep the original public entry points for the team's existing callers.
    def upload_file(self):
        self.trigger('upload_file', '上传文件')

    def point_cloud_denoise(self):
        self.trigger('point_cloud_denoise', '点云去噪')

    def facade_detection(self):
        self.trigger('facade_detection', '立面检测')
