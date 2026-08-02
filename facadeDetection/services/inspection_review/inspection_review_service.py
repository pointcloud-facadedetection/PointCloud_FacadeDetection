"""检测复核页面的业务入口。"""


class InspectionReviewService:
    """保存待复核结果，为后续复核算法和交互保留页面接口。"""

    def __init__(self):
        self._current_result = None

    def set_current_result(self, result):
        self._current_result = result

    def get_current_result(self):
        return self._current_result
