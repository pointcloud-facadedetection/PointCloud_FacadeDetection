"""controller 测试共用的 fake service 与事件泵。"""
import time


def pump_until(qapp, predicate, timeout=5.0):
    """驱动 Qt 事件循环直到条件满足（worker 信号需经事件循环投递）。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class Row:
    """站点行记录（PointCloudStationService.list_stations 的最小替身）。"""

    def __init__(self, i, selected=True):
        self.id = i
        self.is_selected = selected


class SignalRecorder:
    """按发射顺序记录 controller 信号事件。"""

    def __init__(self):
        self.events = []

    def kinds(self):
        return [e[0] for e in self.events]

    def clear(self):
        self.events.clear()
