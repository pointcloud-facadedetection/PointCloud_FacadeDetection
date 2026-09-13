from __future__ import annotations

from ui.dialogs.task_progress_dialog import TaskProgressDialog


class LoadingDialog(TaskProgressDialog):
    """点云加载中的模态等待窗。

    后台解析期间以应用级模态阻挡用户操作，但事件循环保持运行
    （worker 进度/完成信号照常投递）。无关闭按钮：加载只能由完成或
    失败回调结束，避免用户误以为点 X 可以取消。

    实现已收敛到 :class:`TaskProgressDialog`，本类只负责默认标题与
    文案，对外接口（``_bar`` / ``_label`` / ``update_progress`` /
    ``reject``）与历史版本保持逐字兼容。
    """

    DEFAULT_TITLE = '点云加载'
    DEFAULT_MESSAGE = '正在加载点云，请稍候...'
    FINISH_MESSAGE = '点云加载完成'

    def __init__(self, parent=None, *, cancellable: bool = False,
                 auto_close: bool = True, interval_ms: int = 5000,
                 title: str | None = None):
        super().__init__(
            parent,
            cancellable=cancellable,
            auto_close=auto_close,
            interval_ms=interval_ms,
            title=title or self.DEFAULT_TITLE,
        )
