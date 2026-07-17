"""主窗口占位按钮对应的 service。"""


class ButtonService:
    """接收 UI 按钮事件，并在控制台输出触发结果。"""

    def button1(self):
        print('button1被点击了', flush=True)

    def button2(self):
        print('button2被点击了', flush=True)

    def button3(self):
        print('button3被点击了', flush=True)

    def button4(self):
        print('button4被点击了', flush=True)
