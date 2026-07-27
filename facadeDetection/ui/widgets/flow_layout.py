"""从左到右排列、空间不足时自动换行的 Qt 布局。"""

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout


class FlowLayout(QLayout):
    """用于 Header 按钮的轻量流式布局。"""

    def __init__(self, parent=None, margin=0, horizontal_spacing=8, vertical_spacing=8):
        super().__init__(parent)
        self._items = []
        self._horizontal_spacing = horizontal_spacing
        self._vertical_spacing = vertical_spacing
        self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._visible_items():
            size = size.expandedTo(item.minimumSize())

        left, top, right, bottom = self.getContentsMargins()
        return size + QSize(left + right, top + bottom)

    def _do_layout(self, rect, test_only):
        left, top, right, bottom = self.getContentsMargins()
        area = rect.adjusted(left, top, -right, -bottom)
        x = area.x()
        y = area.y()
        line_height = 0

        for item in self._visible_items():
            item_size = item.sizeHint()
            if line_height and x + item_size.width() > area.right() + 1:
                x = area.x()
                y += line_height + self._vertical_spacing
                line_height = 0

            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item_size))

            x += item_size.width() + self._horizontal_spacing
            line_height = max(line_height, item_size.height())

        return y + line_height - rect.y() + bottom

    def _visible_items(self):
        return [item for item in self._items if not item.isEmpty()]
