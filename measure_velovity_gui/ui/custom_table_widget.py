import sys
from PyQt5.QtWidgets import QApplication, QTableWidget, QTableWidgetItem, QShortcut
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence

class MyTableWidget(QTableWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initUI()

    def initUI(self):
        # 启用多选和选择整行
        # self.setSelectionBehavior(QTableWidget.SelectRows)
        # self.setSelectionMode(QTableWidget.MultiSelection)

        # 示例数据填充
        for i in range(self.rowCount()):
            for j in range(self.columnCount()):
                self.setItem(i, j, QTableWidgetItem(f"Item {i},{j}"))

        # 设置快捷键Ctrl+C复制
        shortcut = QShortcut(QKeySequence.Copy, self)
        shortcut.activated.connect(self.copySelection)

    def copySelection(self):
        # 获取选中的范围
        selection = self.selectedRanges()

        if selection:
            copied_text = ''
            for range in selection:
                for row in range(range.topRow(), range.bottomRow() + 1):
                    rowItems = []
                    for col in range(range.leftColumn(), range.rightColumn() + 1):
                        item = self.item(row, col)
                        rowItems.append(item.text() if item else '')
                    copied_text += '\t'.join(rowItems) + '\n'

            # 复制到剪贴板
            QApplication.clipboard().setText(copied_text)
