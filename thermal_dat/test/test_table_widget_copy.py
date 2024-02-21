import sys
from PyQt5.QtWidgets import QApplication, QTableWidget, QTableWidgetItem, QMainWindow
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence, QClipboard

class MyTableWidget(QTableWidget):
    def __init__(self, parent=None):
        super(MyTableWidget, self).__init__(parent)
        self.initUI()

    def initUI(self):
        # 设置选择模式为多选和选择整行
        # self.setSelectionBehavior(QTableWidget.SelectRows)
        # self.setSelectionMode(QTableWidget.MultiSelection)
        print("")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_C and event.modifiers() == Qt.ControlModifier:
            self.copySelection()
        else:
            super().keyPressEvent(event)

    def copySelection(self):
        selection = self.selectedRanges()
        if not selection:
            return

        copied_text = ''
        for range in selection:
            for row in range.topRow(), range.bottomRow() + 1:
                rowItems = []
                for col in range.leftColumn(), range.rightColumn() + 1:
                    item = self.item(row, col)
                    rowItems.append(item.text() if item else '')
                copied_text += '\t'.join(rowItems) + '\n'

        QApplication.clipboard().setText(copied_text)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.tableWidget = MyTableWidget()
        self.setCentralWidget(self.tableWidget)
        self.initUI()

    def initUI(self):
        self.tableWidget.setRowCount(5)  # 示例行数
        self.tableWidget.setColumnCount(3)  # 示例列数
        for i in range(5):  # 填充数据
            for j in range(3):
                self.tableWidget.setItem(i, j, QTableWidgetItem(f"Item {i+1},{j+1}"))

        self.setGeometry(300, 300, 600, 400)
        self.setWindowTitle('Ctrl+C to Copy')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())
