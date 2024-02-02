# coding = utf-8
# @Time : 2024/1/27 21:27
# @Author : moyear
# @File : test_window_ui.y
# @Software : PyCharm
import sys

from PyQt5 import QtCore, QtWidgets

from measure_velovity_gui.ui.measure_velocity import Ui_MainWindow

QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)

my_app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)

# 初始化界面
# ui.init_view(MainWindow)

MainWindow.show()
sys.exit(my_app.exec_())