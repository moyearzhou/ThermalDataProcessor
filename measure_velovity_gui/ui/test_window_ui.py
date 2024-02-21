import sys

from PyQt5 import QtCore, QtWidgets

from measure_velovity_gui.ui.measure_velocity import Ui_MainWindow

# 设置异常处理钩子
# sys.excepthook = handle_exception

# 高分辨率适配
# QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)

my_app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)
# 绑定监听器
# ui.bindListener(MainWindow)

# 初始化界面
ui.init()

MainWindow.show()
sys.exit(my_app.exec_())