# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'app_gui.py'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
from pyqt5_plugins.examplebutton import QtWidgets

from measure_velovity_gui.ui.measure_velocity_0 import Ui_MainWindow
from measure_velovity_gui.velocity_measurer import VelocityMeasure

video_measurer = VelocityMeasure()


if __name__ == "__main__":
    import sys
    my_app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(my_app.exec_())
