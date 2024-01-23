# coding = utf-8
# @Time : 2024/1/23 20:18
# @Author : moyear
# @File : custom_graphic_view.y
# @Software : PyCharm

import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsEllipseItem


class CustomGraphicsView(QGraphicsView):

    dot_radius = 10

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.Antialiasing)
        self.clicked_point = None
        self.selected_point = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.pos())
            self.add_red_dot(pos.x(), pos.y())

    def add_red_dot(self, x, y):
        self.selected_point = (x, y)
        # print("点击x：", x, "y: ", y)
        self.clear_selected_point()

        red_dot = QGraphicsEllipseItem(x - 5, y - 5, 10, 10)
        red_dot.setBrush(QColor("red"))

        scene = self.scene()
        scene.addItem(red_dot)
        self.clicked_point = red_dot

    def get_selected_point(self):
        return self.selected_point

    def clear_selected_point(self):
        scene = self.scene()
        if self.clicked_point is not None and self.clicked_point in scene.items():
            scene.removeItem(self.clicked_point)