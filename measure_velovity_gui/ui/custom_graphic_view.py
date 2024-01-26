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
    # 用于显示在界面上的点
    clicked_points = []
    # 底层保存的点
    selected_points = []

    multi_select = False
    click_enable = True

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.Antialiasing)

    def mousePressEvent(self, event):
        if not self.click_enable:
            return

        if event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.pos())
            self.add_red_dot(pos.x(), pos.y())

    def add_red_dot(self, x, y):
        if not self.multi_select:
            # 单选模式
            self.clear_selected_points()

        # print("点击x：", x, "y: ", y)

        red_dot = QGraphicsEllipseItem(x - 5, y - 5, 10, 10)
        red_dot.setBrush(QColor("red"))

        scene = self.scene()
        scene.addItem(red_dot)

        self.selected_points.append((x, y))
        self.clicked_points.append(red_dot)

    def get_selected_point(self):
        if self.selected_points is None or len(self.selected_points) == 0:
            return None
        # 返回最后一个元素
        return self.selected_points[len(self.selected_points) - 1]

    def get_selected_points(self):
        return self.selected_points

    def clear_selected_points(self):
        self.selected_points = []

        # 移除场景里面绘制的点
        scene = self.scene()
        for clicked_pt in self.clicked_points:
            if clicked_pt is not None and clicked_pt in scene.items():
                scene.removeItem(clicked_pt)

        self.clicked_points = []
