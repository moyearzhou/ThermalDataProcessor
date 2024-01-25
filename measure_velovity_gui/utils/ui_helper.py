# coding = utf-8
# @Time : 2024/1/23 10:38
# @Author : moyear
# @File : ui_hlper.y
# @Software : PyCharm
import datetime

import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView


def frame_to_time_progress(frame_index, fps):
    seconds = frame_index / fps
    # return str(datetime.timedelta(seconds=int(seconds)))

    time_delta = datetime.timedelta(seconds=int(seconds))

    hours = time_delta.seconds // 3600
    if hours == 0:
        return str(time_delta)[2:]  # Skip the "0:" part
    else:
        return str(time_delta)


def image_scale_to_graphic_view(image, graphicView: QGraphicsView):
    '''
    缩放图片到自适应view大小
    :param image:
    :param graphicView:
    :return:
    '''
    view_width = graphicView.width()
    view_height = graphicView.height()

    # print("view宽度:", self.imgOri.width(), "view高度:", self.imgOri.height())

    # Resize the image
    image = cv2.resize(image, (view_width, view_height))

    qimage = None
    # Convert to QImage
    height, width, channel = image.shape
    bytesPerLine = channel * width
    qimage = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)

    # Create QPixmap from QImage
    pixmap = QPixmap.fromImage(qimage)

    # Create QGraphicsScene and add QPixmap
    scene = QGraphicsScene()
    scene.addPixmap(pixmap)

    # Set QGraphicsScene in QGraphicsView
    graphicView.setScene(scene)


def extract_hsv_numbers(text):
    numbers = text.split(',')
    result = []

    for number in numbers:
        number = number.strip()  # 去除空格
        try:
            number = int(number)
            if 0 <= number <= 255:
                result.append(number)
        except ValueError:
            pass
    print(result)
    return result


def convert_to_number(text):
    try:
        number = int(text)
        return number
    except ValueError:
        return -1