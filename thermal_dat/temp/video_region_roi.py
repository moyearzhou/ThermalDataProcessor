# coding = utf-8
# @Time : 2023/12/6 21:21
# @Author : moyear
# @File : video_region_tiqu.y
# @Software : PyCharm

import cv2
import numpy as np


def find_roi(image_path):
    image = get_croped_image(image_path)

    # 预处理图像
    # 在这里，您可以根据需要进行灰度化、二值化、边缘检测等预处理步骤

    # 检测圆形
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=5, minDist=100, param1=50, param2=30, minRadius=10,
                               maxRadius=20)

    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)

        for (x, y, r) in circles:
            print("圆的位置")

            # 计算质心位置
            moments = cv2.moments(image)
            centroid_x = int(moments["m10"] / moments["m00"])
            centroid_y = int(moments["m01"] / moments["m00"])

            # 绘制圆形和质心位置
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
            cv2.circle(image, (centroid_x, centroid_y), 5, (0, 0, 255), -1)

    else:
        print("没有找到圆形")

    # 显示结果
    cv2.imshow('Image with Centroids', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def get_croped_image(image_path):
    # 读取图像为黑白
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path)
    # # 图像旋转90度
    # image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # 弹出对话框选择ROI区域
    rect = cv2.selectROI('Image', image, fromCenter=False, showCrosshair=True)
    # 提取矩形框的坐标
    x, y, w, h = rect

    # 裁剪图像
    image = image[y:y + h, x:x + w]
    return image

# 定义全局变量
selected_points = []
selected_points = []


image_path = r"../res/pic_counters_2.jpg"
image = cv2.imread(image_path)


def mouse_callback(event, x, y, flags, param):
    global selected_points, selected_points
    if event == cv2.EVENT_LBUTTONDOWN:
        print("点击坐标：", x, y)

        if len(points) < 4:
            points.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('Image', image)
        if len(points) == 4:
            selected_points = np.float32(points)
            cv2.destroyWindow('Image')
        # 提取ROI图像
    if len(selected_points) == 4:
        width = int(max(np.linalg.norm(selected_points[0] - selected_points[1]),
                        np.linalg.norm(selected_points[2] - selected_points[3])))
        height = int(max(np.linalg.norm(selected_points[0] - selected_points[3]),
                         np.linalg.norm(selected_points[1] - selected_points[2])))

        target_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

        matrix = cv2.getPerspectiveTransform(selected_points, target_points)
        roi_image = cv2.warpPerspective(image, matrix, (width, height))

        cv2.imshow('ROI Image', roi_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



def manual_find(image_path):
    image = cv2.imread(image_path)

    cv2.imshow("Image", image)
    cv2.setMouseCallback('Image', mouse_callback)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # find_roi(image_path)
    manual_find(image_path)

