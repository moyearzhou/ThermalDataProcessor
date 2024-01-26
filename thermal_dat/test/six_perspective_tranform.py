# coding = utf-8
# @Time : 2024/1/25 11:52
# @Author : moyear
# @File : four_perspective_tranform.y
# @Software : PyCharm
import cv2
import numpy as np

import cv2
import numpy as np

image_path = r"E:\Users\Moyear\Desktop\test1.png"


# 全局变量
selected_points = []  # 存储选择的点
img = None   # 存储图像

image_for_show = None


# 鼠标回调函数，用于捕获点
def click_event(event, x, y, flags, param):
    global selected_points, image_for_show
    if event == cv2.EVENT_LBUTTONDOWN and len(selected_points) < 6:
        selected_points.append((x, y))
        cv2.circle(image_for_show, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow(window_name, image_for_show)


def order_points(points):
    # 按 y 坐标排序
    points.sort(key=lambda point: point[1])
    # 取出 y 坐标最小的两个点（顶部），并按 x 坐标排序
    top_points = sorted(points[:2], key=lambda point: point[0])
    # 取出 y 坐标最大的两个点（底部），并按 x 坐标排序
    bottom_points = sorted(points[-2:], key=lambda point: point[0])

    # 剩余的点为中部，按 x 坐标排序
    middle_points = sorted(points[2:-2], key=lambda point: point[0])
    # 将三组点合并成一个列表
    sorted_points = top_points + middle_points + bottom_points
    # print(sorted_points)
    return sorted_points


# Function to perform perspective correction
def perform_perspective_correction(frame, selected_points):
    # 对选择的6个点进行排序，顺序为：左上点、右上点、左中点、右中点、左下点、右下点
    selected_points = order_points(selected_points)

    width = 200
    height = 800

    half_height = int(height / 2)

    src_points_upper = np.array([selected_points[0], selected_points[1], selected_points[2], selected_points[3]], dtype='float32')
    src_points_lower = np.array([selected_points[2], selected_points[3], selected_points[4], selected_points[5]], dtype='float32')

    # Define destination points for upper and lower halves
    dst_points = np.array([[0, 0], [width, 0], [0, half_height], [width, half_height]], dtype='float32')

    # Compute the perspective transform matrices for upper and lower halves
    matrix_upper = cv2.getPerspectiveTransform(src_points_upper, dst_points)
    matrix_lower = cv2.getPerspectiveTransform(src_points_lower, dst_points)
    # Apply perspective transformation for upper and lower halves
    transformed_upper = cv2.warpPerspective(frame, matrix_upper, (width, half_height))
    transformed_lower = cv2.warpPerspective(frame, matrix_lower, (width, half_height))

    # Concatenate the two halves
    transformed_image = np.concatenate((transformed_upper, transformed_lower), axis=0)
    return transformed_image


video_out_path = "output_video.mp4"

video_path = r"E:\Moyear\文档\冲刷实验\测试数据\20240105_3_第三段冲刷俯视_Trim.mp4"
cap = cv2.VideoCapture(video_path)

# 坡面视频的长度和宽度，也是变换目标大小, 是一个元组，第0个位置是宽长度，第1个位置是高长度
dst_rect = (200, 800)
# 输出结果的长宽
output_width = dst_rect[0]
output_height = dst_rect[1]

fps = 25


# 获取第一帧并显示
ret, frame = cap.read()
if not ret:
    print("无法读取视频帧")
    exit(1)

# 创建窗口并设置鼠标回调函数
window_name = 'Select ROI'
cv2.namedWindow(window_name)

# 显示第一帧并等待用户点击标靶位置
image_for_show = frame

cv2.imshow(window_name, image_for_show)
cv2.setMouseCallback(window_name, click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(selected_points) != 6:
    print("需要选择6个角点")
    exit(1)

print(selected_points)

# 视频输出设置
output_video = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (output_width, output_height))

# 逐帧处理视频4321    ·
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 应用透视变换
    warped_frame = perform_perspective_correction(frame, selected_points)
    warped_frame = cv2.rotate(warped_frame, cv2.ROTATE_180)

    # img_gray = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)


    # 显示透视变换后的帧
    cv2.imshow('Warped Frame', warped_frame)
    # cv2.imshow('Warped Frame', thresholded_image)

    # 图像播放间隔时间
    delay = int((float(1 / int(fps)) * 1000))

    key = cv2.waitKey(delay)

    # 将透视变换后的帧写入输出视频文件
    output_video.write(warped_frame)

cap.release()
output_video.release()