# coding = utf-8
# @Time : 2024/1/1 22:18
# @Author : moyear
# @File : video_perspective_transform.y
# @Software : PyCharm
import cv2
import numpy as np

fps = 25

video_out_path = "output_video.mp4"

video_path = r"E:\Moyear\文档\冲刷实验\测试数据\20240105_3_第三段冲刷俯视_Trim.mp4"
cap = cv2.VideoCapture(video_path)

# todo 计算选择后的长宽
dst_rect = (200, 800)  # 变换目标大小, 是一个元组，第0个位置是宽长度，第1个位置是高长度

# 获取第一帧并显示
ret, frame = cap.read()
if not ret:
    print("无法读取视频帧")
    exit(1)

# 创建窗口并设置鼠标回调函数
window_name = 'Select ROI'
cv2.namedWindow(window_name)
selected_points = []  # 存储选定的角点


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow(window_name, frame)


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    # 左上角点
    rect[0] = pts[np.argmin(s)]
    # 右下角点
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


# 显示第一帧并等待用户点击标靶位置
cv2.imshow(window_name, frame)
cv2.setMouseCallback(window_name, mouse_callback)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(selected_points) != 4:
    print("需要选择4个角点")
    exit(1)

print(selected_points)

# 定义透视变换的源点和目标点
src_points = np.float32(selected_points)

# 对选择的点进行重排序，位置顺序：左上、右上、右下、左下
src_points = order_points(src_points)

# 输出结果的长宽
output_width = dst_rect[0]
output_height = dst_rect[1]
dst_points = np.float32([[0, 0], [output_width, 0], [output_width, output_height], [0, output_height]])

# 计算透视变换矩阵
perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# 视频输出设置
output_video = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (output_width, output_height))

# 逐帧处理视频4321    ·
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 应用透视变换
    warped_frame = cv2.warpPerspective(frame, perspective_matrix, (output_width, output_height))

    img_gray = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)

    # # 计算灰度直方图
    # hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    # hist = hist.flatten()

    # # 绘制概率分布图
    # plt.plot(hist, color='black')
    # plt.xlabel('Pixel Value')
    # plt.ylabel('Probability')
    # plt.title('Grayscale Histogram')
    # plt.xlim([0, 256])
    # plt.ylim([0, max(hist)])
    # plt.show()

    # # 应用阈值分割
    # threshold_value = 55  # 阈值
    # max_value = 255  # 最大像素值
    # _, thresholded_image = cv2.threshold(img_gray, threshold_value, max_value, cv2.THRESH_TOZERO)

    # print(warped_frame)
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