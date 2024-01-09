# coding = utf-8
# @Time : 2024/1/9 11:10
# @Author : moyear
# @File : raw_roi_wxtract.y
# @Software : PyCharm

import io

import cv2
import numpy as np
from matplotlib import pyplot as plt

from thermal_dat.parse_utils import parse_real_temp

p_list = []  # 左上，右上，左下，右下顺序点击

img_path = r"E:\Users\Moyear\Desktop\测试视频\00000000.00000000"

img2 = ""
img = ""

height = 256
width = 192

scale = 4

is_rotate_clockwise = True

CUR_FRAME_BYTES = ""


# todo 计算选择后的长宽
dst_rect = (200, 800)  # 变换目标大小, 是一个元组，第0个位置是宽长度，第1个位置是高长度


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


def auto_perspective_transform(image, pts):
    # todo 这个方法是根据四个角点的位置直接算出新的透射变化后的图形的长与宽，
    #  但是这种方法可能与实际坡面的长宽比不一致
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def four_point_transform(img, rect, dst_rect):
    # todo 输入一个图像，并指定四个角点的位置，然后再根据需要透射变换后的目标图像的长宽进行透射变换
    # 透射变换后的4个角点顺序：左上、右上、右下、左下
    pts2 = np.float32([[0, 0], [dst_rect[0], 0], [dst_rect[0], dst_rect[1]], [0, dst_rect[1]]])

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, pts2)
    dst = cv2.warpPerspective(img, M, dst_rect)
    return dst


# def capture_event(event, x, y, flags, params):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         # 绘制点击的位置
#         cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
#         cv2.imshow("original_img", img)
#         p_list.append([x, y])
#
#         if len(p_list) == 4:
#             # 对选择的4个角点进行排序，顺序：左上、右上、右下、左下
#             rect = order_points(np.float32(p_list))
#
#             img_dst = four_point_transform(img2, rect, dst_rect)  # 按照指定的长宽进行透射变换
#             # img_dst = auto_perspective_transform(img2, rect)  # 根据选择的角点位置，自动计算变换后的长宽，然后进行透射变换
#
#             print("after perspective transformation, image height: {0}, width: {1}"
#                   .format(img_dst.shape[0], img_dst.shape[1]))
#
#             # _, img_dst = cv2.threshold(img_dst[:,:,2], 127, 255, cv2.THRESH_BINARY)
#
#             cv2.imshow("result_img", img_dst)
#             cv2.imwrite('../output/transformed.jpg', img_dst)  # 输出图像到文件


# 定义鼠标事件回调函数
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:

        real_x = int(x / scale)
        real_y = int(y / scale)

        # 逆时针
        if is_rotate_clockwise:
            real_x = width - int(y / scale)
            real_y = int(x / scale)

        # 读取温度数据
        pos = 4640 + (192 * real_y + real_x) * 2

        # pos = x * y * 2
        byte_array = CUR_FRAME_BYTES[pos: pos+2]
        # 解析温度数据
        temp = parse_real_temp(byte_array)
        print("点击坐标：", x, y, "温度：", temp)


def stretch_colors(data, min_temp=15.0, max_temp=30):
    # 数值范围
    min_val = min_temp if (min_temp is not None) else np.min(data)
    max_val = max_temp if (max_temp is not None) else np.max(data)

    # 归一化数据
    normalized_data = (data - min_val) / (max_val - min_val)

    # 使用colormap进行颜色映射
    # colormap = plt.get_cmap('jet')
    # mathmatplot自带的inferno颜色映射最接近海康威视官方的方案，但是海康官方的整体颜色更亮更清透
    colormap = plt.get_cmap('inferno')
    colored_data = colormap(normalized_data)

    return colored_data


with open(img_path, 'rb') as file_stream_data:
    byte_stream = io.BytesIO(file_stream_data.read())
    byte_data = byte_stream.read()

    CUR_FRAME_BYTES = byte_data

    # 读取各点的温度数据, 从4640到102944之间是全屏温度数据，一共98304（19f2*256*2）
    list_real_temps = []

    for i in range(4640, 102944, 2):
        # 读取温度数据
        # file_bytes.seek(i)
        byte_array = byte_data[i: i + 2]
        # 解析温度数据
        real_temp = parse_real_temp(byte_array)
        list_real_temps.append(real_temp)

    # 生成一维数组数据
    # 将列表转换为指定大小的NumPy数组
    data = np.reshape(list_real_temps, (256, 192))

    # print(data)

    # 对数组进行颜色拉伸
    colored_data = stretch_colors(data)

    # # 显示伪彩色影像
    # plt.imshow(colored_data)
    # plt.axis('off')
    # plt.show()

    # 归一化后需要*255才是正常rgb图像
    img = (colored_data * 255).astype(np.uint8)

    # plt与opencv的颜色模式不一样必须先转换一下
    bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 重新缩放显示
    img = cv2.resize(img, (width * scale, height * scale))

    if is_rotate_clockwise:
        # 因为拍摄的时候不是竖屏，所以需要旋转
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # 将图像转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用阈值化或者其他方法来检测标靶
    # 这里我们使用阈值化，你可能需要根据你的图像进行调整
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 使用 findContours 找到标靶位置
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 定义圆度阈值和面积阈值
    circularity_threshold = 0.7
    area_threshold = 50  # 你需要根据实际情况设置这个值

    # 筛选出近似圆形的轮廓
    circular_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > area_threshold:
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity > circularity_threshold:
                circular_contours.append(cnt)

    # # 查看轮廓的面积和长度
    # for contour in contours:
    #     # 计算轮廓的面积
    #     area = cv2.contourArea(contour)
    #     # 计算轮廓的周长，第二个参数表示轮廓是否闭合
    #     perimeter = cv2.arcLength(contour, True)
    #     print("轮廓面积：{0}，周长：{1}".format(area, perimeter))

    # 按照质心的 y 坐标进行排序
    circular_contours.sort(key=lambda cnt: cv2.moments(cnt)['m01'] / cv2.moments(cnt)['m00'])

    # 将排好序的轮廓分为 3 行
    rows = [circular_contours[i:i + 4] for i in range(0, len(circular_contours), 4)]

    # 对每行的轮廓按照 x 坐标进行排序
    for i in range(len(rows)):
        rows[i] = sorted(rows[i], key=lambda cnt: cv2.moments(cnt)['m10'] / cv2.moments(cnt)['m00'])

    # 将排序好的轮廓放入 ndarray 中
    sorted_contours = np.array(rows)

    print(sorted_contours.shape)

    # 前两列为左边的轮廓，后两列为右边的轮廓
    left_contours = sorted_contours[:, :2]
    right_contours = sorted_contours[:, 2:]

    print(type(left_contours))

    # 打印左边和右边的轮廓数量
    print(f'Left contours: {left_contours.size}, Right contours: {right_contours.size}')

    # # 在 output 图像上绘制左边的轮廓，我们用白色表示轮廓，线宽设为 2
    # for cnt in left_contours:
    #     cv2.drawContours(img, [cnt], -1, (255, 255, 255), 2)

    # 绘制轮廓到原图像上，-1表示所有轮廓，(0,255,0)是颜色，2是线条粗细
    cv2.drawContours(img, left_contours.flatten(), -1, (0, 0, 255), 2)

    cv2.imshow("Image", img)
    # 设置鼠标回调函数
    cv2.setMouseCallback('Image', mouse_callback)
    cv2.waitKey(0)