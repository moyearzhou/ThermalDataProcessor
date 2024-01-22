# coding = utf-8
# @Time : 2024/1/19 17:08
# @Author : moyear
# @File : blue_contours_extract.y
# @Software : PyCharm
import cv2
import numpy as np

from measure_velovity_gui.velocity_measurer import VelocityMeasure

# file_path = r'../res/blue_contour.png'
video_path = r"E:\Moyear\文档\冲刷实验\测试数据\轨迹.mp4"

fps = 25

frame_width = -1
frame_height = -1

# 槽子的长度为160cm
real_height = 1600
# 槽子的宽度40cm
real_width = 400

# 像素到毫米的比例，根据实际情况调整
pixel_to_mm = -1

video_measurer = VelocityMeasure()


def get_center_point(contour):
    M = cv2.moments(contour)

    center = (-1, -1)
    if M['m00'] != 0:
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
    return center


def video_extract():
    video_measurer.init_with_video(video_path)

    frame_width = video_measurer.frame_width
    frame_height = video_measurer.frame_height

    # frame_width = int(cap.get(3))
    # frame_height = int(cap.get(4))
    print('视频宽：{0}，高：{1}'.format(frame_width, frame_height))
    pixel_to_mm = real_height / frame_height

    print("在纵向上，每个像素为{0}mm".format(pixel_to_mm))

    # 示踪剂前缘点的轮廓
    pre_contour = None

    # 开始滴水的帧索引
    start_frame_index = 0

    frame_index = 0
    # 逐帧处理视频4321    ·
    while True:
        ret, frame = video_measurer.read()
        if not ret:
            break

        frame_index += 1

        cv2.imshow('ori', frame)
        # mask = video_measurer.get_blue_mask()
        # 通过形态学操作，过滤出蓝色流体的区域
        contours = video_measurer.get_blue_contours()

        # todo 对这些轮廓进行一定过滤
        # contours_centers = get_contour_centers(contours)

        contour_image = video_measurer.get_image_with_contours()

        # todo 根据轮廓计算流速
        # 计算流速
        if len(contours) > 0:
            # 初始化最大y坐标和对应的轮廓索引
            max_y = -1
            max_contour_index = -1

            # 遍历轮廓
            for i, contour in enumerate(contours):
                # 计算当前轮廓的边界框
                x, y, w, h = cv2.boundingRect(contour)
                # 获取当前轮廓的y坐标
                contour_y = y + h

                # 检查当前轮廓的y坐标是否大于最大y坐标
                if contour_y > max_y:
                    max_y = contour_y
                    max_contour_index = i

            max_contour = None
            # 检查是否找到了具有最大y坐标的轮廓
            if max_contour_index != -1:
                # 获取最大y坐标对应的轮廓
                max_contour = contours[max_contour_index]

                # 计算最大y坐标对应轮廓的边界框
                x, y, w, h = cv2.boundingRect(max_contour)

                # 获取最大y坐标对应轮廓的x和y坐标值
                contour_x = x
                contour_y = y + h

                # 设置文字参数
                text = "({0},{1})".format(contour_x, contour_y)
                position = (contour_x + 10, contour_y)
                font = cv2.FONT_HERSHEY_SIMPLEX
                # 指定字体为宋体
                # font = "SimSun"
                font_scale = 0.4
                color = (0, 255, 0)  # BGR color format (红色)
                thickness = 1

                # 绘制视频进度的文字
                cv2.putText(contour_image, text, position, font, font_scale, color, thickness)

                # 打印结果
                # print("最大y坐标的轮廓的x坐标：", contour_x)
                print("最大y坐标的轮廓的y坐标：", contour_y)
            else:
                print("未找到轮廓")

            if pre_contour is None:
                pre_contour = max_contour
                start_frame_index = frame_index
            else:
                pre_center = get_center_point(pre_contour)
                cur_center = get_center_point(max_contour)

                x0 = pre_center[0]
                x1 = cur_center[0]

                y0 = pre_center[1]
                y1 = cur_center[1]

                if y1 > y0:
                    displacement = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
                    real_distance = displacement * pixel_to_mm  # 计算真实移动的距离，单位mm
                    # print(real_distance)

                    if real_distance >= 100:
                        time_interval = (frame_index - start_frame_index) / fps  # 时间间隔，单位为ms
                        # print(time_interval)

                        flow_speed = real_distance / time_interval  # 计算流速，单位：mm/ms即是m/s
                        # print('Flow Speed:', flow_speed, "m/s")

                        # todo 如果在它前面则更新
                        # if y1 > y0:
                        pre_contour = max_contour
                        start_frame_index = frame_index

        # print(contours)

        # if largest_contour is not None:
        #     cv2.drawContours(contour_image, [largest_contour], -1, (255, 0, 0), 2)

        # img = frame
        cv2.imshow("img", contour_image)
        cv2.waitKey(int(1000 / fps))


video_extract()
