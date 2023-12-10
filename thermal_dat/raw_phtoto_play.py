# coding = utf-8
# @Time : 2023/12/1 11:14
# @Author : moyear
# @File : raw_phtoto_play.y
# @Software : PyCharm
import io
import os
import re
import time

import cv2
import numpy as np
from PIL import Image

from parse_utils import read_env_temp, read_min_temp, read_max_temp, read_average_temp, \
    parse_real_temp, yuv_2_rgb_2, format_time, extra_raw_video, generate_thermal_image, read_rgb_from

height = 256
width = 192

cur_file_path = ""

is_rotate_clockwise = True

scale = 4

# 视频播放帧率
fps = 25


# 定义鼠标事件回调函数
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        with open(cur_file_path, 'rb') as file_stream_data:

            real_x = int(x / scale)
            real_y = int(y / scale)

            # 逆时针
            if is_rotate_clockwise:
                real_x = width - int(y / scale)
                real_y = int(x / scale)

            # 读取温度数据
            pos = 4640 + (192 * real_y + real_x) * 2
            # pos = x * y * 2
            file_stream_data.seek(pos)
            byte_array = file_stream_data.read(2)
            # 解析温度数据
            temp = parse_real_temp(byte_array)
            print("点击坐标：", x, y, "温度：", temp)


def play_video_series(video_path):
    '''
    这是播放ThermalCam拍摄的自定义格式的视频
    :param video_path:
    :return:
    '''

    # 初始化播放状态标志
    is_playing = True

    for file_name in os.listdir(video_path):
        if not re.fullmatch(r"\d{8}", file_name):
            continue

        if not is_playing:
            key = cv2.waitKey()
            # 按下空格键切换播放状态
            if key == ord(' '):
                is_playing = not is_playing

        file_path = video_path + "/" + file_name

        time_0 = time.time()

        rgb = read_rgb_from(file_path, width, height)

        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        img = cv2.resize(img, (width * 2, height * 2))
        cv2.imshow("Image", img)

        time_consuming = time.time() - time_0

        print("正在解析：{0}, 耗时：{1} s".format(file_name, round(time_consuming, 2)))

        # 图像播放间隔时间
        delay = int(float(1 / int(fps) * 1000))

        key = cv2.waitKey(delay)
        # 按下空格键切换播放状态
        if key == ord(' '):
            is_playing = not is_playing
        # 按下 'q' 键退出循环
        elif key == ord('q'):
            break


def play_raw_series(video_path, using_yuv=False):
    '''
       这是播放ThermalCam拍摄的自定义格式的zip视频
       :param video_path:
       :return:
       '''
    global cur_file_path

    # 初始化播放状态标志
    is_playing = True

    # extract zip file of raw video, and get the path to extract
    raw_pic_dir = extra_raw_video(video_path)

    # todo 手动选择4个点运用透射变换

    for file_name in os.listdir(raw_pic_dir):
        if not re.fullmatch(r"\d{8}", file_name):
            continue

        if not is_playing:
            key = cv2.waitKey()
            # 按下空格键切换播放状态
            if key == ord(' '):
                is_playing = not is_playing

        file_path = raw_pic_dir + "/" + file_name

        cur_file_path = file_path

        time_0 = time.time()

        img = []

        if using_yuv:
            rgb = read_rgb_from(file_path, width, height)
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            with open(file_path, 'rb') as file:
                img = generate_thermal_image(file)

        # todo 对热红外进行预处理，提取水流特征

        # 重新缩放显示
        img = cv2.resize(img, (width * scale, height * scale))

        if is_rotate_clockwise:
            # 因为拍摄的时候不是竖屏，所以需要旋转
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        cv2.imshow("Image", img)
        # 设置鼠标回调函数
        cv2.setMouseCallback('Image', mouse_callback)

        cur_frame = int(file_name)
        str_progress = format_time(int(cur_frame / fps))

        time_consuming = time.time() - time_0
        print("{0} 正在解析：{1}, 解析耗时：{2} s".format(str_progress, file_name, round(time_consuming, 2)))

        # 图像播放间隔时间
        delay = int((float(1 / int(fps)) * 1000))

        key = cv2.waitKey(delay)
        # 按下空格键切换播放状态
        if key == ord(' '):
            is_playing = not is_playing
        # 按下 'q' 键退出循环
        elif key == ord('q'):
            break






