# coding = utf-8
# @Time : 2023/12/1 11:14
# @Author : moyear
# @File : raw_phtoto_play.y
# @Software : PyCharm
import datetime
import io
import os
import re
import time
import zipfile

import cv2
import numpy as np
from PIL import Image

from parse_utils import read_env_temp, read_min_temp, read_max_temp, read_average_temp, \
    parse_real_temp, yuv_2_rgb_2, format_time, extra_raw_video, generate_thermal_image, read_rgb_from, \
    read_rgb_from_bytes, generate_thermal_image_from_bytes
from thermal_dat.utils import get_shoot_time, add_time

height = 256
width = 192

CUR_FRAME_BYTES = ""

is_rotate_clockwise = True

scale = 4

# 视频播放帧率
fps = 25

using_yuv = False

draw_time = True

# 当前视频开始拍摄的时间
str_time_start_to_shoot = ""


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


def play_video_series(video_path):
    '''
    这是播放ThermalCam拍摄的自定义格式的视频
    :param video_path:
    :return:
    '''
    global CUR_FRAME_BYTES, using_yuv
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
        CUR_FRAME_BYTES = file_path

        time_0 = time.time()

        # rgb = read_rgb_from(file_path, width, height)
        # img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        img = []
        if using_yuv:
            rgb = read_rgb_from(file_path, width, height)
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            with open(file_path, 'rb') as file:
                img = generate_thermal_image(file)

        img = cv2.resize(img, (width * scale, height * scale))

        if is_rotate_clockwise:
            # 因为拍摄的时候不是竖屏，所以需要旋转
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        cv2.imshow("Image", img)
        # 设置鼠标回调函数
        cv2.setMouseCallback('Image', mouse_callback)

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


def draw_time_in_image(img, str_progress, str_shoot_time):
    # 设置文字参数
    text = "progress: {0}".format(str_progress)
    position = (50, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 指定字体为宋体
    # font = "SimSun"
    font_scale = 0.5
    color = (0, 0, 255)  # BGR color format (红色)
    thickness = 2

    # 绘制视频进度的文字
    cv2.putText(img, text, position, font, font_scale, color, thickness)

    text = "time: {0}".format(str_shoot_time)
    position = (50, 80)
    color = (0, 255, 0)  # BGR color format (绿色)
    # 绘制视频进度的文字
    cv2.putText(img, text, position, font, font_scale, color, thickness)


def play_raw_series(video_path):
    '''
       这是播放ThermalCam拍摄的自定义格式的zip视频
       :param video_path:
       :return:
       '''
    global CUR_FRAME_BYTES, using_yuv

    # 初始化播放状态标志
    is_playing = True

    zip_file_path = video_path

    # 获取当前图像序列开始拍摄的时间
    time_start_shoot = get_shoot_time(video_path)

    # 打开 ZIP 文件
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # 获取 ZIP 文件中的所有文件列表
        file_list = zip_ref.namelist()

        # 遍历文件列表
        for file_info in file_list:
            # 仅处理 raw 文件夹下的文件
            if file_info.startswith('raw/'):
                file_name = file_info.replace("raw/", "")

                if not re.fullmatch(r"\d{8}", file_name):
                    continue

                cur_frame = int(file_name)

                file_bytes = zip_ref.read(file_info)

                CUR_FRAME_BYTES = file_bytes

                if not is_playing:
                    key = cv2.waitKey()
                    # 按下空格键切换播放状态
                    if key == ord(' '):
                        is_playing = not is_playing

                # CUR_FRAME_FILE_PATH = target_file_path

                time_0 = time.time()

                img = []
                if using_yuv:
                    rgb = read_rgb_from_bytes(file_bytes, width, height)
                    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                else:
                    img = generate_thermal_image_from_bytes(file_bytes)
                # todo 对热红外进行预处理，提取水流特征

                # 重新缩放显示
                img = cv2.resize(img, (width * scale, height * scale))

                if is_rotate_clockwise:
                    # 因为拍摄的时候不是竖屏，所以需要旋转
                    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # 视频的播放进度时间
                str_progress = format_time(int(cur_frame / fps))
                # 当前帧的拍摄时间
                str_shoot_time = add_time(time_start_shoot, str_progress)

                # 在图上绘制时间标注
                if draw_time:
                    draw_time_in_image(img, str_progress, str_shoot_time)

                cv2.imshow("Image", img)
                # 设置鼠标回调函数
                cv2.setMouseCallback('Image', mouse_callback)

                time_consuming = time.time() - time_0
                print("{0} 正在解析：{1}, 解析耗时：{2} s".format(str_progress, file_info, round(time_consuming, 2)))

                # 图像播放间隔时间
                delay = int((float(1 / int(fps)) * 1000))

                key = cv2.waitKey(delay)
                # 按下空格键切换播放状态
                if key == ord(' '):
                    is_playing = not is_playing
                # 按下 'q' 键退出循环
                elif key == ord('q'):
                    break

    # todo 手动选择4个点运用透射变换





