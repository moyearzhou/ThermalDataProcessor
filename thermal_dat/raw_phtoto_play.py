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

FILE_STREAM = ""


# def show_img(title, img):
#     cv2.imshow(title, img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# def show_img_rgb(title, img):
#     # 将rgb图像转换为bgr图像，才能用open cv正常显示
#     img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     # 显示图像
#     show_img(title, img_bgr)


# def display_images(images):
#     res = np.hstack(images)
#     cv2.imshow("title", res)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# def show_image_in_gallery(img):
#     # 转换为PIL图像对象并显示图像
#     image = Image.fromarray(img.astype(np.uint8))
#     image.show()


# 定义鼠标事件回调函数
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 读取温度数据
        pos = 4640 + (192 * y + x) * 2
        # pos = x * y * 2
        FILE_STREAM.seek(pos)
        byte_array = FILE_STREAM.read(2)
        # 解析温度数据
        temp = parse_real_temp(byte_array)
        print("点击坐标：", x, y, "温度：", temp)


def open_stream_file():
    with open('res/stream.dat', 'rb') as file_stream_data:
        global FILE_STREAM
        FILE_STREAM = file_stream_data
        byte_stream = io.BytesIO(file_stream_data.read())
        byte_data = byte_stream.read()

        print("==============读取全局温度数据概况=================")
        # env 温度
        read_env_temp(file_stream_data)
        # min 温度
        read_min_temp(file_stream_data)
        # max 温度
        read_max_temp(file_stream_data)
        # avg 温度
        read_average_temp(file_stream_data)

        # yuv_bytes = read_yuv_bytes(file_stream_data)

        # 读取实时温度数据
        file_stream_data.seek(4640)
        yuv_bytes = file_stream_data.read(98304)
        img_test = yuv_2_rgb_2(yuv_bytes)

        # 读取yuv温度图像数据
        print("yuv数据长度：{0}".format(len(byte_data) - 98304))
        file_stream_data.seek(len(byte_data) - 98304)
        yuv_bytes = file_stream_data.read(98304)
        rgb = yuv_2_rgb_2(yuv_bytes)

        # 生成新的温度影像图
        img_by_temperature = generate_thermal_image(file_stream_data)

        # res = np.hstack((rgb, img_test))
        # show_image_in_gallery(res)

        # 将rgb图像转为bgr图像让opencv显示
        img_by_yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # img_test = cv2.cvtColor(img_test, cv2.COLOR_RGB2BGR)
        # display_images((rgb_img, img_test))

        # cv2.imshow("Image YUV", img_by_yuv)
        cv2.imshow("Image Temp", img_by_temperature)
        # 设置鼠标回调函数
        cv2.setMouseCallback('Image Temp', mouse_callback)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


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

        # 如何稳定25帧率播放
        fps = 25
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

    # 初始化播放状态标志
    is_playing = True

    # 视频播放帧率
    fps = 25

    is_rotate = True

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
        img = cv2.resize(img, (width * 4, height * 4))

        if is_rotate:
            # 因为拍摄的时候不是竖屏，所以需要旋转
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        cv2.imshow("Image", img)

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






