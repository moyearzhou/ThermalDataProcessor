# coding = utf-8
# @Time : 2023/12/1 11:14
# @Author : moyear
# @File : raw_phtoto_play.y
# @Software : PyCharm
import io
import mmap
import os
import re
import time

import cv2
import numpy as np
from PIL import Image

from matplotlib import pyplot as plt

from thermal_data.parse_utils import read_env_temp, read_min_temp, read_max_temp, read_average_temp, \
    parse_real_temp, yuv_2_rgb_2, yuv422_to_rgb

height = 256
width = 192

FILE_STREAM = ""


def show_img(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_img_rgb(title, img):
    # 将rgb图像转换为bgr图像，才能用open cv正常显示
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # 显示图像
    show_img(title, img_bgr)


def display_images(images):
    res = np.hstack(images)
    cv2.imshow("title", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_image_in_gallery(img):
    # 转换为PIL图像对象并显示图像
    image = Image.fromarray(img.astype(np.uint8))
    image.show()

    # return ave_temp


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


def stretch_colors(data):
    # 数值范围
    min_val = np.min(data)
    max_val = np.max(data)

    # 归一化数据
    normalized_data = (data - min_val) / (max_val - min_val)

    # 使用colormap进行颜色映射
    # colormap = plt.get_cmap('jet')
    # mathmatplot自带的inferno颜色映射最接近海康威视官方的方案，但是海康官方的整体颜色更亮更清透
    colormap = plt.get_cmap('inferno')
    colored_data = colormap(normalized_data)

    return colored_data


def generate_thermal_image(file_stream_data):
    print("==============读取各点的温度数据=================")
    # 读取各点的温度数据, 从4640到102944之间是全屏温度数据，一共98304（19f2*256*2）
    list_real_temps = []

    for i in range(4640, 102944, 2):
        # 读取温度数据
        file_stream_data.seek(i)
        byte_array = file_stream_data.read(2)
        # 解析温度数据
        real_temp = parse_real_temp(byte_array)
        list_real_temps.append(real_temp)

    # 生成一维数组数据
    # 将列表转换为指定大小的NumPy数组
    data = np.reshape(list_real_temps, (256, 192))

    # 对数组进行颜色拉伸
    colored_data = stretch_colors(data)
    print(colored_data)

    # 显示伪彩色影像
    plt.imshow(colored_data)
    plt.axis('off')
    plt.show()


# def read_yuv_bytes(file_stream_data):
#     byte_stream = io.BytesIO(file_stream_data.read())
#     byte_data = byte_stream.read()
#
#     print("yuv数据长度：{0}".format(len(byte_data) - 98304))
#     file_stream_data.seek(len(byte_data) - 98304)
#     yuv_bytes = file_stream_data.read(98304)
#     rgb = yuv_2_rgb_2(yuv_bytes)
#     return


def open_stream_file():
    with open('res/stream.dat', 'rb') as file_stream_data:
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
        generate_thermal_image(file_stream_data)

        # res = np.hstack((rgb, img_test))
        # show_image_in_gallery(res)

        # 将rgb图像转为bgr图像让open cv显示
        rgb_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # img_test = cv2.cvtColor(img_test, cv2.COLOR_RGB2BGR)
        # display_images((rgb_img, img_test))

        cv2.imshow("Image", rgb_img)
        # 设置鼠标回调函数
        cv2.setMouseCallback('Image', mouse_callback)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def read_bytes_from_position(file_path, position, num_bytes):
    try:
        with open(file_path, 'rb') as file:
            with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                mmapped_file.seek(position)
                bytes_data = mmapped_file.read(num_bytes)
                return bytes_data
    except FileNotFoundError:
        print(f"文件 '{file_path}' 不存在")
        return None


def convert_yuv_to_rgb(yuv_bytes, width, height):
    # 将YUV422字节数据转换为numpy数组
    yuv_data = np.frombuffer(yuv_bytes, dtype=np.uint8)

    # 调整数据形状为(height, width*2)
    yuv_data = yuv_data.reshape(height, width * 2)

    # 分离Y、U和V分量
    yuv_data = yuv_data[:, :width], yuv_data[:, width:width * 2]

    # 将YUV分量转换为RGB分量
    yuv_data = [yuv_data[0], np.repeat(yuv_data[1], 2, axis=1), np.repeat(yuv_data[2], 2, axis=1)]
    yuv_data = np.stack(yuv_data, axis=-1)

    # 使用OpenCV将YUV转换为RGB
    rgb_data = cv2.cvtColor(yuv_data, cv2.COLOR_YUV2RGB_YUYV)

    return rgb_data


def export_video(video_path):
    fps = 25  # 帧率
    frame_width = width  # 视频宽度
    frame_height = height  # 视频高度

    output_path = 'output/output.mp4'

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 选择视频编码器（这里选择MP4V）
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    for file_name in os.listdir(video_path):
        if not re.fullmatch(r"\d{8}", file_name):
            continue

        file_path = video_path + "/" + file_name

        time_0 = time.time()

        file_size = os.path.getsize(file_path)
        yuv_bytes = read_bytes_from_position(file_path, file_size - 98304, 98304)

        rgb = yuv422_to_rgb(yuv_bytes, width, height)
        rgb = rgb.astype(np.uint8)

        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        output_video.write(img)  # 将图片写入视频

        print("正在写入第{0}帧, 耗时：{1} ms".format(file_name, time.time() - time_0))

    output_video.release()
    print("转换图像序列成视频完成，保存位置为：{0}".format(output_path))


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

        file_size = os.path.getsize(file_path)
        yuv_bytes = read_bytes_from_position(file_path, file_size - 98304, 98304)

        rgb = yuv422_to_rgb(yuv_bytes, width, height)
        rgb = rgb.astype(np.uint8)

        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        img = cv2.resize(img, (width * 2, height * 2))
        cv2.imshow("Image", img)

        print("正在解析：{0}, 耗时：{1} ms".format(file_name, time.time() - time_0))

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
