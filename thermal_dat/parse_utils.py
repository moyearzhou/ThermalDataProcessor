# coding = utf-8
# @Time : 2023/7/25 13:13
# @Author : moyear
# @File : parse_utils.y
# @Software : PyCharm
import mmap
import os
import struct
import zipfile

import cv2
import numpy as np
from matplotlib import pyplot as plt


def byte_2_float(bytes):
    '''
    # 将字节数据转换为浮点数
    :param bytes:
    :return:
    '''
    return struct.unpack('f', bytes)[0]


def parse_real_temp(byte_array):
    # 温度数据解析 : 按照小端, 低字节在后, 高字节在前方式, (每两个字节/64) – 50℃ = 实际温度
    integer = int.from_bytes(byte_array, byteorder='little', signed=True)  # 小端字节顺序，有符号
    real_temp = integer / 64 - 50
    # print("位置：{0}, 温度：{1}".format(i - 4640, real_temp))
    # print(real_temp)
    return real_temp


def read_env_temp(file):
    '''
    读取env温度数据
    :param file:
    :return:
    '''
    env_temp_bytes = []

    file.seek(140)
    env_temp_bytes = file.read(4)
    env_temp = byte_2_float(env_temp_bytes)
    print("Env温度：{0}".format(env_temp))
    return env_temp


def read_min_temp(file):
    '''
    读取min温度数据
    :param file:
    :return:
    '''
    file.seek(144)
    min_temp_bytes = file.read(4)
    min_temp = byte_2_float(min_temp_bytes)
    print("最低温度：{0}".format(min_temp))
    return min_temp


def read_max_temp(file):
    '''
    读取max温度数据
    :param file:
    :return:
    '''
    file.seek(148)
    max_temp_bytes = file.read(4)
    max_temp = byte_2_float(max_temp_bytes)
    print("最高温度：{0}".format(max_temp))
    return max_temp


def read_average_temp(file):
    '''
    读取average温度数据
    :param file:
    :return:
    '''
    file.seek(152)  # 设置读取的起始位置（假设起始位置为 4）
    ave_temp_bytes = file.read(4)  # 读取 4 个字节的数据
    ave_temp = byte_2_float(ave_temp_bytes)
    print("平均温度：{0}".format(ave_temp))


def yuv_2_rgb_2(yuv_data, width, height):
    # 将YUY2字节数据转换为图像
    yuy2 = np.frombuffer(yuv_data, dtype=np.uint8)
    yuv = yuy2.reshape((height, width * 2))

    # 提取Y、U、V分量的数据
    y_data = yuv[:, ::2]
    u_data = yuv[:, 1::4]
    v_data = yuv[:, 3::4]

    # 执行YUY2到RGB的转换
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(height):
        for col in range(0, width, 2):
            y0 = y_data[row, col]
            y1 = y_data[row, col + 1]
            u = u_data[row, col // 2]
            v = v_data[row, col // 2]

            # 计算RGB值
            r0 = y0 + 1.402 * (v - 128)
            g0 = y0 - 0.344136 * (u - 128) - 0.714136 * (v - 128)
            b0 = y0 + 1.772 * (u - 128)

            r1 = y1 + 1.402 * (v - 128)
            g1 = y1 - 0.344136 * (u - 128) - 0.714136 * (v - 128)
            b1 = y1 + 1.772 * (u - 128)

            # 将计算得到的RGB值存储到相应位置
            rgb[row, col] = [r0, g0, b0]
            rgb[row, col + 1] = [r1, g1, b1]
    return rgb


def yuv422_to_rgb(yuv_bytes, width, height):
    # 将YUV422字节数据转换为numpy数组
    yuv_array = np.frombuffer(yuv_bytes, dtype=np.uint8)
    # Reshape为(height, width*2)的数组，每两个元素代表一个YUV对
    yuv_image = yuv_array.reshape((height, width * 2))

    # 注意：需要使用int16的类型进行运算，否则计算结果不对！！！
    yuv_image = np.array(yuv_image, np.int16)

    # 提取Y、U、V分量
    y = yuv_image[:, ::2]
    u = yuv_image[:, 1::4]
    v = yuv_image[:, 3::4]

    # 扩展 U 和 V 分量到与 Y 分量的维度一致
    u = np.repeat(u, 2, axis=1)
    v = np.repeat(v, 2, axis=1)

    # 进行颜色转换：YUV to RGB
    r = np.clip((y + 1.402 * (v - 128)), 0, 255)
    g = np.clip((y - 0.34414 * (u - 128) - 0.71414 * (v - 128)), 0, 255)
    b = np.clip((y + 1.772 * (u - 128)), 0, 255)

    # 重新组合RGB分量
    rgb_data = np.stack((r, g, b), axis=-1)
    return rgb_data


def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    time_string = "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)
    return time_string



def extra_raw_video(video_path):
    '''
    extract all files and retun the path to extract
    :param video_path:
    :return:
    '''
    # 指定zip文件的路径和要提取文件的目标目录
    zip_file_path = video_path
    extract_dir = 'output/temp/'

    print("extract compress file {0}...".format(zip_file_path))
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    print("success to extract raw video to {0}".format(extract_dir))

    raw_pic_dir = extract_dir + "raw"
    return raw_pic_dir


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


def read_bytes_from(bytes, start, num_bytes):
    return bytes[start: num_bytes]


def read_yuv_bytes_from(file_path):
    file_size = os.path.getsize(file_path)
    yuv_bytes = read_bytes_from_position(file_path, file_size - 98304, 98304)
    return yuv_bytes


def read_rgb_from(file_path, width, height):
    yuv_bytes = read_yuv_bytes_from(file_path)
    return read_rgb_from_bytes(yuv_bytes, width, height)
    #
    # rgb = yuv422_to_rgb(yuv_bytes, width, height)
    # rgb = rgb.astype(np.uint8)
    # return rgb


def read_rgb_from_bytes(file_bytes, width, height):
    # print(len(file_bytes))

    yuv_bytes = file_bytes[-98304:]

    rgb = yuv422_to_rgb(yuv_bytes, width, height)
    rgb = rgb.astype(np.uint8)
    return rgb


def stretch_colors(data, min_temp=9.0, max_temp=30):
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


def generate_thermal_image(stream_file):
    '''
    输入原始的raw图像文件的字节bytes，查找其中的实况温度数据，并且把它根据最大最小温度进行归一化颜色拉伸，
    最终返回处理后的温度数据
    :param stream_file:
    :return:
    '''
    # print("==============读取各点的温度数据=================")
    # 读取各点的温度数据, 从4640到102944之间是全屏温度数据，一共98304（19f2*256*2）
    list_real_temps = []

    for i in range(4640, 102944, 2):
        # 读取温度数据
        stream_file.seek(i)
        byte_array = stream_file.read(2)
        # 解析温度数据
        real_temp = parse_real_temp(byte_array)
        list_real_temps.append(real_temp)

    # 生成一维数组数据
    # 将列表转换为指定大小的NumPy数组
    data = np.reshape(list_real_temps, (256, 192))

    # 对数组进行颜色拉伸
    #
    colored_data = stretch_colors(data)
    # print(type(colored_data))

    # # 显示伪彩色影像
    # plt.imshow(colored_data)
    # plt.axis('off')
    # plt.show()

    # 归一化后需要*255才是正常rgb图像
    img = (colored_data * 255).astype(np.uint8)

    # plt与opencv的颜色模式不一样必须先转换一下
    bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return bgr_img



def generate_thermal_image_from_bytes(file_bytes):
    '''
    输入原始的raw图像文件的字节bytes，查找其中的实况温度数据，并且把它根据最大最小温度进行归一化颜色拉伸，
    最终返回处理后的温度数据
    :param stream_file:
    :return:
    '''
    # print("==============读取各点的温度数据=================")
    # 读取各点的温度数据, 从4640到102944之间是全屏温度数据，一共98304（19f2*256*2）
    list_real_temps = []

    for i in range(4640, 102944, 2):
        # 读取温度数据
        # file_bytes.seek(i)
        byte_array = file_bytes[i: i+2]
        # 解析温度数据
        real_temp = parse_real_temp(byte_array)
        list_real_temps.append(real_temp)

    # 生成一维数组数据
    # 将列表转换为指定大小的NumPy数组
    data = np.reshape(list_real_temps, (256, 192))

    # 对数组进行颜色拉伸
    #
    colored_data = stretch_colors(data)
    # print(type(colored_data))

    # # 显示伪彩色影像
    # plt.imshow(colored_data)
    # plt.axis('off')
    # plt.show()

    # 归一化后需要*255才是正常rgb图像
    img = (colored_data * 255).astype(np.uint8)

    # plt与opencv的颜色模式不一样必须先转换一下
    bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return bgr_img
