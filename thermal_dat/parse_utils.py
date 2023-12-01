# coding = utf-8
# @Time : 2023/7/25 13:13
# @Author : moyear
# @File : parse_utils.y
# @Software : PyCharm
import struct

import numpy as np

height = 256
width = 192


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


def yuv_2_rgb_2(yuv_data):
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

