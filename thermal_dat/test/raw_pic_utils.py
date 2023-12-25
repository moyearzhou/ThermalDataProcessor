# coding = utf-8
# @Time : 2023/12/11 10:50
# @Author : moyear
# @File : raw_pic_utils.y
# @Software : PyCharm
import io

import cv2

from thermal_dat.parse_utils import yuv_2_rgb_2, generate_thermal_image, read_env_temp, read_min_temp, read_max_temp, \
    read_average_temp, parse_real_temp

CUR_FRAME_BYTES = "res/stream.dat"

scale = 1


# 定义鼠标事件回调函数
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        with open(CUR_FRAME_BYTES, 'rb') as file_stream_data:

            real_x = int(x / scale)
            real_y = int(y / scale)

            # if is_rotate:
            #     real_x = int(y / scale)
            #     real_y = int(x / scale)

            # 读取温度数据
            pos = 4640 + (192 * real_y + real_x) * 2
            # pos = x * y * 2
            file_stream_data.seek(pos)
            byte_array = file_stream_data.read(2)
            # 解析温度数据
            temp = parse_real_temp(byte_array)
            print("点击坐标：", x, y, "温度：", temp)


def open_stream_file():
    with open(CUR_FRAME_BYTES, 'rb') as file_stream_data:
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

        # 将rgb图像转为bgr图像让opencv显示
        img_by_yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # cv2.imshow("Image YUV", img_by_yuv)
        cv2.imshow("Image Temp", img_by_temperature)
        # 设置鼠标回调函数
        cv2.setMouseCallback('Image Temp', mouse_callback)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
