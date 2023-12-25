# coding = utf-8
# @Time : 2023/12/10 12:17
# @Author : moyear
# @File : raw_video_convert.y
# @Software : PyCharm
import os
import re
import time
import zipfile

import cv2

from parse_utils import extra_raw_video, generate_thermal_image, read_rgb_from, format_time, read_rgb_from_bytes, \
    generate_thermal_image_from_bytes
from thermal_dat.utils import count_files, add_time, get_shoot_time, count_frame_in_raws, get_name_without_extension

height = 256
width = 192

is_rotate_clockwise = True

# 允许视频缩放
enable_scale = True
# 视频缩放
scale = 2

using_yuv = True

# 在正式写入视频之前是否先预览第一张图片
preview_begin_convert = False

# 是否在画面上绘制时间
draw_time = True


def draw_time_in_image(img, str_progress, str_shoot_time):
    # 设置文字参数
    text = "progress: {0}".format(str_progress)
    position = (10, 10)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 指定字体为宋体
    # font = "SimSun"
    font_scale = 0.4
    color = (0, 0, 255)  # BGR color format (红色)
    thickness = 1

    # 绘制视频进度的文字
    cv2.putText(img, text, position, font, font_scale, color, thickness)

    text = "time: {0}".format(str_shoot_time)
    position = (10, 25)
    color = (0, 255, 0)  # BGR color format (绿色)
    # 绘制视频进度的文字
    cv2.putText(img, text, position, font, font_scale, color, thickness)


def export_video(video_path):
    global using_yuv
    fps = 25  # 帧率

    frame_width = width  # 视频宽度
    frame_height = height  # 视频高度

    if is_rotate_clockwise:
        frame_width = height  # 视频宽度
        frame_height = width  # 视频高度

    if enable_scale:
        frame_width = frame_width * scale
        frame_height = frame_height * scale

        # 使用os.path.basename()获取文件名

    file_name = os.path.basename(video_path)

    name_without_extension = get_name_without_extension(video_path)
    # output_path = 'output/output.mp4'
    output_path = 'output/{0}.mp4'.format(name_without_extension)

    # print(output_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 选择视频编码器（这里选择MP4V）
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    total_frame = count_frame_in_raws(video_path)

    # 获取当前图像序列开始拍摄的时间
    time_start_shoot = get_shoot_time(video_path)

    # 打开 ZIP 文件
    with zipfile.ZipFile(video_path, 'r') as zip_ref:
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

                time_0 = time.time()

                # img = []
                if using_yuv:
                    # todo get thermal image from yuv bytes in raw file
                    rgb = read_rgb_from_bytes(file_bytes, width, height)
                    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                else:
                    # todo get thermal image from termpature data in raw file
                    img = generate_thermal_image_from_bytes(file_bytes)

                if is_rotate_clockwise:
                    # 因为拍摄的时候不是竖屏，所以需要旋转
                    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                if enable_scale and scale != 1:
                    # 重新缩放显示
                    img = cv2.resize(img, (height * scale, width * scale))

                # 视频的播放进度时间
                str_progress = format_time(int(cur_frame / fps))
                # 当前帧的拍摄时间
                str_shoot_time = add_time(time_start_shoot, str_progress)

                # 在图上绘制时间标注
                if draw_time:
                    # todo 解决分辨率太低了导致图像文字不清楚的问题
                    draw_time_in_image(img, str_progress, str_shoot_time)

                if preview_begin_convert and cur_frame == 0:
                    cv2.imshow("Image", img)
                    key = cv2.waitKey(0)
                    cv2.destroyAllWindows()

                output_video.write(img)  # 将图片写入视频

                time_consuming = time.time() - time_0

                progress_percentage = "{:.2f}%".format((cur_frame / float(total_frame)) * 100)

                print("\r将{0}转换为mp4文件，进度: {1}, 第[{2}/{3}]帧, 耗时：{4} s".format(file_name, progress_percentage, cur_frame, total_frame,
                                                                round(time_consuming, 2)), end="")
        output_video.release()
        print("\n{0}转换图像序列成mp4视频完成，保存位置为：{1}".format(file_name, output_path))
