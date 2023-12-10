# coding = utf-8
# @Time : 2023/12/10 12:17
# @Author : moyear
# @File : raw_video_convert.y
# @Software : PyCharm
import os
import re
import time

import cv2

from parse_utils import extra_raw_video, generate_thermal_image, read_rgb_from
from thermal_dat.utils import count_files

height = 256
width = 192


def export_video(video_path, using_yuv=False):
    fps = 25  # 帧率

    is_rotate = True

    frame_width = width  # 视频宽度
    frame_height = height  # 视频高度

    if is_rotate:
        frame_width = height  # 视频宽度
        frame_height = width  # 视频高度

    output_path = 'output/output.mp4'

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 选择视频编码器（这里选择MP4V）
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # extract zip file of raw video, and get the path to extract
    raw_pic_dir = extra_raw_video(video_path)

    total_frame = count_files(raw_pic_dir)

    for file_name in os.listdir(raw_pic_dir):
        if not re.fullmatch(r"\d{8}", file_name):
            continue

        file_path = raw_pic_dir + "/" + file_name

        time_0 = time.time()

        # img = []
        if using_yuv:
            # todo get thermal image from yuv bytes in raw file
            rgb = read_rgb_from(file_path, width, height)
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            # todo get thermal image from termpature data in raw file
            with open(file_path, 'rb') as file:
                img = generate_thermal_image(file)

        if is_rotate:
            # 因为拍摄的时候不是竖屏，所以需要旋转
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        output_video.write(img)  # 将图片写入视频

        print("正在写入第{0}/{1}帧, 耗时：{2} s".format(int(file_name), total_frame, time.time() - time_0))

    output_video.release()
    print("转换图像序列成视频完成，保存位置为：{0}".format(output_path))
