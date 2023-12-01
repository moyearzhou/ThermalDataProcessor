# coding = utf-8
# @Time : 2023/6/23 16:20
# @Author : moyear
# @File : main.pyt.y
# @Software : PyCharm
import ffmpeg
import numpy as np
import cv2




def read_official_video(video_path):
    '''
    读取并播放官方app拍摄出来的视频
    :param video_path: 这个视频路径是微影热视app拍出来的视频
    :return:
    '''
    start_time = 18  # 起始时间（以秒为单位）
    end_time = 30  # 结束时间（以秒为单位）

    # 打开视频文件
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # 获取输入视频的宽度和高度
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("当前帧率：{0}，视频宽度：{1}，视频高度：{2}".format(fps, frame_width, frame_height))

    # 计算起始和结束帧数
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # 设置当前帧数为起始帧
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 用于视频裁剪的参数
    x = 200
    y = 0
    width = 300
    height = frame_height

    # 旋转角度
    angle = -4

    # 初始化播放状态标志
    is_playing = True

    # 循环读取视频帧
    for frame_num in range(start_frame, end_frame):
        if not is_playing:
            key = cv2.waitKey()
            # 按下空格键切换播放状态
            if key == ord(' '):
                is_playing = not is_playing

        ret, frame = video_capture.read()
        if not ret:
            break

        # 在这里可以对读取到的帧进行处理
        # 例如，可以显示帧或对其进行其他操作

        # 旋转图像
        M = cv2.getRotationMatrix2D((frame_width // 2, frame_height // 2), angle, 1.0)
        frame = cv2.warpAffine(frame, M, (frame_width, frame_height))

        # 裁剪图像
        frame = frame[y:y + height, x:x + width]

        # 二值化的阈值
        threshold_value = 100

        # # 将图像分割为通道
        # b, g, r = cv2.split(frame)
        # # 以红色通道作为灰度图
        # gray_image = r

        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
        _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_TOZERO)

        # 显示帧
        cv2.imshow('Original Image', frame)
        # 显示二值化后的图像
        cv2.imshow("Gray Image", gray_image)
        # 图像播放间隔时间
        delay = int(float(1 / int(fps) * 1000))
        # 显示二值化后的图像
        cv2.imshow("Binary Image", binary_image)

        key = cv2.waitKey(delay)
        # 按下空格键切换播放状态
        if key == ord(' '):
            is_playing = not is_playing
        # 按下 'q' 键退出循环
        elif key == ord('q'):
            break

    # 释放视频捕获对象和关闭窗口
    video_capture.release()
    cv2.destroyAllWindows()
