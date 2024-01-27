# coding = utf-8
# @Time : 2024/1/22 20:31
# @Author : moyear
# @File : velocity_measurer.y
# @Software : PyCharm
import csv
import os

import cv2
import numpy as np


class VelocityMeasure:
    video_path = None
    cap = None

    fps = 25

    frame_width = -1
    frame_height = -1

    frame_width_transformed = 200
    frame_height_transformed = 800

    cur_frame_index = 0

    # 当前帧图像bgr模式
    cur_frame_raw = None

    cur_frame_transformed_rgb = None

    cur_frame_transformed_with_contours = None

    # 槽子的长度为160cm
    real_height = 1600
    # 槽子的宽度40cm
    real_width = 400

    # 像素到毫米的比例，根据实际情况调整
    pixel_to_mm = -1

    rotation_type = 0

    # 在第几帧开始冲刷的
    frame_index_to_scouring = None

    measure_points = []

    # 视频名称
    video_name = ''

    # 坡面名称
    slope_name = ''

    # 最终的测量结果
    result_measures = []

    # 腐蚀操作次数
    times_erosion = 1

    # 膨胀操作次数
    times_dilation = 5

    # hsv的下限范围
    hsv_lower = [100, 50, 20]

    # hsv的范围上限
    hsv_upper = [150, 255, 255]

    # 用于透视变换的控制点
    points_for_transform = []
    # 用于透视变换的控制点是否排好序
    is_transformed_points_ordered = False
    # 是否应用了透视变换
    is_transform_applied = False

    def init_with_video(self, vid_path):
        self.video_path = vid_path

        self.video_name = os.path.basename(vid_path)

        self.cap = cv2.VideoCapture(vid_path)

        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        print('视频宽：{0}，高：{1}'.format(self.frame_width, self.frame_height))

        self.pixel_to_mm = self.real_height / self.frame_height_transformed
        print("在纵向上，每个像素为{0}mm".format(self.pixel_to_mm))

        # 导入视频先把之前的变换置空
        self.is_transform_applied = False
        self.points_for_transform = []

    def is_init(self):
        if self.video_path is None or self.video_path == "" or self.cap is None:
            return False

        return True

    def set_slope_name(self, name):
        self.slope_name = name

    def set_transform_points(self, points):
        if points is None or len(points) != 6:
            return
        self.points_for_transform = points

        self.is_transformed_points_ordered = False
        self.is_transform_applied = True

    def read(self):
        ret, frame = self.cap.read()

        self.cur_frame_raw = frame

        # print(self.points_for_transform)

        # image_roi = frame

        if self.is_transform_applied and len(self.points_for_transform) == 6:
            image_roi = self.perform_perspective_correction(frame, self.points_for_transform)
            self.cur_frame_transformed_rgb = cv2.cvtColor(image_roi, cv2.COLOR_BGRA2RGB)

        self.cur_frame_index += 1
        return ret, frame

    def get_rotated_image(self, image):
        rotate_image = image
        # 对视频进行一定的旋转操作，使水流自上而下流动
        if self.rotation_type == 1:
            rotate_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation_type == 2:
            rotate_image = cv2.rotate(image, cv2.ROTATE_180)
        elif self.rotation_type == 3:
            rotate_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return rotate_image

    def skip_to_frame(self, frame_num):
        if frame_num < 0 or frame_num >= self.get_total_frames():
            print('要跳转到帧数[{0}]有误！！'.format(frame_num))
            return

        # 跳转到第 100 帧
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        self.cur_frame_index = frame_num
        self.read()

    def forward(self, frame_num):
        target_frame_index = self.cur_frame_index + frame_num

        if target_frame_index < 0 or target_frame_index >= self.get_total_frames():
            print('要的前进帧数[{0}]有误！！'.format(frame_num))
            return
        self.skip_to_frame(target_frame_index)

    def rewind(self, frame_num):
        target_frame_index = self.cur_frame_index - frame_num

        if target_frame_index < 0 or target_frame_index >= self.get_total_frames():
            print('要后退的帧数[{0}]有误！！'.format(frame_num))
            return

        self.skip_to_frame(target_frame_index)

    def get_total_frames(self):
        # Get the total number of frames
        total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return total_frames

    def get_video_image(self):
        image_video = self.cur_frame_raw.copy()
        image_video = cv2.cvtColor(image_video, cv2.COLOR_BGRA2RGB)

        radius = 10  # 圆点的半径
        color = (0, 0, 255)  # 圆点的颜色，这里使用红色
        thickness = -1  # 圆点的填充，-1表示填充整个圆

        font = cv2.FONT_HERSHEY_SIMPLEX  # 字体类型
        font_scale = 1  # 字体缩放因子
        text_color = (0, 255, 0)  # 文字的颜色，这里使用绿色
        text_thickness = 2  # 文字的线宽

        if self.points_for_transform is not None and self.points_for_transform != []:
            for point in self.points_for_transform:
                x = int(point[0])
                y = int(point[1])

                # 绘制圆点
                center = (x, y)  # 圆点的中心坐标
                cv2.circle(image_video, center, radius, color, thickness)

                # 添加文字
                text = "({0},{1})".format(x, y)
                org = (int(point[0] + 10), int(point[1]) + 10)  # 文字的起始坐标

                cv2.putText(image_video, text, org, font, font_scale, text_color, text_thickness)

        return image_video

    # def get_blue_mask(self):
    #     '''
    #     根据蓝色提取出坡面中可能为染色的区域
    #     :return:
    #     '''
    #     return self.mask_by_hsv(self.cur_frame_transformed_rgb)

    def get_transformed_rgb_image(self):
        '''
        获取原始图片
        :return:
        '''
        if not self.is_transform_applied:
            return None

        transformed_image_rgb = self.cur_frame_transformed_rgb

        # 对视频进行进行旋转操作
        transformed_image_rgb = self.get_rotated_image(transformed_image_rgb)

        return transformed_image_rgb

    def get_transformed_image_with_contours(self):
        '''
        获取在图片上绘制检测到目标轮廓的图像
        :return:
        '''
        if not self.is_transform_applied:
            return None

        cur_frame_with_contours = None

        contours = self.get_blue_contours()
        if contours is not None:
            cur_frame_with_contours = self.cur_frame_transformed_rgb.copy()
            cv2.drawContours(cur_frame_with_contours, contours, -1, (0, 0, 255), 2)

        # 对图像进行一定程度的旋转
        cur_frame_with_contours = self.get_rotated_image(cur_frame_with_contours)
        return cur_frame_with_contours

    def get_transformed_image_with_measure_points(self):
        if not self.is_transform_applied:
            return None

        if self.cur_frame_transformed_rgb is None:
            return None

        image_with_measure_points = self.cur_frame_transformed_rgb.copy()

        image_with_measure_points = self.get_rotated_image(image_with_measure_points)

        # # todo 绘制刻度线
        # # 计算刻度线的宽度和高度
        # scale_width = 20  # 刻度线的宽度
        # scale_height = self.frame_height // 5  # 刻度线的高度，将图片高度分成5等分
        #
        # width = self.frame_width
        #
        # # 在图片右侧绘制刻度线
        # for i in range(5):
        #     y_start = i * scale_height
        #     y_end = y_start + scale_height
        #     image_with_measure_points = cv2.line(image_with_measure_points, (width, y_start), (width + scale_width, y_start), (0, 255, 0), 2)
        #     image_with_measure_points = cv2.line(image_with_measure_points, (width, y_end), (width + scale_width, y_end), (0, 255, 0), 2)

        radius = 5  # 圆点的半径
        color = (0, 0, 255)  # 圆点的颜色，这里使用红色
        thickness = -1  # 圆点的填充，-1表示填充整个圆

        font = cv2.FONT_HERSHEY_SIMPLEX  # 字体类型
        font_scale = 0.5  # 字体缩放因子
        text_color = (0, 255, 0)  # 文字的颜色，这里使用绿色
        text_thickness = 1  # 文字的线宽

        # print(self.measure_points)

        if self.measure_points is not None and self.measure_points != []:
            for point in self.measure_points:
                x = int(point[0])
                y = int(point[1])

                # frame_index = point[1]
                # point_time = frame_to_time_progress(frame_index, self.fps)

                # 绘制圆点
                center = (x, y)  # 圆点的中心坐标
                cv2.circle(image_with_measure_points, center, radius, color, thickness)

                # 添加文字
                text = "({0},{1})".format(x, y)
                org = (int(point[0] + 10), int(point[1]) + 10)  # 文字的起始坐标

                cv2.putText(image_with_measure_points, text, org, font, font_scale, text_color, text_thickness)

        return image_with_measure_points

    # def get_mask_after_morphological_operate(self):
    #     mask = self.get_blue_contours()
    #     mask = self.morphological_operate(mask)
    #     return mask

    def get_blue_contours(self):
        if self.cur_frame_transformed_rgb is None:
            return None

        mask = self.mask_by_hsv(self.cur_frame_transformed_rgb)
        return self.find_blue_contours(mask)

    def clear_measure_points(self):
        self.measure_points = []

    def set_current_frame_as_init_scouring(self):
        self.frame_index_to_scouring = self.cur_frame_index

    def add_measure_point(self, x, y, frame_index):
        if x < 0 or x >= self.frame_width_transformed:
            print('测速点x位置不合理')
            return
        if y < 0 or y >= self.frame_height_transformed:
            print('测速点y位置不合理')
            return
        if frame_index < 0 or frame_index >= self.get_total_frames():
            print('测速点的时间帧不合理')
            return
        self.measure_points.append((x, y, frame_index))

    def export_measures_to_csv(self, save_path):
        if len(self.result_measures) == 0:
            return False

        print(self.result_measures)

        save_to_csv(self.result_measures, save_path)
        return True

    def mask_by_hsv(self, image):
        '''
        提取坡面中的蓝色部分区域
        :param image:
        :return:
        '''
        # 进行高斯滤波
        # 第二个参数是高斯核的大小，必须是奇数
        # 第三个参数是高斯核的标准差，在X和Y方向上，如果设为0，则函数会自动计算
        blurred_img = cv2.GaussianBlur(image, (5, 5), 0)

        # Convert the image to HSV (Hue, Saturation, Value) color space
        image_hsv = cv2.cvtColor(blurred_img, cv2.COLOR_RGB2HSV)

        # Define the range of blue color in HSV
        lower_blue = np.array(self.hsv_lower)
        upper_blue = np.array(self.hsv_upper)

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(image_hsv, lower_blue, upper_blue)
        return mask

    def morphological_operate(self, mask):
        '''
        :param mask:
        :return:
        '''
        # Perform a series of erosions and dilations on the mask to remove small blobs
        kernel = np.ones((3, 3), np.uint8)

        mask = cv2.erode(mask, kernel, iterations=self.times_erosion)
        mask = cv2.dilate(mask, kernel, iterations=self.times_dilation)
        return mask

    def find_blue_contours(self, mask):
        '''
        通过腐蚀和膨胀等形态学操作识别对轮廓进行过滤
        :param mask:
        :return:
        '''
        mask = self.morphological_operate(mask)
        # cv2.imshow("mask", mask)

        # Find contours in the masked image
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    # Function to perform perspective correction
    def perform_perspective_correction(self, frame, selected_points):
        # 对选择的6个点进行排序，顺序为：左上点、右上点、左中点、右中点、左下点、右下点
        if not self.is_transformed_points_ordered:
            selected_points = order_points(selected_points)

        width = 200
        height = 800

        half_height = int(height / 2)

        src_points_upper = np.array([selected_points[0], selected_points[1], selected_points[2], selected_points[3]],
                                    dtype='float32')
        src_points_lower = np.array([selected_points[2], selected_points[3], selected_points[4], selected_points[5]],
                                    dtype='float32')

        # Define destination points for upper and lower halves
        dst_points = np.array([[0, 0], [width, 0], [0, half_height], [width, half_height]], dtype='float32')

        # Compute the perspective transform matrices for upper and lower halves
        matrix_upper = cv2.getPerspectiveTransform(src_points_upper, dst_points)
        matrix_lower = cv2.getPerspectiveTransform(src_points_lower, dst_points)
        # Apply perspective transformation for upper and lower halves
        transformed_upper = cv2.warpPerspective(frame, matrix_upper, (width, half_height))
        transformed_lower = cv2.warpPerspective(frame, matrix_lower, (width, half_height))

        # Concatenate the two halves
        transformed_image = np.concatenate((transformed_upper, transformed_lower), axis=0)
        return transformed_image


def get_contour_centers(contours):
    # 计算每个轮廓的中心点
    contours_centers = []
    for cnt in contours:
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        contours_centers.append([cX, cY])
    return contours_centers


def get_center_point(contour):
    M = cv2.moments(contour)

    center = (-1, -1)
    if M['m00'] != 0:
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
    return center


def save_to_csv(data, file_path):
    keys = data[0].keys()

    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)


def order_points(points):
    '''
    对选择的6个点进行排序，顺序为：左上点、右上点、左中点、右中点、左下点、右下点
    :param points:
    :return:
    '''
    # 按 y 坐标排序
    points.sort(key=lambda point: point[1])
    # 取出 y 坐标最小的两个点（顶部），并按 x 坐标排序
    top_points = sorted(points[:2], key=lambda point: point[0])
    # 取出 y 坐标最大的两个点（底部），并按 x 坐标排序
    bottom_points = sorted(points[-2:], key=lambda point: point[0])

    # 剩余的点为中部，按 x 坐标排序
    middle_points = sorted(points[2:-2], key=lambda point: point[0])
    # 将三组点合并成一个列表
    sorted_points = top_points + middle_points + bottom_points
    # print(sorted_points)
    return sorted_points


