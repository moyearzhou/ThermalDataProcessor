# coding = utf-8
# @Time : 2024/1/22 20:31
# @Author : moyear
# @File : velocity_measurer.y
# @Software : PyCharm
import cv2
import numpy as np


class VelocityMeasure:
    video_path = None
    cap = None

    fps = 25

    frame_width = -1
    frame_height = -1

    frame_index = 0

    cur_frame = None
    cur_frame_with_contours = None

    # 槽子的长度为160cm
    real_height = 1600
    # 槽子的宽度40cm
    real_width = 400

    # 像素到毫米的比例，根据实际情况调整
    pixel_to_mm = -1

    rotation_type = 0

    def init_with_video(self, vid_path):
        self.video_path = vid_path
        self.cap = cv2.VideoCapture(vid_path)

        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        print('视频宽：{0}，高：{1}'.format(self.frame_width, self.frame_height))
        self.pixel_to_mm = self.real_height / self.frame_height

        print("在纵向上，每个像素为{0}mm".format(self.pixel_to_mm))

    def read(self):
        ret, frame = self.cap.read()

        # 对视频进行一定的旋转操作，使水流自上而下流动
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        self.cur_frame = frame
        self.cur_frame += 1
        return ret, frame

    def get_blue_mask(self):
        '''
        根据蓝色提取出坡面中可能为染色的区域
        :return:
        '''
        return mask_by_hsv(self.cur_frame)

    def get_ori_image(self):
        '''
        获取原始图片
        :return:
        '''
        return self.cur_frame

    def get_image_with_contours(self):
        '''
        获取在图片上绘制检测到目标轮廓的图像
        :return:
        '''
        contours = self.get_blue_contours()
        cur_frame_with_contours = self.cur_frame.copy()
        cv2.drawContours(cur_frame_with_contours, contours, -1, (0, 0, 255), 2)
        return cur_frame_with_contours

    def get_mask_after_morphological_operate(self):
        mask = self.get_blue_contours()
        mask = morphological_operate(mask)
        return mask

    def get_blue_contours(self):
        mask = mask_by_hsv(self.cur_frame)
        return find_blue_contours(mask)


def mask_by_hsv(image):
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
    image_hsv = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)

    # Define the range of blue color in HSV
    lower_blue = np.array([100, 50, 20])
    upper_blue = np.array([150, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(image_hsv, lower_blue, upper_blue)
    return mask


def morphological_operate(mask):
    '''
    :param mask:
    :return:
    '''
    # Perform a series of erosions and dilations on the mask to remove small blobs
    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=5)
    return mask


def find_blue_contours(mask):
    '''
    通过腐蚀和膨胀等形态学操作识别对轮廓进行过滤
    :param mask:
    :return:
    '''
    mask = morphological_operate(mask)
    cv2.imshow("mask", mask)

    # Find contours in the masked image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


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
