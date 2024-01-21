# coding = utf-8
# @Time : 2024/1/19 17:08
# @Author : moyear
# @File : blue_contours_extract.y
# @Software : PyCharm
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
file_path = r'../res/blue_contour.png'

video_path = r"E:\Moyear\文档\冲刷实验\测试数据\轨迹.mp4"


def find_largest_blue_contour(image):

    # Convert the image to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image to HSV (Hue, Saturation, Value) color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range of blue color in HSV
    lower_blue = np.array([100, 40, 10])
    upper_blue = np.array([150, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(image_hsv, lower_blue, upper_blue)

    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)

    # Perform a series of erosions and dilations on the mask to remove small blobs
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=3)
    mask = cv2.dilate(mask, kernel, iterations=3)

    cv2.imshow("mask", mask)

    # Find contours in the masked image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea) if contours else None

    # Draw the largest contour on the original image
    contour_image = image_rgb.copy()
    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 2)

    # if largest_contour is not None:
    #     cv2.drawContours(contour_image, [largest_contour], -1, (255, 0, 0), 2)

    return contour_image


def pic_extract():
    # Read the image
    image = cv2.imread(file_path)

    # Process the image and find the largest blue contour
    contour_image = find_largest_blue_contour(image)

    cv2.imshow("img", contour_image)
    cv2.waitKey(0)

    # Convert the contour image to RGB for visualization
    # contour_image_rgb = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)

    # Display the image with the largest contour
    # plt.imshow(contour_image_rgb)
    # plt.axis('off')
    # plt.show()


def video_extract():
    cap = cv2.VideoCapture(video_path)
    # 逐帧处理视频4321    ·
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('ori', frame)
        img = find_largest_blue_contour(frame)
        # img = frame
        cv2.imshow("img", img)
        cv2.waitKey(int(1000 / 25))


video_extract()