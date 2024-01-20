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


def find_largest_blue_contour(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image to HSV (Hue, Saturation, Value) color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range of blue color in HSV
    lower_blue = np.array([100, 5, 5])
    upper_blue = np.array([150, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(image_hsv, lower_blue, upper_blue)

    # Perform a series of erosions and dilations on the mask to remove small blobs
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=4)

    # Find contours in the masked image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea) if contours else None

    # Draw the largest contour on the original image
    contour_image = image_rgb.copy()
    if largest_contour is not None:
        cv2.drawContours(contour_image, [largest_contour], -1, (255, 0, 0), 2)

    return contour_image

# Process the image and find the largest blue contour
contour_image = find_largest_blue_contour(file_path)

# Convert the contour image to RGB for visualization
contour_image_rgb = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)

# Display the image with the largest contour
plt.imshow(contour_image_rgb)
plt.axis('off')
plt.show()