# coding = utf-8
# @Time : 2023/12/15 14:03
# @Author : moyear
# @File : thermal_flow_extract.y
# @Software : PyCharm
import cv2
from matplotlib import pyplot as plt

img = ''



def extra_boundary(image_path):
    global img
    img = cv2.imread(img_path)  # 输入图像
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # 计算灰度直方图
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    hist = hist.flatten()

    # 绘制概率分布图
    plt.plot(hist, color='black')
    plt.xlabel('Pixel Value')
    plt.ylabel('Probability')
    plt.title('Grayscale Histogram')
    plt.xlim([0, 256])
    plt.ylim([0, max(hist)])
    plt.show()

    # 应用阈值分割
    threshold_value = 28  # 阈值
    max_value = 255  # 最大像素值
    _, thresholded_image = cv2.threshold(img_gray, threshold_value, max_value, cv2.THRESH_TOZERO)


    # cv2.imshow("gray", img_gray)

    cv2.imshow("ori",  img)
    cv2.imshow("result_img", thresholded_image)

    # cv2.setMouseCallback("original_img", capture_event)
    cv2.waitKey(0)



if __name__ == '__main__':
    img_path = r"../output/transformed.jpg"
    extra_boundary(img_path)
