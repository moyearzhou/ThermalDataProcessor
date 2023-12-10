# coding = utf-8
# @Time : 2023/12/3 22:47
# @Author : moyear
# @File : camera_correct.y
# @Software : PyCharm
import os

import numpy as np
import cv2

# 棋盘格的行数和列数
rows = 6
cols = 9

# 棋盘格的尺寸（单位：毫米）
square_size = 25    # todo 根据实际棋盘格子的大小修改

# 存储棋盘格角点的世界坐标和图像坐标
world_points = []
image_points = []

# 生成棋盘格的世界坐标
for i in range(rows):
    for j in range(cols):
        world_points.append([j * square_size, i * square_size, 0])

# 设置相机参数
camera_matrix = np.zeros((3, 3))
dist_coeffs = np.zeros((5, 1))

image_path = "res"

# 获取图像列表
image_files = []

for file_name in os.listdir(image_path):
    file_path = image_path + "/" + file_name
    image_files.append(file_path)

# 遍历图像列表进行标定
for image_file in image_files:
    # 读取图像
    image = cv2.imread(image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 寻找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

    # 如果找到角点，则添加到标定数据中
    if ret:
        # 添加世界坐标
        world_points.append(world_points)

        # 添加图像坐标
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        image_points.append(corners2)

        # 绘制角点
        cv2.drawChessboardCorners(image, (cols, rows), corners2, ret)

    # 显示图像
    cv2.imshow('Chessboard', image)
    cv2.waitKey(500)

# 进行相机标定
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(world_points, image_points, gray.shape[::-1], None, None)

# 打印相机矩阵和畸变系数
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)

# 保存标定结果
np.save("camera_matrix.npy", camera_matrix)
np.save("dist_coeffs.npy", dist_coeffs)

# 关闭窗口
cv2.destroyAllWindows()


