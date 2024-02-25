# coding = utf-8
# @Time : 2024/2/23 22:35
# @Author : moyear
# @File : ply_roughness_cal.y
# @Software : PyCharm
import open3d as o3d
import numpy as np


path_ply = r"E:\Users\Moyear\Desktop\3d\TEST_1.ply"

# 读取PLY文件
pcd = o3d.io.read_point_cloud(path_ply)

# 将点云数据转换为numpy数组
points = np.asarray(pcd.points)

# 计算高程的标准差
roughness = np.std(points[:, 2])

print("Surface roughness: ", roughness)