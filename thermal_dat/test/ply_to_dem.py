# coding = utf-8
# @Time : 2024/1/13 20:28
# @Author : moyear
# @File : ply_to_dem.y
# @Software : PyCharm

import numpy as np
import rasterio
from rasterio.transform import from_origin
from pyntcloud import PyntCloud
from scipy.spatial import Delaunay

path_ply = r"E:\Users\Moyear\Desktop\3d\纯坡面点云.ply"

out_path = r"../output/output_dem.tif"

# 读取点云数据
cloud = PyntCloud.from_file(path_ply)

# 获取点云数据的 x, y, z 值
points = cloud.points
x = points["x"]
y = points["y"]
z = points["z"]

# 创建一个空的 DEM 数据
dem = np.zeros((int(y.max() - y.min() + 1), int(x.max() - x.min() + 1)))

# # 将点云数据转换为 DEM 数据
# for xi, yi, zi in zip(x, y, z):
#     dem[int(yi - y.min()), int(xi - x.min())] = zi
#
# # 创建一个新的 rasterio 数据集
# with rasterio.open(
#     out_path,
#     'w',
#     driver='GTiff',
#     height=dem.shape[0],
#     width=dem.shape[1],
#     count=1,
#     dtype=dem.dtype,
#     crs='+proj=latlong',
#     transform=from_origin(x.min(), y.max(), 1, 1),
# ) as dst:
#     dst.write(dem, 1)

# 进行 Delaunay 三角剖分
tri = Delaunay(np.vstack((x, y)).T)

# 对每个三角形进行插值
for i in range(tri.simplices.shape[0]):
    # 获取三角形的顶点和对应的 z 值
    vertices = tri.points[tri.simplices[i]]
    z_values = z[tri.simplices[i]]

    # 计算三角形的边界
    min_x = vertices[:, 0].min()
    max_x = vertices[:, 0].max()
    min_y = vertices[:, 1].min()
    max_y = vertices[:, 1].max()

    # 对边界内的每个点进行插值
    for xi in range(int(min_x), int(max_x + 1)):
        for yi in range(int(min_y), int(max_y + 1)):
            # 计算点到三角形各顶点的距离
            distances = np.sqrt((vertices[:, 0] - xi) ** 2 + (vertices[:, 1] - yi) ** 2)

            # 计算插值的 z 值
            z_interpolated = np.sum(z_values / distances) / np.sum(1 / distances)

            # 更新 DEM 数据
            dem[yi - int(y.min()), xi - int(x.min())] = z_interpolated

# 创建一个新的 rasterio 数据集
with rasterio.open(
    'output.tif',
    'w',
    driver='GTiff',
    height=dem.shape[0],
    width=dem.shape[1],
    count=1,
    dtype=dem.dtype,
    crs='+proj=latlong',
    transform=from_origin(x.min(), y.max(), 1, 1),
) as dst:
    dst.write(dem, 1)