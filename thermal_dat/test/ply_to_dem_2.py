# coding = utf-8
# @Time : 2024/1/14 2:06
# @Author : moyear
# @File : ply_to_dem_2.y
# @Software : PyCharm
import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
# from tqdm import tqdm


path_ply = r"E:\Users\Moyear\Desktop\3d\纯坡面点云.ply"

out_path = r"../output/output_dem.tif"

# 读取点云数据
point_cloud = o3d.io.read_point_cloud(path_ply)

# 获取点云坐标
points = point_cloud.points

# 创建TIN
tri = Delaunay(points)

# 定义DEM网格范围和分辨率
x_min, x_max = min(points[:, 0]), max(points[:, 0])
y_min, y_max = min(points[:, 1]), max(points[:, 1])
resolution = 1.0  # DEM网格分辨率

# 生成DEM数据
x = np.arange(x_min, x_max, resolution)
y = np.arange(y_min, y_max, resolution)
X, Y = np.meshgrid(x, y)
Z = griddata(points[:, :2], points[:, 2], (X, Y), method='linear')

# 显示进度条
# progress_bar = tqdm(total=len(points))

total = len(points)
progress = 0

# 模拟处理每个点
for point in points:
    # 处理点云数据...
    # 更新进度条
    # progress_bar.update(1)
    progress += 1
    print("{0}/{1}".format(progress, total))


# 关闭进度条
# progress_bar.close()
