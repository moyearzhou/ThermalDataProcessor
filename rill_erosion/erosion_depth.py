# coding = utf-8
# @Time : 2024/2/29 15:39
# @Author : moyear
# @File : erosion_depth.y
# @Software : PyCharm
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt


def calculate_min_elevation(ply_path):
    # 加载点云数据
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)

    # 动态获取y值的范围，y的范围大概在-800到800之间
    y_min, y_max = points[:, 1].min(), points[:, 1].max()

    # 创建y值的数组，以确保覆盖所有可能的y值
    y_step = 10  # 以10 mm为一个采样
    y_bins = np.arange(y_min, y_max + y_step, y_step)
    min_z_values = np.full(y_bins.shape, np.nan)  # 初始化最小高程值数组

    # 根据y值分组并找到每组的最低z值
    for i, y in enumerate(y_bins):
        z_values_at_y = points[(points[:, 1] >= y) & (points[:, 1] < y + y_step), 2]
        if z_values_at_y.size > 0:
            min_z_values[i] = np.min(z_values_at_y)

    # 用前一个有效值填充NaN值
    valid_min_z = np.where(np.isnan(min_z_values),
                           np.interp(y_bins, y_bins[~np.isnan(min_z_values)], min_z_values[~np.isnan(min_z_values)]),
                           min_z_values)
    return y_bins - y_min, valid_min_z[::-1]


def get_slope_min_elevation(ply_path_before, ply_path_after):
    # 计算侵蚀前和侵蚀后的最低高程
    y_plot_before, x_plot_before = calculate_min_elevation(ply_path_before)
    y_plot_after, x_plot_after = calculate_min_elevation(ply_path_after)

    # 找到两个数组的共享范围
    y_min = max(np.min(y_plot_before), np.min(y_plot_after))
    y_max = min(np.max(y_plot_before), np.max(y_plot_after))

    # 计算侵蚀的深度
    # erosion_depth = x_plot_after - x_plot_before

    # 只选择这个范围内的数值
    mask_before = (y_plot_before >= y_min) & (y_plot_before <= y_max)
    mask_after = (y_plot_after >= y_min) & (y_plot_after <= y_max)

    y_plot_before, x_plot_before = y_plot_before[mask_before], x_plot_before[mask_before]
    y_plot_after, x_plot_after = y_plot_after[mask_after], x_plot_after[mask_after]

    # 计算侵蚀的深度
    erosion_depth = x_plot_after - x_plot_before

    return y_plot_before, erosion_depth


# def save_slope_min_elevation(ply_path_before, ply_path_after):
#     # 计算侵蚀前和侵蚀后的最低高程
#     y_plot_before, erosion_depth = get_slope_min_elevation(ply_path_before, ply_path_after)
#
#     # 保存数据到CSV文件
#     output_path = r"outputs/erosion_depth.csv"
#     np.savetxt(output_path, np.column_stack((y_plot_before, erosion_depth)), delimiter=',',
#                header='Distance from Top of Slope (y),Erosion Depth (z)', comments='')


def test_slope_elevation():
    # 侵蚀前和侵蚀后的点云文件路径
    ply_path_before = r"E:\Users\Moyear\Desktop\3d\slopes\D1_0_20240116_B.ply"
    ply_path_after = r"E:\Users\Moyear\Desktop\3d\slopes\D1_4_20240116_B.ply"

    # 计算侵蚀前和侵蚀后的最低高程
    y_plot_before, erosion_depth = get_slope_min_elevation(ply_path_before, ply_path_after)

    # 绘制侵蚀深度曲线
    plt.figure(figsize=(10, 5))
    plt.plot(y_plot_before, erosion_depth, label='Erosion Depth')
    plt.xlabel('Distance from Top of Slope (y)')
    plt.ylabel('Erosion Depth (z)')
    plt.title('Erosion Depth Profile Along Slope')
    plt.legend()
    plt.show()

