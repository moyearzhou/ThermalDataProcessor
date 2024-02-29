# coding = utf-8
# @Time : 2024/2/29 22:20
# @Author : moyear
# @File : erosion_width.y
# @Software : PyCharm
import numpy as np
from matplotlib import pyplot as plt

from rill_erosion.erosion_dem_extract import cal_volume_change


# def calculate_erosion_width(dem_diff, y_min, y_max, erosion_depth=-0.5, step=10):
#     """
#     计算每个y值对应的侵蚀区域宽度
#     :param dem_diff: 高程差异矩阵
#     :param y_min: y轴的最小值
#     :param y_max: y轴的最大值
#     :param erosion_depth: 侵蚀深度阈值
#     :param step: y轴采样步长
#     :return: 采样后的y轴坐标和对应的侵蚀宽度
#     """
#     y_coords = np.arange(y_min, y_max, step)
#     erosion_widths = []
#
#     print(y_coords)
#
#     for y in y_coords:
#         # 在当前步长内，找出所有侵蚀区域
#         erosion_areas = dem_diff[(dem_diff[:, 1] >= y) & (dem_diff[:, 1] < y + step) & (dem_diff[:, 2] < erosion_depth)]
#         # 如果有侵蚀区域，计算宽度
#         if erosion_areas.size > 0:
#             # 侵蚀宽度是x轴方向上侵蚀区域长度之和
#             width = np.sum(np.unique(erosion_areas[:, 0]))
#             erosion_widths.append(width)
#         else:
#             # 如果没有侵蚀区域，则宽度为0
#             erosion_widths.append(0)
#
#     return y_coords, erosion_widths


# def calculate_erosion_width(dem_diff, erosion_depth=-0.5, step=10):
#     """
#     计算每个y值对应的侵蚀区域宽度
#     :param dem_diff: 高程差异矩阵
#     :param erosion_depth: 侵蚀深度阈值
#     :param step: y轴采样步长
#     :return: 采样后的y轴坐标和对应的侵蚀宽度
#     """
#     y_max = dem_diff.shape[1]
#     erosion_widths = []
#     y_coords = np.arange(0, y_max, step)
#
#     # 遍历y轴
#     for y in range(0, y_max, step):
#         y_end = min(y + step, y_max)  # 确保不会越界
#         # 在当前步长内，找出所有侵蚀区域的高程值
#         erosion_mask = dem_diff[:, y:y_end] < erosion_depth
#         # 计算宽度
#         width = np.sum(erosion_mask, axis=0)
#         erosion_widths.append(np.mean(width))
#
#     return y_coords[:-1], erosion_widths  # 返回结果，除去最后一个可能的不完整区间


def calculate_erosion_width(dem_diff, erosion_depth=-5, step=10):
    y_max = dem_diff.shape[1]
    erosion_widths = []
    y_coords = []

    for y in range(0, y_max, step):
        if y + step > y_max:
            # If adding step exceeds the bounds, just take the remaining portion
            y_end = y_max
        else:
            y_end = y + step

        # Append the current y coordinate for plotting
        y_coords.append(y)

        # Find erosion width for the current step
        erosion_mask = dem_diff[:, y:y_end] < erosion_depth
        width = np.sum(erosion_mask, axis=0)
        erosion_widths.append(np.mean(width))

    return np.array(y_coords), np.array(erosion_widths)


def get_slope_min_elevation(path_slope_base, path_target):
    dem_diff = cal_volume_change(path_slope_base, path_target)

    # 采样最低高程变化
    y_coords, erosion_widths = calculate_erosion_width(dem_diff)

    # 反转一下才是从坡顶到坡地的宽度
    erosion_widths = erosion_widths[::-1]

    return y_coords, erosion_widths


def test_slope_width():
    path_slope_base = r"E:\Users\Moyear\Desktop\3d\slopes\D1_0_20240116_B.ply"
    path_target = r"E:\Users\Moyear\Desktop\3d\slopes\D1_4_20240116_B.ply"

    threshold = 0

    # dem_diff = cal_volume_change(path_slope_base, path_target)

    # roi_region = (dem_diff < threshold)
    # print(roi_region.shape)

    # 获取y轴的最小值和最大值
    # y_min, y_max = np.min(dem_diff[:, 1]), np.max(dem_diff[:, 1])

    # 计算侵蚀宽度
    # y_coords, erosion_widths = calculate_erosion_width(dem_diff, y_min, y_max)

    # 采样最低高程变化
    # y_coords, erosion_widths = calculate_erosion_width(dem_diff)
    #
    # # 反转一下才是从坡顶到坡地的宽度
    # erosion_widths = erosion_widths[::-1]

    y_coords, erosion_widths = get_slope_min_elevation(path_slope_base, path_target)

    print(erosion_widths)

    # 绘制侵蚀宽度变化曲线
    plt.figure(figsize=(12, 6))
    plt.plot(y_coords, erosion_widths)
    plt.xlabel('Distance from Top of Slope (y)')
    plt.ylabel('Erosion Width')
    plt.title('Erosion Width Change Along Slope')
    plt.grid(True)
    plt.show()
