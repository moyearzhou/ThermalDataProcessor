# coding = utf-8
# @Time : 2024/2/29 14:51
# @Author : moyear
# @File : test.y
# @Software : PyCharm


# ply_path = r"E:\Users\Moyear\Desktop\3d\slopes\B2_4_20240109_A.ply"
#
# # 加载点云数据
# pcd = o3d.io.read_point_cloud(ply_path)
# points = np.asarray(pcd.points)
#
# # 动态获取y值的范围，y的范围大概在-800到800之间
# y_min, y_max = points[:, 1].min(), points[:, 1].max()
#
# # 创建y值的数组，以确保覆盖所有可能的y值
# y_step = 10  # 以10 mm为一个采样
# y_bins = np.arange(y_min, y_max + y_step, y_step)
# min_z_values = np.full(y_bins.shape, np.nan)  # 初始化最小高程值数组
#
# # 根据y值分组并找到每组的最低z值
# for i, y in enumerate(y_bins):
#     z_values_at_y = points[(points[:, 1] >= y) & (points[:, 1] < y + y_step), 2]
#     if z_values_at_y.size > 0:
#         min_z_values[i] = np.min(z_values_at_y)
#
# # 用前一个有效值填充NaN值
# valid_min_z = np.where(np.isnan(min_z_values),
#                        np.interp(y_bins, y_bins[~np.isnan(min_z_values)], min_z_values[~np.isnan(min_z_values)]),
#                        min_z_values)
#
# # 最后绘图用的数据x，y数据
# y_plot = y_bins - y_min
# # x_plot = valid_min_z
#
# # 根据最终的效果是要进行反转一下再显示
# x_plot = valid_min_z[::-1]
#
# print(x_plot.size)
#
# # 绘制曲线
# plt.figure(figsize=(10, 5))
# plt.plot(y_plot, x_plot, label='Minimum Elevation')
# plt.xlabel('Distance from Top of Slope (y)')
# plt.ylabel('Minimum Elevation (z)')
# plt.title('Minimum Elevation Profile Along Slope After Erosion')
# plt.legend()
# plt.show()
