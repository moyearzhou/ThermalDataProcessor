# coding = utf-8
# @Time : 2024/2/29 10:07
# @Author : moyear
# @File : main_slopes_operate.y
# @Software : PyCharm
import os

import cv2
import numpy as np
import pandas as pd

from rill_erosion.cal_fractal_dimension import fractal_dimension
from rill_erosion.erosion_dem_extract import get_detailed_slope_erosion_image, cal_volume_change
from rill_erosion.erosion_depth import test_slope_elevation, get_slope_min_elevation
from rill_erosion.erosion_width import test_slope_width


def get_slopes_ply_by_name(slope_name):
    dir_plys = r"E:\Users\Moyear\Desktop\3d\slopes"

    # slope_name = 'A0'
    # 初始化列表来保存文件路径
    file_paths_list = []

    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(dir_plys):
        for file in files:
            if file.startswith(slope_name) and file.endswith('.ply'):
                ply_file_path = os.path.join(root, file)
                file_paths_list.append(ply_file_path)

    return file_paths_list


def read_eroded_dem_and_save(slope_name):
    '''
    读取坡面各个阶段的侵蚀dem，并且保存
    :param slope_ply_paths:
    :return:
    '''

    # 坡面的各个阶段的地形
    slope_ply_paths = get_slopes_ply_by_name(slope_name)

    # 获取侵蚀DEM
    slopes_erosion = get_detailed_slope_erosion_image(slope_ply_paths)

    # 使用opencv显示图像
    slopes_erosion_bgr = cv2.cvtColor(slopes_erosion, cv2.COLOR_RGB2BGR)

    # # 使用OpenCV显示图像
    # cv2.imshow(slope_name, slopes_erosion_bgr)
    # cv2.waitKey(0)

    out_path = './outputs/erosion_{0}.jpg'.format(slope_name)
    # 将图像保存到文件
    cv2.imwrite(out_path, slopes_erosion_bgr)
    print(slope_name, "坡面侵蚀图像保存成功到", out_path)


def cal_slope_dimension_fraction(slope_ply_paths):
    for path in slope_ply_paths:
        pass


slope_names = ['A0',
               'B1', 'B2', 'B3',
               'C1', 'C2', 'C3',
               'D1', 'D2', 'D3',
               'E1', 'E2', 'E3']


def get_all_slopes_depth():
    # slope_name = 'D1'

    # 创建一个Excel写入器
    writer = pd.ExcelWriter(r"outputs/erosion_depth.xlsx", engine='xlsxwriter')

    for slope_name in slope_names:
        print("========================", slope_name, "========================")
        # 坡面的各个阶段的地形
        slope_ply_paths = get_slopes_ply_by_name(slope_name)

        # 读取坡面各个阶段的侵蚀dem，并且保存
        # read_eroded_dem_and_save(slope_name)

        df = pd.DataFrame()

        path_slope_base = slope_ply_paths[0]
        for i in range(1, len(slope_ply_paths)):
            path_slope_target = slope_ply_paths[i]
            # print(path_slope_target)

            # 计算侵蚀前和侵蚀后的最低高程
            y_plot_before, erosion_depth = get_slope_min_elevation(path_slope_base, path_slope_target)

            if 'Distance' not in df.columns:
                # 将结果添加到DataFrame中
                df['Distance'] = y_plot_before

                # 如果 'Round i' 列不存在或者长度不匹配，那么就更新这一列
            if f'Round {i}' not in df.columns or len(df[f'Round {i}']) != len(erosion_depth):
                erosion_depth = np.pad(erosion_depth, (0, len(df) - len(erosion_depth)), 'constant',
                                       constant_values=0)  # 使用0填充缺失的值
                df[f'Round {i}'] = erosion_depth

            # df[f'Round {i}'] = erosion_depth

            # 将DataFrame写入Excel文件
            df.to_excel(writer, sheet_name=slope_name, index=False)

            print('处理坡面', slope_name, i, "完成")

    # 保存Excel文件
    writer.save()


def get_all_slopes_width():
    # slope_name = 'D1'

    # 创建一个Excel写入器
    writer = pd.ExcelWriter(r"outputs/erosion_width.xlsx", engine='xlsxwriter')

    for slope_name in slope_names:
        print("========================", slope_name, "========================")
        # 坡面的各个阶段的地形
        slope_ply_paths = get_slopes_ply_by_name(slope_name)

        df = pd.DataFrame()

        path_slope_base = slope_ply_paths[0]
        for i in range(1, len(slope_ply_paths)):
            path_slope_target = slope_ply_paths[i]
            # print(path_slope_target)

            # 计算侵蚀前和侵蚀后的最低高程
            y_plot_before, erosion_depth = get_slope_min_elevation(path_slope_base, path_slope_target)

            if 'Distance' not in df.columns:
                # 将结果添加到DataFrame中
                df['Distance'] = y_plot_before

                # 如果 'Round i' 列不存在或者长度不匹配，那么就更新这一列
            if f'Round {i}' not in df.columns or len(df[f'Round {i}']) != len(erosion_depth):
                erosion_depth = np.pad(erosion_depth, (0, len(df) - len(erosion_depth)), 'constant',
                                       constant_values=0)  # 使用0填充缺失的值
                df[f'Round {i}'] = erosion_depth

            # df[f'Round {i}'] = erosion_depth

            # 将DataFrame写入Excel文件
            df.to_excel(writer, sheet_name=slope_name, index=False)

            print('处理坡面', slope_name, i, "完成")

    # 保存Excel文件
    writer.save()


def get_all_slopes_dimension_fraction():
    threshold_depth = -0.5

    for slope_name in slope_names:
        print("========================", slope_name, "========================")
        # 坡面的各个阶段的地形
        slope_ply_paths = get_slopes_ply_by_name(slope_name)

        path_slope_base = slope_ply_paths[0]
        for i in range(1, len(slope_ply_paths)):
            # print(i)

            path_target = slope_ply_paths[i]

            dem_diff = cal_volume_change(path_slope_base, path_target)

            # 这里假设高程差异的最大值是阈值
            fd = fractal_dimension(dem_diff, threshold=threshold_depth)

            print(slope_name, i, "分形维数:", fd)


if __name__ == "__main__":
    # main()

    # get_all_slopes_depth()

    # get_all_slopes_dimension_fraction()

    # test_slope_width()

    get_all_slopes_width()

    # test_slope_elevation()

    # path_base = r"E:\Users\Moyear\Desktop\3d\slopes\B1_0_20240102_A.ply"
    # path_target = r"E:\Users\Moyear\Desktop\3d\slopes\B1_4_20240102_A.ply"
    #
    # dem_diff = cal_volume_change(path_base, path_target)
    #

    # # 这里假设高程差异的最大值是阈值
    # fd = fractal_dimension(dem_diff, threshold=1)
    #
    # print(f"分形维数: {fd}")