# coding = utf-8
# @Time : 2024/2/23 22:35
# @Author : moyear
# @File : ply_roughness_cal.y
# @Software : PyCharm
import csv
import os

import open3d as o3d
import numpy as np


def cal_total_roughness(path_ply):
    # 读取PLY文件
    pcd = o3d.io.read_point_cloud(path_ply)

    # 将点云数据转换为numpy数组
    points = np.asarray(pcd.points)

    # 计算高程的标准差
    roughness = np.std(points[:, 2])
    return roughness


def cal_all_slopes_roughness():
    # dir_path = r"E:\Users\Moyear\Desktop\3d\slopes"
    dir_path = r"E:\Users\Moyear\Desktop\3d"

    results = []

    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if not file.endswith('.ply'):
                continue

            ply_file_path = os.path.join(root, file)

            # 获取文件名
            file_name = os.path.basename(ply_file_path)

            slope_name = ''
            stage = -1

            # 提取出第一个_前的字符和数字
            split_name = file_name.split('_')
            # print(split_name)
            if len(split_name) >= 2:
                slope_name = split_name[0]
                stage = split_name[1].split('.')[0]  # 去掉文件扩展名部分

                # print("坡面名称:", slope_name, "阶段:", stage)

            # 在这里可以对Ply文件进行处理，比如读取内容等
            print("==============正在计算粗糙度：:", ply_file_path, "==============")

            roughness = cal_total_roughness(ply_file_path)
            row_data = {
                "坡面名称": slope_name,
                "阶段": stage,
                "坡面粗糙度": roughness,
                "文件名": file_name,
            }

            results.append(row_data)

            print(slope_name, "Surface roughness: ", roughness)

    save_to_csv(results, r'./outputs/result_roughness.csv')


def save_to_csv(data, file_path):
    keys = data[0].keys()

    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)


if __name__ == '__main__':
    # path_ply = r"E:\Users\Moyear\Desktop\3d\slopes\A0_4_20240111_A.ply"
    # path_ply = r"E:\Users\Moyear\Desktop\3d"
    #
    # roughness = cal_total_roughness(path_ply)
    # print("Surface roughness: ", roughness)

    cal_all_slopes_roughness()


