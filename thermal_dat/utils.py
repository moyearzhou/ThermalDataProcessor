# coding = utf-8
# @Time : 2023/12/10 16:00
# @Author : moyear
# @File : utils.y
# @Software : PyCharm
import os


def count_files(folder_path):
    count = 0

    # 遍历文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        # 累加文件个数
        count += len(files)

    return count