# coding = utf-8
# @Time : 2023/12/10 16:00
# @Author : moyear
# @File : utils.y
# @Software : PyCharm
import datetime
import json
import os
import zipfile


def count_files(folder_path):
    count = 0

    # 遍历文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        # 累加文件个数
        count += len(files)

    return count


def count_frame_in_raws(zip_file_path):
    # zip_file_path = "path/to/your/archive.zip"
    folder_name = "raw"

    count = 0
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()

        for file_name in file_list:
            if file_name.startswith(folder_name + "/"):
                count += 1
    # print(f"Total files in '{folder_name}': {count}")
    return count


def get_shoot_time(file_path):
    file_name = os.path.basename(file_path)
    datetime_str = file_name[:14]  # 提取前14位作为日期时间字符串

    # 将日期时间字符串转换为标准的时间格式
    datetime_obj = datetime.datetime.strptime(datetime_str, "%Y%m%d%H%M%S")
    return datetime_obj


def add_time(datetime_str, time_str):
    '''
    将一个时间值（例如2023-12-22 16:04:16）与另一个时间值（例如06:04:12）相加
    :param datetime_str:
    :param time_str:
    :return:
    '''
    # datetime_str = "2023-12-22 16:04:16"
    # time_str = "06:04:12"

    datetime_obj = datetime.datetime.strptime(str(datetime_str), "%Y-%m-%d %H:%M:%S")
    time_obj = datetime.datetime.strptime(time_str, "%H:%M:%S")

    result = datetime_obj + datetime.timedelta(hours=time_obj.hour, minutes=time_obj.minute, seconds=time_obj.second)
    return result


def get_name_without_extension(path):
    file_name = os.path.basename(path)
    # 使用os.path.splitext()获取文件名和扩展名的元组
    name_without_extension = os.path.splitext(file_name)[0]
    # print(name_without_extension)
    return name_without_extension


def get_total_frames(video_path):
    config_path = os.path.join(video_path, "config.json")
    total_frames = -1
    with open(config_path, 'r') as file:
        data = json.load(file)
        total_frames = data["totalFrames"]
    return total_frames
