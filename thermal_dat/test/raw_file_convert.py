# coding = utf-8
# @Time : 2023/12/25 16:12
# @Author : moyear
# @File : raw_file_convert.y
# @Software : PyCharm
import os.path
import re
import zipfile


def modify_zip_structure(zip_file_path):
    '''
    这个方法是将原来拍摄的没有压缩的文件夹格式的项目转换成zip格式的项目格式
    :param zip_file_path:
    :return:
    '''
    out_dir = r"E:\Users\Moyear\Desktop\jlll\output"
    # 检查 zip 文件是否存在
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # 获取ZIP文件名（不包含扩展名）
    zip_file_name = os.path.splitext(os.path.basename(zip_file_path))[0]

    target_file_name = zip_file_name.replace("video", "raws")

    # 最终文件的路径
    path_target_file = os.path.join(out_dir, target_file_name)

    print("正在转换文件：{0}".format(zip_file_path))

    # 创建一个压缩文件
    with zipfile.ZipFile(path_target_file, 'w', zipfile.ZIP_DEFLATED) as outZip:
        # 打开 ZIP 文件
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # 获取 ZIP 文件中的所有文件列表
            file_list = zip_ref.namelist()

            # 遍历文件列表
            for file_info in file_list:
                file_name = ""
                # 查找斜杠的索引
                slash_index = file_info.find("/")
                if slash_index != -1:
                    # 提取斜杠后面的内容
                    file_name = file_info[slash_index + 1:]

                if file_name == "":
                    continue

                file_bytes = zip_ref.read(file_info)

                out_file_info = file_name
                if re.fullmatch(r"\d{8}", file_name):
                    out_file_info = "raw/" + out_file_info

                print("\r正在将{0}写入到{1}的{2}".format(file_name, path_target_file, out_file_info), end="")

                # 将字节流写入到 zip 文件中
                outZip.writestr(out_file_info, file_bytes)


def convert_file_with_page(path_file):
    NUM_PAGING = 1000

    out_dir = "../output/convert"

    out_file_name = ""

    temp_path = os.path.join(out_dir, out_file_name)

    with zipfile.ZipFile(path_file, 'r') as zip_ref:
        # 获取 ZIP 文件中的所有文件列表
        file_list = zip_ref.namelist()

        # 遍历文件列表
        for file_info in file_list:
            # 仅处理 raw 文件夹下的文件
            if file_info.startswith('raw/'):
                file_name = file_info.replace("raw/", "")

                if not re.fullmatch(r"\d{8}", file_name):
                    continue

                cur_frame = int(file_name)

                cur_page = cur_frame / NUM_PAGING
                cur_pos = cur_frame % NUM_PAGING

                page_file = '{:08d}'.format(cur_page)

                print("正在往分页文件{0}中写入第{1}帧，写入位置为{2}".format(page_file, cur_frame, cur_pos))

                file_bytes = zip_ref.read(file_info)

                page_file_path = os.path.join()

            else:
                zip_ref.extract(file_info, temp_path)
