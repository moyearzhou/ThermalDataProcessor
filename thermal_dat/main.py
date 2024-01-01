import glob
import os

import raw_phtoto_play as play_util
import official_video_utils as official_util
from thermal_dat import raw_video_convert
from thermal_dat.test.raw_file_convert import modify_zip_structure


def test_official_video():
    # 这个视频是微影热视app拍出来的视频
    official_video_path = r"res/R_20230601172402173.mp4"
    # 读取并播放官方的视频
    official_util.read_official_video(official_video_path)


def mult_convert():
    folder_path = r"E:\Moyear\文档\冲刷实验\20231226_第二次冲刷实验\1 热红外成像"  # 文件夹路径
    extension = '*.raws'  # 文件后缀名

    # 使用 glob 模块获取所有符合条件的文件路径
    file_paths = glob.glob(os.path.join(folder_path, extension))

    # 遍历文件路径列表
    for file_path in file_paths:
        # 处理每个文件
        print("正在转换mp4：{0}".format(file_path))
        raw_video_convert.export_video(file_path)


def convert_zip_files():
    # path = r"E:\Users\Moyear\Desktop\jlll\20231226175205_9087.video.zip"

    folder_path = r"E:\Users\Moyear\Desktop\jlll"  # 文件夹路径
    extension = '*.zip'  # 文件后缀名

    # 使用 glob 模块获取所有符合条件的文件路径
    file_paths = glob.glob(os.path.join(folder_path, extension))

    # 遍历文件路径列表
    for file_path in file_paths:
        # 处理每个文件
        print("尝试转换视频文件：{0}".format(file_path))
        modify_zip_structure(file_path)

    # modify_zip_structure(path)


if __name__ == '__main__':

    # # ====================未压缩的raw视频播放======================
    # photo_seq_path = r"res/videos/20231211160928_1773.video"
    # play_util.play_video_series(photo_seq_path)

    # ====================raw视频播放======================
    # zip_video_path = r"res/videos/20231208175018_9120.raws"
    # zip_video_path = r"res/videos/20231211160928_1773.raws"
    # zip_video_path = r"E:\Moyear\文档\冲刷实验\20231222_第一次冲刷实验\1 热红外成像\原始数据\20231222155646_5754.raws"
    # zip_video_path = r"E:\Moyear\文档\冲刷实验\20231226_第二次冲刷实验\1 热红外成像\20231226161512_5643.raws"
    # play_util.play_raw_series(zip_video_path)

    video_path = r"E:\Moyear\Dev\PycharmProjects\shallow_flow\thermal_dat\res\videos\20231230213456_7395.video"
    # play_util.play_raw_series_in_folder(video_path)
    raw_video_convert.export_video_in_folder(video_path)

    # ====================raw视频转换为mp4======================
    # raw_video_convert.export_video(zip_video_path)

    # play_util.open_stream_file()

    # mult_convert()

    # ====================图片透视处理======================
    # image_path = r"res/pic_3.png"
    # show_and_transform(image_path)

    # convert_zip_files()


