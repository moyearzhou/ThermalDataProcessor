import glob
import os

import raw_phtoto_play as play_util
import official_video_utils as official_util
from thermal_dat import raw_video_convert


def test_official_video():
    # 这个视频是微影热视app拍出来的视频
    official_video_path = r"res/R_20230601172402173.mp4"
    # 读取并播放官方的视频
    official_util.read_official_video(official_video_path)


def mult_convert():
    folder_path = r"E:\Moyear\文档\冲刷实验\20231222_第一次冲刷实验\1 热红外成像\原始数据"  # 文件夹路径
    extension = '*.raws'  # 文件后缀名

    # 使用 glob 模块获取所有符合条件的文件路径
    file_paths = glob.glob(os.path.join(folder_path, extension))

    # 遍历文件路径列表
    for file_path in file_paths:
        # 处理每个文件
        print("正在转换mp4：{0}".format(file_path))
        raw_video_convert.export_video(file_path)


if __name__ == '__main__':

    # # ====================未压缩的raw视频播放======================
    # photo_seq_path = r"res/videos/20231211160928_1773.video"
    # play_util.play_video_series(photo_seq_path)

    # ====================raw视频播放======================
    # zip_video_path = r"res/videos/20231208175018_9120.raws"
    # zip_video_path = r"res/videos/20231211160928_1773.raws"
    # zip_video_path = r"E:\Moyear\文档\冲刷实验\20231222_第一次冲刷实验\1 热红外成像\原始数据\20231222155646_5754.raws"
    zip_video_path = r"E:\Moyear\文档\冲刷实验\20231222_第一次冲刷实验\1 热红外成像\原始数据\20231222173645_2447.raws"
    play_util.play_raw_series(zip_video_path)

    # ====================raw视频转换为mp4======================
    # raw_video_convert.export_video(zip_video_path)

    # play_util.open_stream_file()

    # mult_convert()

    # ====================图片透视处理======================
    # image_path = r"res/pic_3.png"
    # show_and_transform(image_path)
