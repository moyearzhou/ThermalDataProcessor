import raw_phtoto_play as play_util
import official_video_utils as official_util
import raw_video_convert as video_convert
from thermal_dat.test.img_perspective_transform import show_and_transform


def test_official_video():
    # 这个视频是微影热视app拍出来的视频
    official_video_path = r"res/R_20230601172402173.mp4"
    # 读取并播放官方的视频
    official_util.read_official_video(official_video_path)


if __name__ == '__main__':

    # video_path = r"E:\Users\Moyear\Desktop\1111\20231204172459_3987.video"
    # # video_path = r"res/videos/20230724162347_7996.video"
    # play_util.play_video_series(video_path)
    # play_util.export_video(video_path)

    # ====================raw视频处理======================
    # zip_video_path = r"res/videos/20231208175018_9120.raws"
    # play_util.play_raw_series(zip_video_path)

    # video_convert.export_video(zip_video_path, True)

    # play_util.open_stream_file()

    # ====================图片透视处理======================
    image_path = r"res/pic_3.png"
    show_and_transform(image_path)
