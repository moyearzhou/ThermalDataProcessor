import raw_phtoto_play as play_util
import official_video_utils as official_util


def test_official_video():
    # 这个视频是微影热视app拍出来的视频
    official_video_path = r"res/R_20230601172402173.mp4"
    # 读取并播放官方的视频
    official_util.read_official_video(official_video_path)


if __name__ == '__main__':
    video_path = r"res/videos/20230724162347_7996.video"
    # play_util.play_video_series(video_path)
    play_util.export_video(video_path)
