import argparse

import HandTracking as ht


def main(video_dir="Finger_Tapping/data", video_name="test.mov"):
    res_path = ht.track(video_dir, video_name)
    ht.show_track(video_dir, video_name, res_path, hand="right")


if __name__ == "__main__":
    main()
