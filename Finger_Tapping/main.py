import argparse

import demo_utils as demo_utils
import feature_utils as parser
import hand_tracking as tracker


def main(video_dir: str = "finger_tapping/data", video_name: str = "test.mov"):
    res_path = tracker.track(video_dir, video_name)

    feature_object = parser.FeatureParser(res_path)

    feature_object.compute_angle("right")

    feature_to_plot = "right_angle"

    demo_utils.show_track(
        video_dir,
        video_name,
        feature_object.hand_landmarks_proto_list,
        feature_object.features[feature_to_plot],
    )


if __name__ == "__main__":
    main()
