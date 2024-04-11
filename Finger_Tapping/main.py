import argparse
import os
from typing import Optional

import demo_utils as demo_utils
import feature_utils as parser
import hand_tracking as tracker
import numpy as np


def main(
    video_dir: str = "finger_tapping/data",
    video_name: str = "PD_patient_trimmed.mov",
    tracking_path: Optional[str] = None,
):

    # if tracking_path is None:
    #     tracking_path = os.path.join(
    #         video_dir, "{}.npy".format(video_name.split(".")[0])
    #     )

    # """ Test 'tracker.track' """
    # tracking = tracker.track(video_dir, video_name)

    # np.save(tracking_path, tracking)

    # for testing purposes
    # tracking = np.load(tracking_path, allow_pickle=True)

    # """ Test 'parser.FeatureParser """
    # feature_object = parser.FeatureParser(tracking)
    # # feature_object.compute_angle("right")
    # feature_object.compute_angle("left")
    # feature_to_plot = "left_angle"

    """Test 'demo_utils.create_feature_gif'"""
    # gif_path = os.path.join(video_dir, 'feat.gif')
    # demo_utils.create_feature_gif(feature_object.features[feature_to_plot], gif_path, 256, 64)

    """ Test 'demo_utils.create_tracking_gif' """
    # gif_path = os.path.join(video_dir, 'track.gif')
    # demo_utils.create_tracking_gif(os.path.join(video_dir, video_name), feature_object.hand_landmarks_proto_list,
    #                                 gif_path, width=256)

    """ Test 'demo_utils.stack_gifs' """
    # demo_utils.stack_gifs([os.path.join(video_dir, 'track.gif'),
    #                         os.path.join(video_dir, 'feat.gif')],
    #                          os.path.join(hvideo_dir,  'ffmpeg_concat.gif'))

    """ Test 'demo_utils.save_track_feat' """
    # gif_path = os.path.join(video_dir, "pd_patient_trimmed.gif")
    # demo_utils.save_track_feat(
    #     video_dir,
    #     video_name,
    #     gif_path,
    #     feature_object.hand_landmarks_proto_list,
    #     feature_object.features[feature_to_plot],
    #     width=300
    #             )

    """ Test 'demo_utils.optimize_gif' """
    gif_path = os.path.join(video_dir, "pd_patient_trimmed.gif")
    demo_utils.optimize_gif(gif_path, "-O2", gif_path)


if __name__ == "__main__":
    main()
