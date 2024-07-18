import os
from typing import Callable

import demo_utils as demo_utils
import feature_utils as parser
import hand_tracking as tracker
import numpy as np


def track(video_dir: str, video_name: str, tracking_path: str, overwrite: bool = False):
    """Returns tracking results
    If tracking exists and overwrite is False - loads exsting results
    else runs tracking and saves
    """
    if tracking_path is None:
        tracking_path = os.path.join(
            video_dir, "{}.npy".format(video_name.split(".")[0])
        )
    if os.path.exists(tracking_path) and not overwrite:
        print("Tracking exists, loading results...")
        tracking = np.load(tracking_path, allow_pickle=True)
    else:
        tracking = tracker.track(video_dir, video_name)
        np.save(tracking_path, tracking)

    return tracking


def parse(tracking: list = None, tracking_path: str = None, hand: str = "left"):
    """Parses tracking and returns feature object"""
    if tracking is None:
        if tracking_path:
            tracking = np.load(tracking_path, allow_pickle=True)
        else:
            raise KeyError("Provide tracking results or a path to saved tracking")

    feature_object = parser.FeatureParser(tracking)
    feature_object.compute_angle(hand.lower())
    return feature_object


def plotting_pipeline(
    plot_func: Callable,
    plot_args: dict,
    track_args: dict = None,
    parse_args: dict = None,
    hand: str = "left",
):
    """Runs plotting function and tracking and/or parsing before if needed"""
    if track_args:
        tracking = track(**track_args)
        parse_args["tracking"] = tracking
    if parse_args:
        feature_object = parse(**parse_args)

    if plot_func == demo_utils.create_feature_gif:
        plot_args["feature_list"] = feature_object.features[f"{hand}_angle"]
        plot_func(**plot_args)
    if plot_func == demo_utils.create_tracking_gif:
        plot_args["landmarks_list"] = feature_object.hand_landmarks_proto_list
        plot_func(**plot_args)
    if plot_func == demo_utils.save_track_feat:
        plot_args["feature_list"] = feature_object.features[f"{hand}_angle"]
        plot_args["landmarks_list"] = feature_object.hand_landmarks_proto_list
        plot_func(**plot_args)
    else:
        plot_func(**plot_args)


if __name__ == "__main__":

    data_dir = "Finger_Tapping/data"
    prefix_name = "PD_patient"
    video_path = f"{prefix_name}.mov"
    out_path = os.path.join(data_dir, f"tracking_{prefix_name}.gif")
    tracking_path = os.path.join(data_dir, f"{prefix_name}.npy")

    func = demo_utils.save_track_feat
    # adjust to your needs
    plot_args = {
        "video_dir": data_dir,
        "video_name": video_path,
        "out_path": out_path,
        "width": 256,
    }
    track_args = {
        "video_dir": data_dir,
        "video_name": video_path,
        "tracking_path": tracking_path,
        "overwrite": False,
    }
    hand = "right"
    parse_args = {"tracking_path": tracking_path, "hand": hand}

    plotting_pipeline(func, plot_args, track_args, parse_args, hand)
