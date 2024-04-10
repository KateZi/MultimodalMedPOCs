import logging
import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import BaseOptions, vision

logging.basicConfig(filename="myapp.log", level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "finger_tapping/model/hand_landmarker.task"
NUM_HANDS = 2


def track(video_dir: str, video_name: str):
    """
    Tracks hand/s in video.
    Returns out_path
    """
    if not os.path.exists(os.path.join(video_dir, video_name)):
        raise FileNotFoundError("Please provide valid video path")

    cap = cv2.VideoCapture(os.path.join(video_dir, video_name))
    total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Starting to process {total_frame_num} frames")

    base_options = BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        # running_mode=vision.RunningMode.IMAGE,
        num_hands=NUM_HANDS,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    res = []

    while cap.isOpened():
        ret, frame = cap.read()

        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if frame_num % 60 == 0:
            print(f"Processed {frame_num} / {total_frame_num}")

        if not ret:
            print(f"Failure to read the {frame_num}th frame")
            break

        # tranform the frame into appropriate format
        # experiment whether you need to flip the image
        # no need for frontal camera
        mp_image = cv2.flip(frame, 1)
        mp_image = cv2.cvtColor(mp_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=mp_image)

        res.append(
            detector.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))
        )

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

    return res
