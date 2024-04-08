import logging
import os
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as pthn
from mediapipe.tasks.python import vision
from tqdm import tqdm

logging.basicConfig(filename="myapp.log", level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "finger_tapping/model/hand_landmarker.task"
NUM_HANDS = 2


def track(video_dir: str, video_name: str, out_path: Optional[str] = None):
    """
    Tracks hand/s in video and saves the results at out_path.
    Returns out_path
    """
    if not os.path.exists(os.path.join(video_dir, video_name)):
        print("Provide a valid video path")
        return

    if out_path is None:
        out_path = os.path.join(
            "Finger_Tapping", "data", "{}.npy".format(video_name.split(".")[0])
        )

    cap = cv2.VideoCapture(os.path.join(video_dir, video_name))
    # do i need to explicitly set the width and height?

    base_options = pthn.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        # running_mode=vision.RunningMode.IMAGE,
        num_hands=NUM_HANDS,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    res = []
    i = 0
    pbar = tqdm(total=cap.get(cv2.CAP_PROP_FRAME_COUNT), leave=True, position=1)
    while cap.isOpened():
        pbar.update(i)
        ret, frame = cap.read()

        if not ret:
            logger.info(f"Failure to read a {cap.get(cv2.CAP_PROP_POS_FRAMES)} frame")
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

        i += 1

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

    np.save(out_path, res)

    return out_path
