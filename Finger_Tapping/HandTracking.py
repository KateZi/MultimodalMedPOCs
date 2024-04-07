from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python as pthn
from mediapipe.tasks.python import vision
from tqdm import tqdm

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging

logging.basicConfig(filename="myapp.log", level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "Finger_Tapping/model/hand_landmarker.task"
MARGIN = 10


def draw_landmarks_on_image(frame, detection_result, hand, landmarks=np.arange(21)):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_frame = np.copy(frame)

    # Loop through the detected hands to visualize.
    # find a way to detect the main hand and annotate only it
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        if handedness[0].display_name.lower() != hand.lower():
            continue

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for i, landmark in enumerate(hand_landmarks)
                if i in landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_frame,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style(),
        )

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_frame.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(
            annotated_frame,
            f"{handedness[0].category_name}",
            (text_x, text_y),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1,
            color=(0, 0, 0),
        )

    return annotated_frame


def plot_angle(detection_result, figsize, hand, landmarks=[0, 4, 8]):
    """
    Apex of the angle comes first
    """
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness

    dpi = 100
    fig = plt.figure(figsize=(figsize[1] / 100, 3), dpi=dpi)

    for idx in range(len(hand_landmarks_list)):
        handedness = handedness_list[idx]

        if handedness[0].display_name.lower() != hand.lower():
            continue

        coors = hand_landmarks_list[idx]
        coors = [ele for e, ele in enumerate(coors) if e in landmarks]
        v1 = ((coors[1].x - coors[0].x), (coors[1].y - coors[0].y))
        v2 = ((coors[2].x - coors[0].x), (coors[2].y - coors[0].y))
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        cosx = dot / (
            np.sqrt((v1[0] ** 2) + (v1[1] ** 2)) * np.sqrt((v2[0] ** 2) + (v2[1] ** 2))
        )
        cosx = np.minimum(cosx, 1.0)
        angle = (np.cos(cosx) * 180) / np.pi
        ACC.append(angle)

        plt.plot(ACC)
        plt.scatter(len(ACC) - 1, angle, color="red")

        plt.yticks([])
        plt.xticks([])

    # redraw the canvas
    fig.canvas.draw()

    # convert canvas to image
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    plt.close(fig)

    return img


def show_frame_trajectory(
    detection_result: vision.HandLandmarkerResult, frame, hand, landmarks=[0, 4, 8]
):

    if len(detection_result.hand_landmarks) == 0:
        top_frame = frame
    else:
        top_frame = draw_landmarks_on_image(frame, detection_result, hand)

    bottom_frame = plot_angle(detection_result, top_frame.shape, hand, landmarks)

    vframe = cv2.vconcat([top_frame, bottom_frame])

    cv2.imshow("Hand tracker", vframe)


def track(video_dir: str, video_name: str, out_path=None):

    if not os.path.exists(os.path.join(video_dir, video_name)):
        print("Provide a valid video path")
        return

    if not out_path:
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
        num_hands=1,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    res = []
    i = 0
    pbar = tqdm(total=cap.get(cv2.CAP_PROP_FRAME_COUNT), leave=True, position=1)
    while tqdm(cap.isOpened()):
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


def show_track(
    video_dir: str,
    video_name: str,
    res_path: str,
    hand: str,
    features=["angle"],
    width=300,
    height=500,
):

    if not os.path.exists(os.path.join(video_dir, video_name)):
        print("Provide a valid video path")
        return

    if not os.path.exists(os.path.join(res_path)):
        print("Provide a valid tracking path")
        return

    cap = cv2.VideoCapture(os.path.join(video_dir, video_name))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    tracking_res = np.load(res_path, allow_pickle=True)
    i = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Failure to read frame")
            break

        frame = cv2.flip(frame, 1)

        res = tracking_res[i]

        show_frame_trajectory(res, frame, hand)

        i += 1

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
