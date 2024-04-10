# from pygifsicle import optimize

import math
import os
from typing import Optional

import cv2
import ffmpeg
import imageio
import imutils
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mediapipe import solutions

MARGIN = 10


def stack_gifs(gifs: list, out_path: str, fps: Optional[int] = 30):
    """
    Vertically stacks a list of input sources.
    """
    inputs = [ffmpeg.input(gif) for gif in gifs]
    # fps_list = [ffmpeg.probe(gif)['streams'][0]['r_frame_rate'] for gif in gifs]
    # max_fps = max([float(ifps.split('/')[0])/float(ifps.split('/')[1]) for ifps in fps_list])
    # max_fps = int(math.ceil(max(fps, max_fps)))
    # inputs = [ffmpeg.filter(input, 'fps', fps=max_fps) for input in inputs]
    ffmpeg.filter(inputs, "vstack").output(out_path).run()
    # print(f'The stacked gif is saved with fps {max_fps}')
    print(f"The stacked gif is saved with fps {fps}")


def create_tracking_gif(
    video_path: str, landmarks_list: list, out_path: str, width: int, fps: int = 30
):
    """
    Generates and saves gif of overlayed tracking.
    """
    cap = cv2.VideoCapture(video_path)
    total_frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    out = imageio.get_writer(out_path, mode="I", fps=fps)

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            print(f"Failed to capture frame")
            break

        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if frame_num % 60 == 0:
            print(f"Processed {frame_num} / {total_frame_num}")

        annotated_frame = cv2.flip(frame, 1)
        landmarks = landmarks_list[frame_num]

        # Loop through the detected hands to visualize.
        # find a way to detect the main hand and annotate only it
        for idx in range(len(landmarks)):

            solutions.drawing_utils.draw_landmarks(
                annotated_frame,
                landmarks[idx],
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style(),
            )

        annotated_frame = imutils.resize(annotated_frame, width=width)
        out.append_data(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

    out.close()
    cap.release()
    cv2.destroyAllWindows()

    print(f"Gif saved at {out_path}")


def create_feature_gif(
    feature: np.ndarray,
    out_path: str,
    width: int,
    height: int,
    px: Optional[int] = None,
    fps: Optional[int] = 30,
):
    """
    Generates and saves a gif of feature trajectory
    """
    if px is None:
        px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
    fig, ax = plt.subplots(figsize=(width * px, height * px))
    xdata, ydata = [], []
    (ln,) = ax.plot([], [], color="blue")
    sc = ax.scatter([], [], color="red")

    def init():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, len(feature))
        ax.set_ylim(min(feature) - 1, max(feature) + 1)
        return (ln,)

    def update(frame):
        xdata.append(frame)
        ydata.append(feature[frame])
        ln.set_data(xdata, ydata)
        sc.set_offsets(np.stack([frame, feature[frame]]).T)
        return ((ln, sc),)
        # return ln,

    ani = FuncAnimation(fig, update, frames=len(feature), init_func=init, interval=30)

    ani.save(out_path, writer="ffmpeg", fps=fps, dpi=1 / px)
    print(f"Gif is saved at {out_path}")


def draw_landmarks_on_image(frame: np.ndarray, landmarks: list):
    """
    Draws hand landmarks on top of a frame.
    Returns the overlayed frame.
    """
    annotated_frame = np.copy(frame)

    # Loop through the detected hands to visualize.
    # find a way to detect the main hand and annotate only it
    for idx in range(len(landmarks)):

        solutions.drawing_utils.draw_landmarks(
            annotated_frame,
            landmarks[idx],
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style(),
        )

        # # Get the top left corner of the detected hand's bounding box.
        # height, width, _ = annotated_frame.shape
        # x_coordinates = [landmark.x for landmark in landmarks[idx]]
        # y_coordinates = [landmark.y for landmark in landmarks[idx]]
        # text_x = int(min(x_coordinates) * width)
        # text_y = int(min(y_coordinates) * height) - MARGIN

        # # Draw handedness (left or right hand) on the image.
        # cv2.putText(
        #     annotated_frame,
        #     f"{handedness[0].category_name}",
        #     (text_x, text_y),
        #     fontFace=cv2.FONT_HERSHEY_DUPLEX,
        #     fontScale=1,
        #     color=(0, 0, 0),
        # )

    return annotated_frame


def plot_feature(feature: np.ndarray, figsize: tuple, px: Optional[int] = None):
    """
    Plots feature trajectory in parallel to the video.
    Returns the plot.
    """
    if px is None:
        px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
    fig = plt.figure(figsize=(figsize[1] * px, figsize[0] * px))

    plt.plot(feature)
    plt.scatter(len(feature) - 1, feature[-1], color="red")

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


def concat_frame_trajectory(frame: np.ndarray, landmarks: list, feature: np.ndarray):
    """
    Vertically concatenates the annotated frame and trajectory plot.
    """
    if len(landmarks) == 0:
        top_frame = frame
    else:
        top_frame = draw_landmarks_on_image(frame, landmarks)

    bottom_frame = plot_feature(feature, top_frame.shape)

    vframe = cv2.vconcat([top_frame, bottom_frame])

    return vframe


def save_track_feat(
    video_dir: str,
    video_name: str,
    out_path: str,
    landmarks_list: list,
    feature_list: list,
    width: int = 200,
    fps: int = 30,
):
    """
    Shows the results of the hand traking in a concatenated format.
    Hand landmarks overlay and provided feature trajectory.

    Keyword arguments:
    video_dir       -- directory containing the video
    video_name      -- name of the video
    landmarks_list  -- normalized landmarks used for hand overlay
    feature_list    -- feature to plot
    width           -- width of the reshaped frame
    height          -- height of the reshaped frame
    show            -- whether to show results or only save
    """
    if not os.path.exists(os.path.join(video_dir, video_name)):
        print("Provide a valid video path")
        return

    cap = cv2.VideoCapture(os.path.join(video_dir, video_name))
    total_frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    out = imageio.get_writer(out_path, mode="I", fps=fps)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Failure to read frame")
            break

        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if frame_num % 60 == 0:
            print(f"Processed {frame_num} / {total_frame_num}")

        landmarks = landmarks_list[frame_num]
        feature = feature_list[: max(frame_num, 1)]

        frame = cv2.flip(frame, 1)
        res = concat_frame_trajectory(frame, landmarks, feature)
        res = imutils.resize(res, width=width)
        out.append_data(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))

    out.close()

    cap.release()
    cv2.destroyAllWindows()

    print(f"Gif saved at {out_path}.")
