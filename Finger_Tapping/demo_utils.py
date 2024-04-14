# from pygifsicle import optimize

import os
import subprocess
from typing import Optional

import cv2
import imageio
import imutils
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mediapipe import solutions


def stack_gifs(gifs: list, out_path: str):
    """
    Vertically stacks a list of input sources.
    """
    subprocess.call(
        ["ffmpeg", "-i", gifs[0], "-i", gifs[1], "-filter_complex", "vstack", out_path]
    )
    print(f"The stacked gif is saved")


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
    feature_list: np.ndarray,
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
        ax.set_xlim(0, len(feature_list))
        ax.set_ylim(min(feature_list) - 1, max(feature_list) + 1)
        return (ln,)

    def update(frame):
        xdata.append(frame)
        ydata.append(feature_list[frame])
        ln.set_data(xdata, ydata)
        sc.set_offsets(np.stack([frame, feature_list[frame]]).T)
        return ((ln, sc),)

    ani = FuncAnimation(
        fig, update, frames=len(feature_list), init_func=init, interval=30
    )

    ani.save(out_path, writer="ffmpeg", fps=fps, dpi=1 / px)
    print(f"Gif is saved at {out_path}")


def draw_landmarks_on_image(frame: np.ndarray, landmarks: list):
    """
    Draws hand landmarks on top of a frame.
    Returns the overlayed frame.
    """
    annotated_frame = np.copy(frame)

    landmark_drawing_spec = solutions.drawing_styles.get_default_hand_landmarks_style()
    connect_drawing_spec = solutions.drawing_styles.get_default_hand_connections_style()

    # Loop through the detected hands to visualize.
    # find a way to detect the main hand and annotate only it
    for idx in range(len(landmarks)):

        solutions.drawing_utils.draw_landmarks(
            annotated_frame,
            landmarks[idx],
            solutions.hands.HAND_CONNECTIONS,
            landmark_drawing_spec,
            connect_drawing_spec,
        )

    return annotated_frame


def plot_feature(
    feature_list: np.ndarray, index: int, figsize: tuple, px: Optional[int] = None
):
    """
    Plots feature trajectory in parallel to the video.
    Returns the plot.
    """
    if px is None:
        px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
    fig = plt.figure(figsize=(figsize[1] * px, figsize[0] * px))

    feature = feature_list[: max(index, 1)]
    plt.plot(feature, linewidth=3)
    plt.scatter(len(feature) - 1, feature[-1], color="red")

    plt.ylim(min(feature_list) - 1, max(feature_list) + 1)
    plt.xlim(0, len(feature_list))
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


def concat_frame_trajectory(
    frame: np.ndarray, landmarks_list: list, feature_list: np.ndarray, index: int
):
    """
    Vertically concatenates the annotated frame and trajectory plot.
    """
    landmarks = landmarks_list[index]

    if len(landmarks) == 0:
        top_frame = frame
    else:
        top_frame = draw_landmarks_on_image(frame, landmarks)

    bottom_frame = plot_feature(feature_list, index, top_frame.shape)

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

        frame = cv2.flip(frame, 1)
        res = concat_frame_trajectory(frame, landmarks_list, feature_list, frame_num)
        res = imutils.resize(res, width=width)
        out.append_data(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))

    out.close()

    cap.release()
    cv2.destroyAllWindows()

    print(f"Gif saved at {out_path}.")


def optimize_gif(in_path: str, level: str, out_path: Optional[str] = None):
    """
    Optimizes gif using gifsicle. Saves at out_path if any, or overwrites.

    Keyword arguments:
    in_path -- path to the gif to optimize
    level   -- one of the thre: ['-O1', '-O2', '-O3'],
                where 1 is least optimization and 3 - most
    """
    if out_path is None:
        out_path = in_path
    subprocess.run(["gifsicle", level, in_path, "-o", out_path])
    print("Gif is optimized")
