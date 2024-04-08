import os

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from mediapipe import solutions

ACC_FRAMES = []
MARGIN = 10


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


def plot_feature(feature: np.ndarray, figsize: tuple):
    """
    Plots feature trajectory in parallel to the video.
    Return the plot.
    """
    dpi = 100
    fig = plt.figure(figsize=(figsize[1] / dpi, 5), dpi=dpi)

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


def show_track(
    video_dir: str,
    video_name: str,
    landmarks_list: list,
    feature_list: list,
    width: int = 200,
    height: int = 200,
    show: bool = True,
):
    """
    Shows and saves the results of the hand traking in a concatenated format.
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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    i = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Failure to read frame")
            break

        frame = cv2.flip(frame, 1)

        landmarks = landmarks_list[i]
        feature = feature_list[: max(i, 1)]

        ACC_FRAMES.append(concat_frame_trajectory(frame, landmarks, feature))

        if show:
            cv2.imshow("Hand tracker", ACC_FRAMES[-1])

        i += 1

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    print("Saving the gif")

    out_path = os.path.join(
        "finger_tapping", "data", "concat_{}.gif".format(video_name.split(".")[0])
    )
    with imageio.get_writer(out_path, mode="I") as writer:
        for frame in ACC_FRAMES:
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    print("Results saved at")
    print(out_path)
