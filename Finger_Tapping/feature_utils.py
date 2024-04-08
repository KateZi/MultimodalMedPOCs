from typing import Union

import numpy as np
from mediapipe.framework.formats import landmark_pb2


class FeatureParser:
    """
    Class to easily load and parse tracking results.
    Optionally, computes features.
    """

    def __init__(self, res_path: str, landmarks: list = [0, 4, 8]):
        """
        Keyword arguments:
        res_path -- path to the saved tracking results
        landmarks -- indeced of landmarks used for computing features
        """
        self.tracking = np.load(res_path, allow_pickle=True)
        self.landmarks = landmarks
        self.features = {}
        self.parse_results()
        self.normalize_landmarks()

    def parse_results(self):
        self.hand_landmarks_list = [ele.hand_landmarks for ele in self.tracking]
        self.handedness_list = [ele.handedness for ele in self.tracking]
        self.num_preds = len(self.hand_landmarks_list)

    def normalize_landmarks(self):
        self.hand_landmarks_proto_list = []
        for i in range(self.num_preds):
            landmarks_proto = []
            for hand_landmarks in self.hand_landmarks_list[i]:
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend(
                    [
                        landmark_pb2.NormalizedLandmark(
                            x=landmark.x, y=landmark.y, z=landmark.z
                        )
                        for landmark in hand_landmarks
                    ]
                )
                landmarks_proto.append(hand_landmarks_proto)
            self.hand_landmarks_proto_list.append(landmarks_proto)

    def compute_angle(self, hand: str):
        """
        Computes angle featue for all the frames.
        Stores in the object's features dictionary
        """
        angle = []
        for i in range(self.num_preds):
            angle.append(self._compute_angle(i, hand))

        self.features[f"{hand}_angle"] = interpol(angle)

    def _compute_angle(self, idx: int, hand: str):
        for pred_i in range(len(self.handedness_list[idx])):
            handedness = self.handedness_list[idx][pred_i]

            if handedness[0].display_name.lower() != hand.lower():
                continue

            coors = self.hand_landmarks_proto_list[idx][pred_i]
            coors = [ele for e, ele in enumerate(coors.landmark) if e in self.landmarks]
            v1 = ((coors[1].x - coors[0].x), (coors[1].y - coors[0].y))
            v2 = ((coors[2].x - coors[0].x), (coors[2].y - coors[0].y))
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            cosx = dot / (
                np.sqrt((v1[0] ** 2) + (v1[1] ** 2))
                * np.sqrt((v2[0] ** 2) + (v2[1] ** 2))
            )
            cosx = np.minimum(cosx, 1.0)
            angle = (np.cos(cosx) * 180) / np.pi

            return angle

        return np.nan


# Helper function to return indexes of nans
def nan_helper(y: np.ndarray):
    """
    Helper returning indeces of Nans
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


# Interpolates all nan values of given array
def interpol(arr: Union[list, np.ndarray]):
    """
    Custom interplation function
    Interpolates only over Nans
    """

    y = np.transpose(arr)

    nans, x = nan_helper(y)
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])

    arr = np.transpose(y)

    return arr


def flatten_list(to_flatten: list):
    """
    Custom flatten akin to numpy but for lists
    """
    while len(to_flatten) == 1 and isinstance(to_flatten[0], list):
        to_flatten = to_flatten[0]
    return to_flatten
