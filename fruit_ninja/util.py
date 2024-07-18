import os
from typing import Optional

import numpy as np
import pandas as pd


def convert_time_stamps(df: pd.DataFrame):
    """Ensures the timestamps are in the correct format
    (changes the provided DataFrmae)
    """
    df["Timestamp"] = df["Timestamp"].str.replace(
        r"(.*)_(.*):(\d\d\d)$", r"\1 \2.\3", regex=True
    )
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y-%m-%d %H:%M:%S.%f")


def compute_game_id(df: pd.DataFrame):
    """Computes whether there are more than 1 round of a game
    and assigns ids in a linear manner
    (changes the provided DataFrmae)
    """
    game_id = 0
    game_ids = np.zeros(len(df), dtype=int)
    game_starts = df.fillna("").query("Details.str.contains('game starts')").index
    for game_start in game_starts:
        game_ids[int(game_start) :] = game_id
        game_id += 1
    df["Game_id"] = game_ids


def compute_scroll_dist(scroll: str):
    """Parses and computes distance traveled during the scroll
    using distance over x and y axes.
    Outputs distances in x and y directions
    and the calculated distance in a simple numeric format
    """
    distance_x = float(scroll.split("X: ")[1].split(" ")[0])
    distance_y = float(scroll.split("Y: ")[1].split(" ")[0])
    distance = np.sqrt(distance_x * distance_x + distance_y * distance_y)
    direction_x = np.sign(distance_x).astype(int)
    direction_y = np.sign(distance_y).astype(int)
    return np.abs(distance_x), np.abs(distance_y), distance, direction_x, direction_y


def compute_speed_dir(velocity_x: np.ndarray, velocity_y: np.ndarray):
    """Computes speed and direction
    Outputs speed and direction
    in a simple numeric format
    """
    speed = np.sqrt(velocity_x * velocity_x + velocity_y * velocity_y)
    add_deg = 0
    if velocity_x < 0:
        add_deg = 180 if velocity_y >= 0 else 270
    elif velocity_y <= 0:
        add_deg = 360
    direction = np.arctan(velocity_y / velocity_x)
    direction = np.abs(np.abs(direction * 180 / np.pi) - add_deg)
    return speed, direction


def compute_speed_dir_vec(veloctities: np.ndarray):
    return np.array(
        list(map(compute_speed_dir, veloctities[:, 0], veloctities[:, 1]))
    ).T


def compute_vel_vec(coors: np.ndarray, time_deltas: Optional[np.ndarray] = None):
    if time_deltas is None:
        time_deltas = np.ones(coors.shape[0] - 1)
    return np.diff(coors, axis=0) / time_deltas[:, None]


def compute_time_deltas(df: pd.DataFrame):
    return np.diff(df["Timestamp"]) / np.timedelta64(1, "s")


def compute_fling_vel(fling: str):
    """Parses and computes fling's speed and direction
    Outputs velocities on x and y axes, speed and direction of the fling
    in a simple numeric format
    """
    velocity_x = float(fling.split("X: ")[1].split(" ")[0])
    velocity_y = float(fling.split("Y: ")[1].split(" ")[0])
    speed, direction = compute_speed_dir(velocity_x, velocity_y)
    return velocity_x, velocity_y, speed, direction


def compute_fn_response_time(df: pd.DataFrame):
    """Computes a simplistic fruit ninja response time -
    time difference between when the fruit is cut and spawn
    """
    df["Fruit_id"] = df["Details"].str.extract(r"(\d+)$").astype(int)

    spawn_df = df[df["Details"].str.contains("New")].copy()
    cut_df = df[df["Details"].str.contains("Cut")].copy()

    merge_df = pd.merge(
        spawn_df,
        cut_df,
        on=["Game_id", "Fruit_id"],
        suffixes=("_spawn", "_cut"),
        how="outer",
    )
    merge_df["Response_time"] = (
        merge_df["Timestamp_cut"] - merge_df["Timestamp_spawn"]
    ).dt.total_seconds()

    df = pd.merge(
        df,
        merge_df[["Game_id", "Fruit_id", "Response_time"]],
        on=["Game_id", "Fruit_id"],
    )

    return df


def process_actions(df):
    df[["x", "y"]] = df.apply(
        lambda row: (
            float(row["CoorX"].split(":")[1]),
            float(row["CoorY"].split(":")[1]),
        ),
        axis=1,
        result_type="expand",
    )
    df["Timedelta"] = np.concatenate((compute_time_deltas(df), np.zeros(1)))
    df["distance_x"] = np.concatenate((np.diff(df["x"]), [0]))
    df["distance_y"] = np.concatenate((np.diff(df["y"]), [0]))
    velocities = np.concatenate(
        (compute_vel_vec(df[["x", "y"]].values, df["Timedelta"].values[:-1]), [[0, 0]])
    )
    df["velocity_x"] = velocities[:, 0]
    df["velocity_y"] = velocities[:, 1]


def extract_data(paths: dict, wanted_game_ids: list):
    """Extract data and its features for each csv path in 'paths'
    - wanted_game_ids allow to select which rounds to consider
    (currently allows 1 round from each session)
    """

    res = {key: {} for key in paths.keys()}

    for i, (key, path) in enumerate(paths.items()):
        df = pd.read_csv(path, header=1)
        convert_time_stamps(df)
        compute_game_id(df)
        # df = df.query("Game_id == {}".format(wanted_game_ids[i]))
        df = df.query("Game_id in {}".format(wanted_game_ids[i]))

        touches = (
            df.query("Activity.str.contains('FruitNinja')").reset_index().fillna("")
        )
        res[key]["touches"] = touches

        fruits = df.fillna("").query("Details.str.contains('fruit')").reset_index()
        fruits = compute_fn_response_time(fruits)
        res[key]["fruits"] = fruits

        actions = touches.query("Details.str.contains('Action')")
        process_actions(actions)
        res[key]["actions"] = actions

        scrolls = touches.query("Details.str.contains('Scroll')")
        scrolls[
            ["distance_x", "distance_y", "distance", "direction_x", "direction_y"]
        ] = scrolls.apply(
            lambda row: compute_scroll_dist(row["Details"]),
            axis=1,
            result_type="expand",
        )
        res[key]["scrolls"] = scrolls

        flings = touches.query("Details.str.contains('Fling')")
        flings[["velocity_x", "velocity_y", "speed", "direction"]] = flings.apply(
            lambda row: compute_fling_vel(row["Details"]), axis=1, result_type="expand"
        )
        res[key]["flings"] = flings

        # actions_down = touches.query("Details.str.contains('DOWN')")
        # moves = touches.query("Details.str.contains('MOVE')")
        # actions_up = touches.query("Details.str.contains('UP')")

    return res


def collect_trajectories(dfs_dict: dict, poly_deg=5):
    """Collect trajectories (swipes) from the extracted data
    and fits a polynom of degree 'poly_deg' for smoothness measure
     - poly_deg - needs to be balanced to not overfit the polynoms,
     but to reasonbly fit the data
    """
    trajectories = {key: [] for key in dfs_dict.keys()}
    polyms = {key: [] for key in dfs_dict.keys()}
    times = {key: [] for key in dfs_dict.keys()}

    for key, df in dfs_dict.items():
        df = df["actions"]
        for _, action in df.iterrows():
            if "DOWN" in action["Details"]:
                down_index = action["index"]
            elif ("UP" in action["Details"]) and (action["index"] > down_index):
                up_index = action["index"]

                trajectory = df.query("@down_index < index < @up_index")[
                    ["x", "y"]
                ].values
                if len(trajectory) < 1:
                    # arbitraty high value to ensure the condition is not met
                    # until the next down action
                    down_index = 9999
                    continue

                trajectories[key].append(np.array(trajectory))

                z = np.polyfit(trajectory[:, 0], trajectory[:, 1], poly_deg)
                p = np.poly1d(z)
                polyms[key].append(p(trajectory[:, 0]))

                # m, c = np.linalg.lstsq(np.vstack([trajectory[:,0], np.ones(len(trajectory[:,0]))]).T, trajectory[:,1], rcond=None)[0]
                # polyms[key].append(trajectory[:,0]*m + c)

                times[key].append(
                    df.query("@down_index < index < @up_index")["Timestamp"].values
                )

                # arbitraty high value to ensure the condition is not met
                # until the next down action
                down_index = 9999

    return trajectories, polyms, times
