from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["svg.fonttype"] = "none"


def prepare_features(dfs_dict: dict, features: list):
    acc_feat = {str(feature): [] for feature in features}
    weights = {str(feature): [] for feature in features}

    for df in dfs_dict.values():
        for feature in features:
            f = df[feature[0]]
            if len(feature) > 1:
                f = f[feature[1]]
            if isinstance(f, list):
                f = np.hstack(f)
            acc_feat[str(feature)].append(f)
            weights[str(feature)].append(np.ones(len(f)) / len(f))

    return acc_feat, weights


def plot_hist_stat(arr: np.ndarray, ax: plt.Axes, label: str = "", title: str = ""):
    """Plots histogram and stats (mean, std) of the given data"""
    ax.hist(arr, density=True)
    ax.set_title(title)
    avg = np.mean(arr)
    var = np.std(arr)
    ax.axvline(avg, color="green")
    ax.axvline(avg - var, color="green", linestyle="--")
    ax.axvline(avg + var, color="green", linestyle="--")


def plot_compare_hists(
    feats_dict: dict,
    weights_dict: Optional[dict] = None,
    label: list = [],
    xlabel: Optional[list] = None,
    nbins: int = 10,
):
    """Plots multiple histograms for comparison.
    The dictionary should contain {<data_id>: {<feature[0]>: <DataFrame[feature[1]]>}}
    feature[2] - xlabel unit
    """
    fig, ax = plt.subplots(figsize=(8, 3), ncols=len(feats_dict.keys()), squeeze=False)

    for icol, f_name in enumerate(feats_dict.keys()):
        ax[0, icol].set_title(f_name)
        if weights_dict is not None:
            weights = weights_dict[f_name]
        else:
            weights = [np.ones(f.shape[0]) / f.shape[0] for f in feats_dict[f_name]]
        ax[0, icol].hist(feats_dict[f_name], nbins, weights=weights, label=label)
        if xlabel is not None:
            ax[0, icol].set_xlabel(xlabel[icol])

    fig.tight_layout(rect=[0, 0.1, 1, 0.8])
    plt.legend()


def plot_compare_box(
    feats_dict: dict, xlabels: Optional[list] = None, ylabel: Optional[list] = None
):
    """Plots multiple boxplots for comparison.
    The dictionary should contain {<data_id>: {<feature[0]>: <DataFrame[feature[1]]>}}
    feature[2] - xlabel unit
    """

    fig, ax = plt.subplots(figsize=(8, 3), ncols=len(feats_dict.keys()), squeeze=False)

    for icol, f_name in enumerate(feats_dict.keys()):
        ax[0, icol].set_title(f_name)
        feat = feats_dict[f_name].copy()
        for i, f in enumerate(feat):
            print(
                "Removed {}/{} nan values from the data".format(
                    sum(np.isnan(f)), f.shape[0]
                )
            )
            feat[i] = f[~np.isnan(f)]
        ax[0, icol].boxplot(feat, showfliers=False)
        if xlabels is not None:
            ax[0, icol].set_xticklabels(xlabels)
        if ylabel is not None:
            ax[0, icol].set_ylabel(ylabel[icol])

    fig.tight_layout(rect=[0, 0.1, 1, 0.8])


def plot_trajectory(
    trajectory: np.ndarray,
    idx: int,
    ax: plt.Axes,
    polym: Optional[np.ndarray] = None,
    text_flag: bool = True,
):
    """Plots trajectory (swipes) and its polynom (if present)"""
    f_trajectory = trajectory.copy()
    f_trajectory[:, 1] *= -1

    x, y = f_trajectory[0][0], f_trajectory[0][1]
    ax.scatter(x, y, color="green")
    if text_flag:
        ax.text(x, y, idx)

    x, y = f_trajectory[-1][0], f_trajectory[-1][1]
    ax.scatter(x, y, color="red")

    ax.plot(f_trajectory[:, 0], f_trajectory[:, 1], color="blue")

    if polym is not None:
        smoothness = np.sqrt(np.sum(np.square(polym - trajectory[:, 1])))
        smoothness = np.std(np.square(polym - trajectory[:, 1]))
        if text_flag:
            ax.text(x, y, f"{smoothness:.3f}")
        ax.plot(f_trajectory[:, 0], -polym, color="orange")


def plot_compare_trajectories(
    trajectories: dict,
    num: int = 10,
    polyms: Optional[dict] = None,
    text_flag: bool = True,
    titles: Optional[list] = None,
    exclude: Optional[dict] = None,
    ax: Optional[plt.axes] = None,
):
    """Plots trajectories and their polynoms (if present) for comparison"""
    if ax is None:
        _, ax = plt.subplots(ncols=len(trajectories), sharex=True, sharey=True)
    polym = None
    if exclude is None:
        exclude = {key: [] for key in trajectories.keys()}
    for icol, (key, ele) in enumerate(trajectories.items()):
        for e, trajectory in enumerate(ele):
            if e in exclude[key]:
                continue
            if e >= num:
                break
            if polyms is not None:
                polym = polyms[key][e]
            plot_trajectory(trajectory, e, ax[icol], polym, text_flag)
        if titles is not None:
            ax[icol].set_title(titles[icol])

    plt.tight_layout()

    return ax
