import os
import subprocess
from typing import Optional

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

plt.rcParams["svg.fonttype"] = "none"


N_FFT = 512
HOP_LENGTH = 160
HARMONICS = np.arange(1, 11)
FMAX = 5000
FPS = 16_000


def animate_signal(
    signal: np.ndarray,
    out_path: str,
    width: int = 512,
    height: int = 512,
    px: Optional[int] = None,
    fps: int = FPS,
    chunk_size: int = 512,
):
    """
    Animate signal progression and save in out_path.
    Animation happens in chunks of data
    Choose chunk_size=1 for no chunking
    """

    indeces = np.arange(0, len(signal) - chunk_size, chunk_size)
    indeces = list(zip(indeces[:-1], indeces[1:]))

    # Initialize figure and axes for plotting
    if px is None:
        px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
    fig, ax = plt.subplots(figsize=(width * px, height * px))
    xdata, ydata = [], []
    ax.plot(np.arange(len(signal)), signal, color="blue", alpha=0.25)
    (ln,) = ax.plot([], [], color="blue")

    def init():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-1, len(signal) + 1)
        ax.set_ylim(min(signal) - 1, max(signal) + 1)
        return (ln,)

    def update(frame):
        xdata.append(np.arange(frame[0], frame[1]))
        ydata.append(signal[frame[0] : frame[1]])
        ln.set_data(xdata, ydata)
        return (ln,)

    chunk_fps = fps / chunk_size
    interval = 1 / (chunk_fps)
    ani = FuncAnimation(fig, update, frames=indeces, init_func=init, interval=interval)
    ani.save(out_path, writer="ffmpeg", fps=chunk_fps)
    print(f"Gif is saved at {out_path}")


def sync_viz_audio(viz_path, audio_path, sr=16_000):
    """
    Syncs video and audio and saves them as a single mp4
    Assumes the provided sr corresponds to the vizual
    """
    temp_audio_path = audio_path.split(".")[0] + "_resampled.mp3"
    subprocess.call(
        [
            "ffmpeg",
            "-y",
            "-i",
            audio_path,
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-c:a",
            "libmp3lame",
            "-q:a",
            "0",
            temp_audio_path,
        ]
    )
    out_path = viz_path.split(".")[0] + "_audio.mp4"
    subprocess.call(
        [
            "ffmpeg",
            "-y",
            "-i",
            viz_path,
            "-i",
            temp_audio_path,
            "-map",
            "0",
            "-map",
            "1:a",
            "-c:v",
            "copy",
            out_path,
        ]
    )
    os.remove(temp_audio_path)
    print(f"Sync is saved at {out_path}")


def anim_sync_audios(
    audio_paths: str, out_dir: str, sr: int = FPS, chunk_size: int = 512
):
    for audio_path in audio_paths.values():
        anim_path = audio_path.split(os.sep)[-1]
        anim_path = anim_path.split(".")[0] + "_anim.mp4"
        anim_path = os.path.join(out_dir, anim_path)
        signal, sr = librosa.load(audio_path, sr=sr)
        signal = (signal - signal.min()) / (signal.max() - signal.min())
        animate_signal(signal, anim_path, fps=sr, chunk_size=chunk_size)
        sync_viz_audio(anim_path, audio_path, sr)
    print("Saved all the audio playbacks")


def plot_f0(
    features: dict,
    y_axis: str = "log",
    harmonics: list = HARMONICS,
    ax: Optional[list] = None,
    harmonics_flag=True,
):
    """Plots the fundamental frequency overlaid over a spectrogram
    and if harmonics_flag==True, highlights the harmonics within
    the 'harmonics' range.
    """
    if ax is None:
        ncols = len(features)
        _, ax = plt.subplots(figsize=(12, 6), ncols=ncols, sharex=False)

    for i, (key, feature) in enumerate(features.items()):
        S = feature["S"]
        f0 = feature["f0"]
        sr = feature["sr"]
        n_fft = feature["n_fft"]
        hop_length = feature["hop_length"]
        librosa.display.specshow(
            librosa.amplitude_to_db(S, ref=np.max),
            sr=sr,
            y_axis=y_axis,
            x_axis="time",
            ax=ax[i],
            fmax=FMAX,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        times = librosa.times_like(S, sr=sr, hop_length=hop_length, n_fft=n_fft)
        times -= times[0]
        ax[i].plot(times, f0, linewidth=2, color="white", label="f0")
        if harmonics_flag:
            for h in np.arange(2, len(harmonics) + 1):
                ax[i].plot(times, h * f0, label=f"{h}*f0")
        ax[i].set_title(key)
        ax[i].set_xticks([])
        ax[i].set_xlabel("")
        if i != 0:
            ax[i].set_ylabel("")


def plot_transcripts(
    transcripts: dict,
    features: dict,
    ax: Optional[list] = None,
):
    """Plots transcription - words with timing;
    overlaud over the audio waveform"""
    if ax is None:
        ncols = len(transcripts)
        _, ax = plt.subplots(figsize=(12, 6), ncols=ncols)

    for i, key in enumerate(transcripts.keys()):
        words = transcripts[key]["words"]
        waveform = features[key]["waveform"]
        sr = features[key]["sr"]
        new_sr = 4_000
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=new_sr)
        duration = librosa.get_duration(y=waveform, sr=new_sr)
        times = np.linspace(0, duration, len(waveform))

        ax[i].plot(times, waveform)

        ticks = []
        for word in words:
            ax[i].axvline(word["start"], color="green", linestyle="dotted")  # *1000/sr)
            ax[i].axvline(word["end"], color="red", linestyle="dashed")  # *1000/sr)
            ticks.append((word["end"] - word["start"]) / 2 + word["start"])

        ax[i].set_xticks(ticks)
        ax[i].set_xticklabels(
            [word["word"] for word in words], rotation=30, ha="center"
        )

        ax[i].set_xlim(times[0], times[-1])
        ax[i].set_yticks([])
        if i == 0:
            ax[i].set_ylabel("Loudness")


def plot_harmonics(
    features: dict, harmonics_mult: list = HARMONICS, ax: Optional[list] = None
):
    """Plots harmonics energies through time"""
    if ax is None:
        ncols = len(features)
        _, ax = plt.subplots(figsize=(12, 6), ncols=ncols)
    for i, (i, feature) in enumerate(features.items()):
        harmonics = feature["harmonics_energy"]
        sr = feature["sr"]
        librosa.display.specshow(
            librosa.amplitude_to_db(harmonics, ref=np.max),
            sr=sr,
            x_axis="time",
            ax=ax[-1, i],
        )

        ax[i].set_yticks(harmonics_mult - 1)
        ax[i].set_yticklabels(harmonics_mult)
        ax[i].set(ylabel="Harmonics")


def plot_harmonics_transcription(
    features: dict,
    transcripts: Optional[list] = None,
    y_axis: str = "log",
    harmonics_flag=True,
    title: str = None,
    save_path: str = None,
):
    """Plots the spectragrams with f0 and harmonics overlay
    Optionally: plots Harmonics energy through time
                plots transcription through time
    """

    nrows = 2
    if harmonics_flag:
        nrows += 1
    fig, ax = plt.subplots(
        figsize=(10, 4), nrows=nrows, ncols=len(features), sharey="row"
    )

    plot_f0(features, y_axis, ax=ax[0], harmonics_flag=harmonics_flag)
    plot_transcripts(transcripts, features, ax=ax[1])

    if harmonics_flag:
        prol_harmonics(features, ax=ax[-1])

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
    plt.draw()

    if save_path is not None:
        fig.savefig(save_path)


def plot_radar(
    stats: pd.DataFrame, dropna: bool = True, title: str = None, save_path: str = None
):
    """Plots stats in a radar chart form"""
    if dropna:
        stats = stats.dropna(axis=1)
    stats = (100.0 * stats / stats.sum()).round(0)
    stats -= 30

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    # Direction of the zero angle to the north (upwards)
    ax.set_theta_zero_location("N")
    ax.set_yticklabels([])
    # Make radial gridlines appear behind other elements
    ax.spines["polar"].set_zorder(1)
    ax.spines["polar"].set_color("lightgrey")

    for idx in stats.index:
        labels = stats.columns
        values = stats.loc[idx, labels].values
        # values = np.log10(values)
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        labels = np.concatenate((labels, [labels[0]]))
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))

        ax.plot(angles, values, "o-", linewidth=2, label=idx)
        ax.fill(angles, values, alpha=0.25)

    ax.set_thetagrids(angles * 180 / np.pi, labels, rotation=30)
    ax.grid(True)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    if title is not None:
        fig.suptitle(title)

    plt.draw()

    if save_path is not None:
        fig.savefig(save_path)


def compare_waveforms(waves: list, ax: Optional[plt.Axes] = None):
    """Plots the waveforms for comparison"""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6), nrows=len(waves))
    for i, wave in enumerate(waves, 0):
        ax[i].plot(wave)

    plt.tight_layout()
    plt.draw()


def compare_spectras(
    spectras_list: list, y_axis: str = "log", ax: Optional[plt.Axes] = None
):
    """Plots the spectrograms for comparison"""
    if ax is None:
        nrows = len(spectras_list)
        ncols = 1
        _, ax = plt.subplots(figsize=(12, 6), nrows=nrows, ncols=ncols)

    for i, spectra in enumerate(spectras_list):
        im = librosa.display.specshow(
            librosa.amplitude_to_db(spectra, ref=np.max), ax=ax[i], y_axis=y_axis
        )
        plt.colorbar(im, ax=ax[i])

    plt.tight_layout()
    plt.draw()
