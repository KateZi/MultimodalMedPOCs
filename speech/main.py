import os
import subprocess
from typing import Optional

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io.wavfile import write

SAMPLE_RATE = 22050


def filter_audios(
    audio_paths: list, low: int = 200, high: int = 3000, overwrite: bool = False
):
    """Filters and saves audios using FFMPEG
    if low is None - performs only highpass
    if high is None - only lowpass
    else - bandpass
    """
    filtered_paths = []
    for audio_path in audio_paths:
        filtered_path = os.path.join(
            audio_path.split(os.sep)[0],
            "out",
            "test",
            f"{audio_path.split(os.sep)[-1].split('.')[0]}_{low}_{high}.wav",
        )
        filtered_paths.append(filtered_path)
        if overwrite or not os.path.exists(filtered_path):
            ffmpeg_call = [
                "ffmpeg",
                "-i",
                audio_path,
                "-af",
                f"highpass=f={low}, lowpass=f={high}",
                filtered_path,
            ]
            if low is None:
                ffmpeg_call[4] = f"lowpass=f={high}"
            elif high is None:
                ffmpeg_call[4] = f"highpass=f={low}"
            else:
                pass
            print(f"Saving filtered audio at {filtered_path}")
            subprocess.call(ffmpeg_call)

    return filtered_paths


def separate_harmonics_percussion(
    audio_paths: list, sampling_rates: list = None, overwrite: bool = False
):
    if sampling_rates is None:
        sampling_rates = np.repeat([SAMPLE_RATE], len(audio_paths))
    harmonics, percussions = [], []
    for audio_path, sr in zip(audio_paths, sampling_rates):
        harmonic_path = os.path.join(
            audio_path.split(os.sep)[0],
            "out",
            "test",
            f"{audio_path.split(os.sep)[-1].split('.')[0]}_harmonic.wav",
        )
        percussion_path = os.path.join(
            audio_path.split(os.sep)[0],
            "out",
            "test",
            f"{audio_path.split(os.sep)[-1].split('.')[0]}_percussion.wav",
        )
        if overwrite or not os.path.exists(harmonic_path):
            audio, _ = librosa.load(audio_path, sr=sr)
            y_harmonic, y_percussive = librosa.effects.hpss(y=audio)
            scaled = np.int16(y_harmonic / np.max(np.abs(y_harmonic)) * 32767)
            write(harmonic_path, sr, scaled)
            scaled = np.int16(y_percussive / np.max(np.abs(y_percussive)) * 32767)
            write(percussion_path, sr, scaled)
        harmonics.append(librosa.load(harmonic_path)[0])
        percussions.append(librosa.load(percussion_path)[0])

    return harmonics, percussions


def ts_to_spec(
    time_series_list: list,
    sampling_rates: list = None,
    n_mels: int = 128,
    log: bool = True,
):
    if sampling_rates is None:
        sampling_rates = np.repeat([SAMPLE_RATE], len(time_series_list))
    spectras = []
    for time_series, sr in zip(time_series_list, sampling_rates):
        temp = []
        for ts in time_series:
            melspec = librosa.feature.melspectrogram(y=ts, sr=sr, n_mels=n_mels)
            if log:
                melspec = librosa.amplitude_to_db(melspec)
            temp.append(melspec)
        spectras.append(temp)
    return spectras


def calc_mfcc(
    audios: list, sampling_rates: list = None, n_mfcc: int = 39, exclude_first=True
):
    mfcc_arr = []
    mean_mfcc_arr = []
    if sampling_rates is None:
        sampling_rates = np.repeat([SAMPLE_RATE], len(audios))
    for audio, sr in zip(audios, sampling_rates):
        mfcc_arr.append(
            librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=n_mfcc,
                lifter=22,
                hop_length=int(sr * 0.01),
                window="hamming",
                win_length=int(sr * 0.025),
            )
        )
        if exclude_first:
            mfcc_arr[-1] = mfcc_arr[-1][:, 1:]
        mean_mfcc_arr.append(mfcc_arr[-1].mean(axis=1))
    return mfcc_arr, mean_mfcc_arr


def calc_mel_spec(audios: list, sampling_rates: list = None, n_mels: int = 128):
    res = []
    if sampling_rates is None:
        sampling_rates = np.repeat([SAMPLE_RATE], len(audios))
    for audio, sr in zip(audios, sampling_rates):
        melspec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        log_S = librosa.amplitude_to_db(melspec)
        res.append(log_S)
    return res


def load_audios(audio_paths: list, sr: float = 22050, duration: Optional[float] = None):
    """Loads and returns audios and sampling rates of the provided files"""
    audios = []
    sampling_rates = []
    for audio_path in audio_paths:
        temp = librosa.load(audio_path, sr=sr, duration=duration)
        audios.append(temp[0])
        sampling_rates.append(temp[1])
    return audios, sampling_rates


def compare_waveforms(waves: list):
    """Plots the waveforms for comparison"""
    plt.figure(figsize=(12, 6))
    for i, wave in enumerate(waves, 0):
        plt.subplot(len(waves), 1, i + 1)
        plt.plot(wave)

    plt.tight_layout()
    plt.draw()


def compare_spectras(spectras_list: list):
    plt.figure(figsize=(12, 6))
    n_rows = len(spectras_list)
    n_cols = 1
    for i, spectra in enumerate(spectras_list):
        plt.subplot(n_rows, n_cols, i + 1)
        librosa.display.specshow(spectra)
        plt.colorbar()

    plt.tight_layout()
    plt.draw()


def calc_snr(signal: np.ndarray, window_size: int = 30):
    # TODO: do not be lazy and rewrite in python
    signal_series = pd.Series(signal)
    rolling_mean = signal_series.rolling(window=window_size).mean()
    rolling_std = signal_series.rolling(window=window_size).std()
    rolling_snr = 10 * np.log10((rolling_mean**2) / (rolling_std**2).replace(0, np.finfo(float).eps))  # type: ignore
    return rolling_snr.mean()


if __name__ == "__main__":

    audio_dir = "speech/data/test/"
    out_dir = "speech/out/test/"
    sample_name1 = "Ben as mjf.m4a"
    sample_name2 = "Mj fox.m4a"

    filtered_paths = filter_audios(
        [os.path.join(audio_dir, sample_name1), os.path.join(audio_dir, sample_name2)],
        low=200,
        high=4000,
    )

    audios, sampling_rates = load_audios(filtered_paths)
    compare_waveforms(audios)
    mfccs, mean_mffcs = calc_mfcc(audios, sampling_rates=sampling_rates)
    compare_waveforms(mean_mffcs)
    compare_spectras(mfccs)

    # harmonics, percussions = separate_harmonics_percussion(filtered_paths)
    # # spectras = ts_to_spec(list(zip(harmonics, percussions)))
    # # compare_spectras(spectras)

    # for harmony in harmonics:
    #     hnr = calc_snr(harmony)
    #     print(f"Harmony to Noise Ratio = {hnr:.3f}")

    plt.show()
