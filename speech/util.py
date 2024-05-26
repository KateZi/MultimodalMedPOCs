import os
import subprocess
import warnings
from typing import Optional

import librosa
import noisereduce
import numpy as np
import pandas as pd
from scipy.io.wavfile import write
from scipy.stats import entropy
from Signal_Analysis.features import signal as SA

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


SAMPLE_RATE = 16_000
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 128
HARMONICS = np.arange(1, 11)
FMAX = 5000


def filter_audios(
    audio_paths: list, low: int = 200, high: int = 3000, overwrite: bool = False
):
    """Filters and saves audios with a suffix "<low>_<high>" using FFMPEG
    if low is None - performs only highpass
    if high is None - only lowpass
    else - bandpass

    TODO: filtered_path only considers case of both low and high,
            edit for only low and oly high
    """
    filtered_paths = []
    for audio_path in audio_paths:
        if not os.path.exists(audio_path):
            print(f"Could not locate {audio_path}, skipping")
            continue
        audio_path = audio_path.split(os.sep)
        filtered_path = os.path.join(
            "out",
            audio_path[1],
            f"{audio_path[-1].split('.')[0]}_{low}_{high}.wav",
        )
        if overwrite or not os.path.exists(filtered_path):
            ffmpeg_call = [
                "ffmpeg",
                "-y",
                "-i",
                os.sep.join(audio_path),
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
        filtered_paths.append(filtered_path)

    return filtered_paths


def normalize_loudness(audio_paths: list, overwrite: bool = False):
    """
    Normalizes audios and saves them with suffix "_norm"
    Caution: normalization can disturb the source and
    information
    """
    normalized_paths = []
    for audio_path in audio_paths:
        if not os.path.exists(audio_path):
            print(f"Could not locate {audio_path}, skipping")
            continue
        audio_path = audio_path.split(os.sep)
        normalized_path = os.path.join(
            "out",
            audio_path[1],
            f"{audio_path[-1].split('.')[0]}_norm.wav",
        )
        if overwrite or not os.path.exists(normalized_path):
            subprocess.call(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    os.sep.join(audio_path),
                    "-filter:a",
                    "loudnorm",
                    "-c:a",
                    "ac3",
                    "-c:v",
                    "copy",
                    normalized_path,
                ]
            )
        normalized_paths.append(normalized_path)

    return normalized_paths


def separate_harmonics_percussion(
    audio_paths: list, sampling_rates: list = None, overwrite: bool = False
):
    """Separates harmonics and percussion and returns them separately"""
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


def calc_mel_spec(
    time_series_list: list,
    sampling_rates: list = None,
    n_mels: int = 128,
    log: bool = True,
):
    """Calculates mel-spectrograms for a list of audios"""
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
    """Calculates mfcc for a list of audios"""
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


def load_audios(audio_paths: list, sr: float = 22050, duration: Optional[float] = None):
    """Loads and returns audios and sampling rates of the provided files"""
    audios = []
    sampling_rates = []
    for audio_path in audio_paths:
        temp = librosa.load(audio_path, sr=sr, duration=duration)
        audios.append(temp[0])
        sampling_rates.append(temp[1])
    return audios, sampling_rates


def calc_snr(signal: np.ndarray, window_size: int = 30):
    """
    Calculates signal to noise ratio in a simplified manner

    """
    signal_series = pd.Series(signal)
    rolling_mean = signal_series.rolling(window=window_size).mean()
    rolling_std = signal_series.rolling(window=window_size).std()
    rolling_snr = 10 * np.log10((rolling_mean**2) / (rolling_std**2).replace(0, np.finfo(float).eps))  # type: ignore
    return rolling_snr.mean()


def compute_features(
    audio_paths: dict,
    sr: int = SAMPLE_RATE,
    harmonics: list = HARMONICS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
):
    """Returns a dictionary of features for each file.
    Includes: waveform, sampling rate, melspectrogram,
    fundamental frequency and harmonics' energy,
    as well as paramters used, n_fft and hop_length.
    These features are useful for comparative figures.
    """
    features = {key: {} for key in audio_paths.keys()}

    for key, audio_path in audio_paths.items():
        features[key]["n_fft"] = n_fft
        features[key]["hop_length"] = hop_length

        y, sr = librosa.load(audio_path, sr=sr)
        y = noisereduce.reduce_noise(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        # y = (y - y.min()) / (y.max() - y.min())
        # y = (y - y.mean()) / (y.std())
        amp = 1.0
        y = amp * y / max(abs(max(y)), abs(min(y)))
        features[key]["waveform"] = y
        features[key]["sr"] = sr

        f0, voiced, _ = librosa.pyin(
            y=y,
            sr=sr,
            fmin=40,
            fmax=400,
            hop_length=hop_length,
            frame_length=n_fft,
            max_transition_rate=20,
            n_thresholds=10,
        )

        features[key]["f0"] = f0
        features[key]["voice_flag"] = voiced

        S = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length))
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        harmonic_energy = librosa.f0_harmonics(
            S,
            f0=f0,
            harmonics=harmonics,
            freqs=frequencies,
        )
        features[key]["harmonics_energy"] = harmonic_energy
        features[key]["salience"] = librosa.salience(
            S, freqs=frequencies, harmonics=harmonics, fill_value=0
        )

        S = librosa.feature.melspectrogram(
            S=S**2,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmax=FMAX,
        )
        features[key]["S"] = S

    return features


def runs_of_values(arr: np.ndarray):
    """Returns an array of begin and end indeces of runs of non-nan values"""
    runs = np.where(np.diff(~np.isnan(arr)))[0]
    if runs.shape[0] % 2 != 0:
        runs = np.hstack((runs, arr.shape[-1] - 1))
    return runs.reshape(-1, 2) + 1


def get_Shimmer(waveform: np.ndarray, f0: np.ndarray):
    """Calculates voice shimmer"""
    runs_of_f0 = runs_of_values(f0)
    A = np.array([np.ptp(waveform[b:e]) for b, e in runs_of_f0])
    shimmer = np.sum([abs(20 * np.log10(A[1:] / A[:-1]))])
    shimmer = shimmer / (len(A) - 1)
    return shimmer


def get_stats(features: dict):
    """Returns a dataframe of discrete statistics over the calculated features"""
    pitch_max = 400
    res = {key: {} for key in features.keys()}
    for key in features.keys():
        waveform = features[key]["waveform"]
        f0 = features[key]["f0"]
        sr = features[key]["sr"]
        # harmonics_energy = features[key]["harmonics_energy"]

        res[key]["Pitch (inversed)"] = pitch_max - np.nanmean(f0)
        res[key]["Intonation variability"] = np.nanstd(f0)
        res[key]["Speechiness"] = 1 / np.nanmean(entropy(features[key]["S"], axis=0))
        res[key]["Breathiness (inversed)"] = 1 / get_Shimmer(waveform, f0)

        # jitter = SA.get_Jitter(waveform, sr)
        # res[key]["Voice roughness (inversed)"] = 1 / (jitter["local"])
        # res[key]["Voice clarity"] = SA.get_HNR(waveform, sr, min_pitch=40)

        # add a small constant to avoid high crossing rate at silence (if want)
        # zero_cross_rate = librosa.feature.zero_crossing_rate(waveform+0.0001, frame_length=N_FFT, hop_length=HOP_LENGTH)
        # zero_cross_rate = zero_cross_rate[np.where(zero_cross_rate<np.median(zero_cross_rate))[0]]
        # res[key]["Voice smoothness"] = 1/zero_cross_rate.mean()
        # res[key]["Voice strength"] = harmonics_energy.sum()

    res = pd.DataFrame.from_dict(res, orient="index")
    return res
