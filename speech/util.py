import os
import subprocess
from typing import Optional

import librosa
import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from scipy.io.wavfile import write

SAMPLE_RATE = 22050
N_FFT = 2048
HARMONICS = np.arange(1, 11)
CONTEXT = "This is a transcription of a Parkinson's disease patient. Keep punctuation. Keep stuttering. So uhm, yeaah. Okay, ehm, uuuh."


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
        audio_path = audio_path.split(os.sep)
        filtered_path = os.path.join(
            "out",
            audio_path[2],
            f"{audio_path[-1].split('.')[0]}_{low}_{high}.wav",
        )
        filtered_paths.append(filtered_path)
        if overwrite or not os.path.exists(filtered_path):
            ffmpeg_call = [
                "ffmpeg",
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

    return filtered_paths


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
    """Calculates melspectrograms for a list of audios"""
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
        n_rows = len(spectras_list)
        n_cols = 1
        _, ax = plt.subplots(figsize=(12, 6), nrows=nrows, ncols=ncols)

    for i, spectra in enumerate(spectras_list):
        im = librosa.display.specshow(spectra, ax=ax[i], y_axis=y_axis)
        plt.colorbar(im, ax=ax[i])

    plt.tight_layout()
    plt.draw()


def calc_snr(signal: np.ndarray, window_size: int = 30):
    # TODO: do not be lazy and rewrite in python
    signal_series = pd.Series(signal)
    rolling_mean = signal_series.rolling(window=window_size).mean()
    rolling_std = signal_series.rolling(window=window_size).std()
    rolling_snr = 10 * np.log10((rolling_mean**2) / (rolling_std**2).replace(0, np.finfo(float).eps))  # type: ignore
    return rolling_snr.mean()


def compute_features(
    audio_paths: list,
    sr: int = SAMPLE_RATE,
    harmonics: list = HARMONICS,
    n_fft: int = N_FFT,
):
    """Computes and returns waveform of the audios, spectrogram,
    fundamental frequency and harmonics' energy
    These features are useful for comparative figures.
    """
    waveforms, S_arr, f0_arr, harmonics_arr = [], [], [], []

    for audio_path in audio_paths:
        y, sr = librosa.load(audio_path, sr=sr)
        waveforms.append(y)

        f0, *_ = librosa.pyin(y=y, sr=sr, fmin=20, fmax=350)
        f0_arr.append(f0)

        S = np.abs(librosa.stft(y, n_fft=N_FFT))
        S_arr.append(S)

        frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        harmonic_energy = librosa.f0_harmonics(
            S, f0=f0, harmonics=harmonics, freqs=frequencies
        )
        harmonics_arr.append(harmonic_energy)

    return waveforms, S_arr, f0_arr, harmonics_arr


def plot_f0(
    S_arr: list,
    f0_arr: list,
    times_arr: list,
    y_axis="log",
    harmonics: list = HARMONICS,
    ax: Optional[list] = None,
):
    """Plots the fundamental frequency overlaid over a spectrogram
    and highlights the harmonics with the 'harmonics' range.
    """
    if ax is None:
        ncols = len(S_arr)
        _, ax = plt.subplots(figsize=(12, 6), ncols=ncols)

    for i, (S, f0, times) in enumerate(zip(S_arr, f0_arr, times_arr)):
        librosa.display.specshow(
            librosa.amplitude_to_db(S, ref=np.max),
            y_axis=y_axis,
            x_axis="time",
            ax=ax[i],
        )
        for h in np.arange(1, len(harmonics) + 1):
            if h == 1:
                ax[i].plot(times, f0, linewidth=2, color="white", label="f0")
            else:
                ax[i].plot(times, h * f0, label=f"{h}*f0")


def plot_transcripts(
    transcripts_arr: list,
    waveforms_arr: list,
    sr: int = SAMPLE_RATE,
    ax: Optional[list] = None,
):
    """Plots transcription - words with timing; overlaud over the audio waveform"""
    if ax is None:
        ncols = len(transcripts_arr)
        _, ax = plt.subplots(figsize=(12, 6), ncols=ncols)

    for i, (words, waveform) in enumerate(zip(transcripts_arr, waveforms_arr)):
        duration = librosa.get_duration(y=waveform)
        times = np.arange(0, duration, 1 / sr)

        ax[i].plot(times, waveform)

        ticks = []
        for word in words:
            ax[i].axvline(word["start"], color="green", linestyle="dotted")  # *1000/sr)
            ax[i].axvline(word["end"], color="red", linestyle="dashed")  # *1000/sr)
            ticks.append((word["end"] - word["start"]) / 2 + word["start"])

        ax[i].set_xticks(ticks)
        ax[i].set_xticklabels(
            [word["word"] for word in words], rotation=90, ha="center"
        )

        ax[i].set_xlim(times[0], times[-1])


def plot_harmonics(
    harmonics_arr: list, harmonics: list = HARMONICS, ax: Optional[list] = None
):
    """Plots harmonics energies through time"""
    if ax is None:
        ncols = len(harmonics_arr)
        _, ax = plt.subplots(figsize=(12, 6), ncols=ncols)
    for i, harmonics in enumerate(harmonics_arr):
        librosa.display.specshow(
            librosa.amplitude_to_db(harmonics, ref=np.max), x_axis="time", ax=ax[-1, i]
        )

        ax[i].set_yticks(harmonics - 1)
        ax[i].set_yticklabels(harmonics)
        ax[i].set(ylabel="Harmonics")


def plot_harmonics_transcription(
    S_arr: list,
    f0_arr: list,
    harmonics_arr: Optional[list] = None,
    transcripts_arr: Optional[list] = None,
    waveforms_arr: Optional[list] = None,
    y_axis="log",
):
    """Plots the spectragrams with f0 and harmonics overlay
    Optionally: plots Harmonics energy through time
                plots transcription through time
    """

    transcripts_flag = transcripts_arr is not None
    harmonics_flag = harmonics_arr is not None

    nrows = 1
    if transcripts_flag:
        nrows += 1
    if harmonics_flag:
        nrows += 1
    _, ax = plt.subplots(figsize=(10, 4), nrows=nrows, ncols=len(S_arr))

    times = [librosa.times_like(S) for S in S_arr]
    plot_f0(S_arr, f0_arr, times, y_axis, ax=ax[0])
    if transcripts_flag:
        plot_transcripts(transcripts_arr, waveforms_arr, ax=ax[1])
    if harmonics_flag:
        prol_harmonics(harmonics_arr, ax=ax[-1])

    plt.tight_layout()
    plt.draw()


def transcribe_oai(audio_file_path: str, context: str = ""):
    """Transcribes a single audio using OpenAI"""
    res = load_dotenv(find_dotenv())
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not res and not openai.api_key:
        raise Exception("Please provide and OPENAI_API_KEY")

    client = OpenAI()
    audio_file = open(audio_file_path, "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        prompt=context,
        timestamp_granularities=["word"],
        response_format="verbose_json",
    )

    return transcription


def transcribe(audio_paths: list, context: str = CONTEXT):
    """Sends audios for transcription
    Returns texts and words level separately.
    """
    texts, words = [], []
    for audio_path in audio_paths:
        transcription = transcribe_oai(audio_file_path=audio_path, context=context)

        texts.append(transcription.text)
        words.append(transcription.words)

    return texts, words
