import os

import emo_util
import matplotlib.pyplot as plt
import transcribe_util
import util

if __name__ == "__main__":

    audio_dir = "data/mjfox/"
    out_dir = "out/mjfox/"

    # audio_dir = "speech/data/steven/"
    # out_dir = "speech/out/steven/"

    sample_names = ["Ben_as_mjf.m4a", "Mj_fox.m4a"]
    # sample_names = ["before_name.m4a","after_name.m4a","before_onset.m4a","after_onset.m4a"]
    samples = [os.path.join(audio_dir, sample_name) for sample_name in sample_names]

    filtered_paths = util.filter_audios(samples, low=200, high=4000)

    _, words = transcribe_util.transcribe(filtered_paths)
    waveforms, S, f0, harmonics = util.compute_features(filtered_paths)
    util.plot_harmonics_transcription(
        S, f0, transcripts_arr=words, waveforms_arr=waveforms, titles=["Ben", "MJF"]
    )

    plt.show()

    f0_means, f0_stds = util.get_f0_stats(f0)
    emotions = emo_util.emo_audios(filtered_paths)
    print("For", sample_names, ":")
    print("f0_mean: ", f0_means)
    print("f0_std: ", f0_stds)
    print("Emotions: ", emotions)
