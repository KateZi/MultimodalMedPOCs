import os

import emo_util
import matplotlib.pyplot as plt
import transcribe_util
import util

if __name__ == "__main__":

    audio_dir = "data/mjfox/"
    out_dir = "out/mjfox/"

    # audio_dir = "data/steven/"
    # out_dir = "out/steven/"

    sample_names = ["Ben_as_mjf.m4a", "Mj_fox.m4a"]
    # sample_names = ["before_name.m4a","after_name.m4a","before_onset.m4a","after_onset.m4a"]
    # sample_names = ["before_name.m4a", "after_name.m4a"]
    samples = [os.path.join(audio_dir, sample_name) for sample_name in sample_names]

    filtered_paths = util.filter_audios(samples, low=200, high=5000)
    normalized_paths = util.normalize_loudness(filtered_paths)

    normalized_paths = {"ben": normalized_paths[0], "mjf": normalized_paths[1]}

    transcription = transcribe_util.transcribe(normalized_paths)
    features = util.compute_features(normalized_paths, sr=16_000)
    util.plot_harmonics_transcription(
        features, transcription, harmonics_flag=False, y_axis="mel"
    )

    stats = util.get_stats(features)
    emotions = emo_util.emo_audios(normalized_paths)
    stats = stats.join(emotions)
    util.plot_radar(stats)
    # print("Stats for ", sample_names)
    # print(stats)

    plt.show()
