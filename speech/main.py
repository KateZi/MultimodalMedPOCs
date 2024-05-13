import os

import emo_util
import matplotlib.pyplot as plt
import plot_util
import transcribe_util
import util

if __name__ == "__main__":

    audio_dir = "data/mjfox/"
    out_dir = "out/mjfox/"

    audio_dir = "data/steven/"
    out_dir = "out/steven/"

    fig_path = "figures"

    sample_names = ["Ben_as_mjf.m4a", "Mj_fox.m4a"]
    sample_names = ["before_name.m4a", "after_name.m4a"]
    samples = [os.path.join(audio_dir, sample_name) for sample_name in sample_names]

    filtered_paths = util.filter_audios(samples, low=200, high=5000)
    # normalized_paths = util.normalize_loudness(filtered_paths, overwrite=False)
    normalized_paths = filtered_paths

    normalized_paths = {"Before": normalized_paths[0], "After": normalized_paths[1]}

    plot_util.anim_sync_audios(normalized_paths, out_dir=fig_path)

    transcription = transcribe_util.transcribe(normalized_paths)
    features = util.compute_features(normalized_paths, sr=16_000)
    plot_util.plot_harmonics_transcription(
        features,
        transcription,
        harmonics_flag=False,
        y_axis="mel",
        title="Speech Therapy Results",
        save_path=os.path.join(fig_path, "harmonics_transcription.svg"),
    )

    stats = util.get_stats(features)
    emotions = emo_util.emo_audios(normalized_paths)
    stats = stats.join(emotions)
    plot_util.plot_radar(
        stats,
        title="Speech Therapy Results",
        save_path=os.path.join(fig_path, "radar.svg"),
    )
    # print("Stats for ", sample_names)
    # print(stats)

    # plt.show()
