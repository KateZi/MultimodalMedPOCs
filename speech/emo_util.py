import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

EMOTIONS = ["Arousal", "Dominance", "Valence"]


class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
        self,
        input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits


def process_func(
    model: EmotionModel,
    processor,
    x: np.ndarray,
    sampling_rate: int,
    embeddings: bool = False,
    device: str = "cpu",
) -> np.ndarray:
    r"""Predict emotions or extract embeddings from raw audio signal."""

    # run through processor to normalize signal
    # always returns a batch, so we just get the first entry
    # then we put it on the device
    y = processor(x, sampling_rate=sampling_rate)
    y = y["input_values"][0]
    y = y.reshape(1, -1)
    y = torch.from_numpy(y).to(device)

    # run through model
    with torch.no_grad():
        y = model(y)[0 if embeddings else 1]

    # convert to numpy
    y = y.detach().cpu().numpy()

    return y


def res_to_dict(y: np.ndarray):
    """Returns the results in a dictionary format"""
    y = y.squeeze()
    res = {}
    for i in range(y.shape[0]):
        res[EMOTIONS[i]] = y[i]
    return res


def emo_audio(audio_path: str, device: str = "cpu"):
    """Evaluates 3 emotions from an audio
    Arousal, dominance and valence
    """
    model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = EmotionModel.from_pretrained(model_name).to(device)

    audio, sr = librosa.load(audio_path, sr=16_000)

    return res_to_dict(process_func(model, processor, audio, sr, device=device))


def emo_audios(audio_paths: dict, device: str = "cpu"):
    """Extracts emotions for the dict of audios, returns a pd.DataFrame"""
    emo_res = {}
    for key, audio_path in audio_paths.items():
        emo_res[key] = emo_audio(audio_path, device)
    emo_res = pd.DataFrame.from_dict(emo_res, orient="index")
    return emo_res
