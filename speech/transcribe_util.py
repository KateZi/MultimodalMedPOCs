import os

import openai
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

CONTEXT = "This is a transcription of a Parkinson's disease patient. Keep punctuation. Keep stuttering. So uhm, yeaah. Okay, ehm, uuuh."


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


def transcribe(audio_paths: dict, context: str = CONTEXT):
    """Sends audios for transcription
    Returns dictionary containing the results.
    """
    transcription_res = {key: {} for key in audio_paths.keys()}
    texts, words = [], []
    for key, audio_path in audio_paths.items():
        transcription = transcribe_oai(audio_file_path=audio_path, context=context)

        transcription_res[key]["text"] = transcription.text
        transcription_res[key]["words"] = transcription.words

    return transcription_res
