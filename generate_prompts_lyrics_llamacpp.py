#!/usr/bin/env python3

import argparse
import base64
import os
from io import BytesIO

import requests
import torchaudio

from generate_prompts_lyrics import (
    PROMPT,
    PROMPT_LYRICS,
    QWEN_SYSTEM_PROMPT,
    parse_prompt_lyrics,
)

QWEN_SAMPLE_RATE = 16000


def inference(file_path, host, port, do_lyrics, temperature):
    audio, sr = torchaudio.load(file_path)
    if sr != QWEN_SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, QWEN_SAMPLE_RATE)
        sr = QWEN_SAMPLE_RATE

    buffer = BytesIO()
    torchaudio.save(buffer, audio, sample_rate=sr, format="wav")
    audio = base64.b64encode(buffer.getvalue()).decode("utf-8")
    del buffer

    payload = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT_LYRICS if do_lyrics else PROMPT,
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio,
                            "format": "wav",
                        },
                    },
                ],
            },
        ],
        "cache_prompt": True,
        "temperature": temperature,
        "top_k": 50,
        "top_p": 1,
        "min_p": 0,
        "max_tokens": 1000,
    }

    url = f"http://{host}:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    content = response.json()
    content = content["choices"][0]["message"]["content"]
    return content


def do_files(data_dir, host, port, overwrite, do_lyrics):
    # Formats supported by torchaudio
    extensions = {
        ".aac",
        ".flac",
        ".m4a",
        ".mp3",
        ".ogg",
        ".wav",
    }

    for file in sorted(os.listdir(data_dir)):
        stem, ext = os.path.splitext(file)
        if ext.lower() not in extensions:
            continue

        file_path = os.path.join(data_dir, file)
        stem_path = os.path.join(data_dir, stem)
        prompt_path = stem_path + "_prompt.txt"
        lyrics_path = stem_path + "_lyrics.txt"

        need_prompt = overwrite or (not os.path.exists(prompt_path))
        need_lyrics = do_lyrics and (overwrite or (not os.path.exists(lyrics_path)))

        if not (need_prompt or need_lyrics):
            continue

        print(file)
        content = inference(file_path, host, port, do_lyrics, temperature=1)
        prompt, lyrics = parse_prompt_lyrics(content)

        if need_prompt:
            with open(prompt_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(prompt)
        if need_lyrics:
            with open(lyrics_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(lyrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"C:\data\audio")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--lyrics", action="store_true")
    args = parser.parse_args()

    do_files(
        data_dir=args.data_dir,
        host=args.host,
        port=args.port,
        overwrite=args.overwrite,
        do_lyrics=args.lyrics,
    )


if __name__ == "__main__":
    main()
