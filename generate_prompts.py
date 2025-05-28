#!/usr/bin/env python3

import argparse
import base64
import json
import os
from io import BytesIO

import requests
import torchaudio

QWEN_SAMPLE_RATE = 16000

# Roughly based on the official prompt https://github.com/ace-step/ACE-Step/blob/main/TRAIN_INSTRUCTION.md
PROMPT = """Analyze the input audio and generate a description. The description must be < 200 characters. Follow these exact definitions:

1. `genre`: Most representative genres of the audio.
2. `subgenre`: Three or more tags of specific sub-genres and techniques.
3. `instrument`: All audibly present instruments in the audio, except vocal.
4. `tempo`: Tags describing the tempo of the audio. Do not use number or BPM.
5. `mood`: Tags describing the mood of the audio.
6. `has_vocal`: Whether there is any vocal in the audio.
7. `vocal`: If there is any vocal in the audio, then output a list of tags describing the vocal timbre. Otherwise, output an empty list.

Strictly ignore any information derived solely from the lyrics when performing the analysis, especially for identifying instruments.

Output format:
```json
{
  "genre": <str list>,
  "subgenre": <str list>,
  "instrument": <str list>,
  "tempo": <str list>,
  "mood": <str list>,
  "has_vocal": <bool>,
  "vocal": <str list>
}
```"""


def get_tags(file_path, host, port, temperature):
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
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio,
                            "format": "wav",
                        },
                    },
                    {
                        "type": "text",
                        "text": PROMPT,
                    },
                ],
            }
        ],
        "cache_prompt": True,
        "temperature": temperature,
        "top_k": 50,
        "top_p": 1,
        "min_p": 0,
    }

    url = f"http://{host}:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)

    content = response.json()
    content = content["choices"][0]["message"]["content"]
    content = content.replace("```json", "")
    content = content.replace("```", "")
    content = content.strip()

    tags = []
    try:
        data = json.loads(content)
        tags += data["genre"]
        tags += data["subgenre"]
        tags += data["instrument"]
        tags += data["tempo"]
        tags += data["mood"]
        tags += [x.strip() + " vocal" for x in data["vocal"]]
    except Exception:
        print("Failed to parse content")
        print(content)
    return tags


def do_files(data_dir, host, port, overwrite):
    for root, _dirs, files in os.walk(data_dir):
        for file in sorted(files):
            stem, ext = os.path.splitext(file)
            # Formats supported by torchaudio
            if ext.lower() not in [
                ".aac",
                ".flac",
                ".mp3",
                ".ogg",
                ".wav",
            ]:
                continue

            file_path = os.path.join(root, file)
            stem_path = os.path.join(root, stem)
            prompt_path = stem_path + "_prompt.txt"

            if (not overwrite) and os.path.exists(prompt_path):
                continue

            tags = []
            # Try 3 different temperatures
            for temperature in [0, 0.5, 1]:
                tags += get_tags(file_path, host, port, temperature)

            tags = [x.strip().lower() for x in tags]
            tags = list(set(tags))
            tags = ", ".join(tags)

            print(file, tags)
            with open(prompt_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(tags)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"C:\data\audio")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    do_files(
        data_dir=args.data_dir,
        host=args.host,
        port=args.port,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
