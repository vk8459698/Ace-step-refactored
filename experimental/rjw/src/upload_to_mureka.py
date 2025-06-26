import requests
from pydub import AudioSegment
import io
import os
import sys
import tempfile

def upload_audio(local_filepath, purpose, token):
    assert purpose in ['reference', 'vocal', 'melody', 'instrumental', 'voice'], "Invalid purpose."

    filename, ext = os.path.splitext(local_filepath)
    # Convert to mp3 if not already mp3 or m4a
    if ext not in ['mp3', 'm4a']:
        try:
            print("Converting audio to mp3 format..." , ext)

            audio = AudioSegment.from_file(local_filepath)#, format=ext
            # audio = AudioSegment.from_wav(input_wav_path)
            with tempfile.TemporaryFile() as fp:
                audio.export(fp, format="mp3")
                buffer = io.BytesIO()
                audio.export(buffer, format="mp3")
                buffer.seek(0)
                file_data = ("audio.mp3", buffer, "audio/mpeg")
            
        except Exception as e:
            return f"Audio conversion failed: {e}"
    else:
        audio = AudioSegment.from_file(local_filepath, format=ext)
        buffer = io.BytesIO()
        audio.export(buffer, format="mp3")
        buffer.seek(0)
        file_data = ("audio." + ext, buffer, "audio/mpeg")

    files = {
        "file": file_data,
    }
    data = {
        "purpose": purpose,
    }
    headers = {
        "Authorization": f"Bearer {token}",
    }

    upload_url = "https://api.mureka.ai/v1/files/upload"  # Replace if needed
    res = requests.post(upload_url, files=files, data=data, headers=headers)

    if res.status_code == 200:
        return res.json().get("id", "No ID in response")
    else:
        return f"Upload failed: {res.status_code}, {res.text}"



# "https://storage.googleapis.com/x-one/681babcdc9b9a92b09132ff5/68515673c380294c90d90dc5/manual_trim/59705137-f23d-4e6b-9edb-ead78fde4de6.wav"