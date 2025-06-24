import os, subprocess, tempfile, requests

def transcribe_openai(audio_bytes: bytes, filename: str = "audio.aac") -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".aac") as input_f, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as output_f:
            input_f.write(audio_bytes)
            input_path = input_f.name
            output_path = output_f.name

        subprocess.run(["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", output_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        with open(output_path, "rb") as f:
            wav_bytes = f.read()

        response = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
            data={'model': 'whisper-1', 'language': 'ca', 'response_format': 'text'},
            files={'file': ('audio.wav', wav_bytes)}
        )

        if response.status_code != 200:
            print("❌ Error transcrivint amb OpenAI:", response.text)
            return "(Error transcrivint amb OpenAI)"

        return response.text.strip()
    except Exception as e:
        print("❌ Error al convertir i transcriure:", e)
        return "(Error convertint/transcrivint àudio)"
    finally:
        os.remove(input_path)
        os.remove(output_path)
