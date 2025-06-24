import os, requests

def openai_tts(text: str) -> bytes:
    res = requests.post(
        "https://api.openai.com/v1/audio/speech",
        headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}", "Content-Type": "application/json"},
        json={"model": "tts-1", "input": text, "voice": "nova", "response_format": "mp3"},
    )
    return res.content if res.status_code == 200 else b""
