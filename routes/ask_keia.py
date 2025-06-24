from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from utils.stt import transcribe_openai
from utils.tts import openai_tts
from utils.image import compress_image
from utils.firebase import save_msg
from utils.openai_utils import blocks_to_text
from openai import OpenAI
import os, io, time
from firebase_admin import auth

router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID")

@router.post("/stream")
def ask_keia_stream(
    token: str = Form(...),
    thread_id: str = Form(...),
    image: UploadFile = File(...),
    audio: UploadFile = File(None)
):
    try:
        uid = auth.verify_id_token(token)["uid"]
        if image is None:
            raise HTTPException(400, "Cal una imatge.")

        img_bytes = image.file.read()
        audio_bytes = audio.file.read() if audio else None

        compressed_bytes = compress_image(img_bytes)
        question = transcribe_openai(audio_bytes, audio.filename or "audio.m4a") if audio_bytes else "(Sense veu — només imatge)"

        fname = image.filename or "img.jpg"
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            fname += ".jpg"

        img_file = client.files.create(file=(fname, compressed_bytes), purpose="assistants")

        print("📤 Pregunta:", question)
        print("📤 ID imatge:", img_file.id)
        save_msg(uid, thread_id, "user", question)

        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=[
                {"type": "text", "text": question},
                {"type": "image_file", "image_file": {"file_id": img_file.id}},
            ],
        )

        run = client.beta.threads.runs.create(thread_id=thread_id, assistant_id=ASSISTANT_ID, tool_choice="auto")
        while True:
            info = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            if info.status in ["completed", "failed"]:
                break
            time.sleep(0.5)

        if info.status == "failed":
            raise HTTPException(500, "La run ha fallat.")

        messages = sorted(
            client.beta.threads.messages.list(thread_id=thread_id).data,
            key=lambda m: m.created_at,
            reverse=True
        )

        for m in messages:
            if m.role == "assistant":
                txt = blocks_to_text(m.content).strip()
                if txt:
                    save_msg(uid, thread_id, "assistant", txt)
                    audio_data = openai_tts(txt)
                    return StreamingResponse(iter([audio_data]), media_type="audio/mpeg")

        return StreamingResponse(iter(["No he trobat cap resposta."]), media_type="text/plain")

    except Exception as e:
        print("❌ /askKeia/stream ERROR:", e)
        raise HTTPException(500, "backend error")
