# 📁 routes/embeddings.py
from fastapi import APIRouter, UploadFile, Form, File, HTTPException
from fastapi.responses import StreamingResponse
from typing import List
from openai import OpenAI
import os
import time
from firebase_admin import auth
from utils.image import compress_image
from utils.detection import detect_confident_objects, update_best_detection, get_best_images, reset_best_detection
from utils.stt import transcribe_openai
from utils.tts import openai_tts
from utils.firebase import save_msg
from utils.openai_utils import blocks_to_text
from utils.firebase import save_temp_text, get_temp_text  # ho crearem a continuació


router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID")


@router.post("/prepareTextFromAudio")
async def prepare_text_from_audio(thread_id: str = Form(...), audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    text = transcribe_openai(audio_bytes, audio.filename or "audio.m4a")
    print(f"📅 Transcripció rebuda: {text}")
    
    # 🧠 Desa la transcripció temporalment vinculada al thread
    save_temp_text(thread_id, text)

    return {"status": "ok", "text": text}

@router.post("/prepareImage")
async def prepare_image(
    thread_id: str = Form(...),
    index: int = Form(...),
    image: UploadFile = File(...),
    text: str = Form(None),  
    force_reset: bool = Form(False),
):
    if force_reset:
        reset_best_detection(thread_id)
    try:
        if not text:
            text = get_temp_text(thread_id)
            if not text:
                raise HTTPException(status_code=400, detail="No hi ha cap transcripció disponible")

        print(f"📷 Rebuda imatge {index} per a thread {thread_id}")
        image_bytes = await image.read()
        detections = detect_confident_objects(image_bytes, text)
        update_best_detection(thread_id, image_bytes, detections)

        best_score = max([d["confidence"] for d in detections], default=0.0)
        print(f"✅ Confiança imatge {index}: {best_score:.3f}")
        return {"status": "ok", "confidence": best_score}

    except Exception as e:
        print(f"❌ Error processant imatge {index}: {e}")
        raise HTTPException(status_code=500, detail="Error al processar la imatge")


# 🔍 3. Endpoint per buscar objecte (enviar millor imatge + transcripció a OpenAI)
@router.post("/searchObject")
async def search_object(
    token: str = Form(...),
    thread_id: str = Form(...)
):
    try:
        uid = auth.verify_id_token(token)["uid"]

        # 📄 Recuperem la transcripció temporal guardada abans
        transcription = get_temp_text(thread_id)
        if not transcription:
            raise HTTPException(400, "No s'ha trobat cap transcripció per aquest thread.")

        print(f"📅 Transcripció recuperada: {transcription}")
        save_msg(uid, thread_id, "user", transcription)

        # 🖼️ Agafem les dues millors imatges
        best_images = get_best_images(thread_id)
        if not best_images or len(best_images) == 0:
            raise HTTPException(400, "No s'ha trobat cap imatge vàlida.")

        # Comprimim i les convertim en fitxers per a l'API d'OpenAI
        files = [
            client.files.create(
                file=(f"image_{i}.jpg", compress_image(img)), 
                purpose="assistants"
            )
            for i, img in enumerate(best_images)
        ]

        # Preparem el contingut multimodal
        content = [{"type": "text", "text": transcription}]
        for f in files:
            content.append({"type": "image_file", "image_file": {"file_id": f.id}})

        # Enviem a l'assistent
        client.beta.threads.messages.create(
            thread_id=thread_id, role="user", content=content
        )

        run = client.beta.threads.runs.create(
            thread_id=thread_id, assistant_id=ASSISTANT_ID, tool_choice="auto"
        )

        # Esperem la resposta
        while True:
            info = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            if info.status in ["completed", "failed"]:
                break
            time.sleep(0.5)

        if info.status == "failed":
            raise HTTPException(500, "La run ha fallat.")

        # 🔁 Obtenim resposta de Keia
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
        print("❌ /searchObject ERROR:", e)
        raise HTTPException(500, "backend error")
