from fastapi import APIRouter, Form, HTTPException
from firebase_admin import auth, firestore
from openai import OpenAI
import os

router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
db = firestore.client()

@router.get("/exists")
def thread_exists(token: str, thread_id: str):
    try:
        uid = auth.verify_id_token(token)["uid"]
        doc_ref = db.collection("users").document(uid).collection("threads").document(thread_id)
        return {"exists": doc_ref.get().exists}
    except Exception as e:
        print("❌ /threads/exists ERROR:", e)
        raise HTTPException(500, "Error consultant si existeix el thread")

@router.post("/create_thread")
def create_thread(token: str = Form(...)):
    try:
        uid = auth.verify_id_token(token)["uid"]
        thread = client.beta.threads.create()
        db.collection("users").document(uid).collection("threads").document(thread.id).set({
            "created": firestore.SERVER_TIMESTAMP,
            "title": "Conversa sense títol"
        })
        return {"thread_id": thread.id}
    except Exception as e:
        print("❌ /threads/create ERROR:", e)
        raise HTTPException(500, "Error creant thread")
