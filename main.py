from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os, json, base64
from firebase_admin import credentials, initialize_app
from pathlib import Path

# Carrega variables del .env
load_dotenv()

# ── Firebase ────────────────────────────────────────────
firebase_creds_b64 = os.getenv("FIREBASE_CREDS_B64")
firebase_creds = json.loads(base64.b64decode(firebase_creds_b64))
cred = credentials.Certificate(firebase_creds)
initialize_app(cred)

# ── Google Cloud ────────────────────────────────────────
google_creds_b64 = os.getenv("GOOGLE_CREDS_B64")

google_creds_path = str(Path(__file__).parent / "google-credentials.json")

with open(google_creds_path, "w") as f:
    json.dump(json.loads(base64.b64decode(google_creds_b64)), f)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_creds_path


# ── App FastAPI ─────────────────────────────────────────
app = FastAPI(title="Keia · backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Import i registre de rutes ──────────────────────────
@app.get("/")
async def root():
    return {"status": "Keia backend is alive."}

from routes.ask_keia import router as ask_keia_router
from routes.thread import router as thread_router
from routes.embeddings import router as embeddings_router
from routes.yolo import router as yolo_router

app.include_router(ask_keia_router, prefix="/askKeia")
app.include_router(thread_router, prefix="/threads")
app.include_router(embeddings_router, prefix="/embedding")
app.include_router(yolo_router, prefix="/vision")