from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import credentials, initialize_app
import os, dotenv

# ── Config global ────────────────────────────────────
dotenv.load_dotenv()

cred = credentials.Certificate("firebase-admin-sdk.json")
initialize_app(cred)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google-credentials.json"

# ── App FastAPI ──────────────────────────────────────
app = FastAPI(title="Keia · backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Import i registre de rutes ───────────────────────
from routes.ask_keia import router as ask_keia_router
from routes.thread import router as thread_router
from routes.embeddings import router as embeddings_router

app.include_router(ask_keia_router, prefix="/askKeia")
app.include_router(thread_router, prefix="/threads")
app.include_router(embeddings_router, prefix="/embedding")
