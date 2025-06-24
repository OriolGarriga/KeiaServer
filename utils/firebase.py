from firebase_admin import firestore

db = firestore.client()

def save_msg(uid: str, thread_id: str, role: str, content: str):
    doc_ref = db.collection("users").document(uid).collection("threads").document(thread_id)
    doc_ref.set({"created": firestore.SERVER_TIMESTAMP}, merge=True)
    return doc_ref.collection("messages").add({
        "role": role,
        "content": content,
        "ts": firestore.SERVER_TIMESTAMP
    })

def save_temp_text(thread_id: str, text: str):
    db = firestore.client()
    db.collection("temp_embeddings").document(thread_id).set({"text": text})

def get_temp_text(thread_id: str) -> str:
    db = firestore.client()
    doc = db.collection("temp_embeddings").document(thread_id).get()
    return doc.to_dict().get("text") if doc.exists else ""
