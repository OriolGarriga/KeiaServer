# Fitxer: utils/detection.py

import io
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from deep_translator import GoogleTranslator

# Carreguem els models una sola vegada
dino_model_id = "IDEA-Research/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
dino_processor = AutoProcessor.from_pretrained(dino_model_id)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to(device)

# Magatzem temporal per a deteccions
best_detection_store = {}

def reset_best_detection(thread_id: str):
    if thread_id in best_detection_store:
        del best_detection_store[thread_id]

def detect_confident_objects(image_bytes: bytes, user_text_cat: str, threshold: float = 0.2):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    try:
        text_en = GoogleTranslator(source='ca', target='en').translate(user_text_cat)
        print(text_en)
    except Exception as e:
        print("❌ Error traduint:", e)
        text_en = user_text_cat

    if not text_en.strip().endswith('.'):
        text_en += '.'

    inputs = dino_processor(images=image, text=text_en, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino_model(**inputs)

    results = dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=threshold,
        text_threshold=threshold,
        target_sizes=[image.size[::-1]]
    )[0]

    detections = []
    for box, label, score in zip(results["boxes"], results["text_labels"], results["scores"]):
        x0, y0, x1, y1 = map(int, box.tolist())
        detections.append({
            "label": label,
            "confidence": round(score.item(), 3),
            "bbox": [x0, y0, x1, y1]
        })

    return detections

def update_best_detection(thread_id: str, image_bytes: bytes, detections: list):
    """ Desa les dues millors deteccions d'imatge per aquest thread_id """
    if not detections:
        return

    best = max(detections, key=lambda d: d["confidence"])
    score = best["confidence"]

    # Inicialitzem si no existeix
    if thread_id not in best_detection_store:
        best_detection_store[thread_id] = [(score, image_bytes)]
        return

    current_list = best_detection_store[thread_id]

    # Afegim la nova detecció
    current_list.append((score, image_bytes))

    # Ens quedem només amb les dues millors
    current_list = sorted(current_list, key=lambda x: x[0], reverse=True)[:2]
    best_detection_store[thread_id] = current_list

def get_best_images(thread_id: str):
    """ Retorna una llista amb les dues millors imatges (bytes) """
    return [img for _, img in best_detection_store.get(thread_id, [])]