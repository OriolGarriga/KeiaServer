from fastapi import APIRouter, UploadFile, File
from ultralytics import YOLO
from deep_translator import GoogleTranslator
import cv2
import numpy as np

router = APIRouter()
model = YOLO("models/yolov8n.pt") 
model.fuse()

def translate_to_catalan(text):
    try:
        return GoogleTranslator(source='en', target='ca').translate(text)
    except Exception:
        return text 

@router.post("/detect_objects")
async def detect_objects(image: UploadFile = File(...)):
    contents = await image.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    height, width, _ = img.shape
    results = model(img)[0]

    detections = []
    for box in results.boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label_en = model.names[cls_id]
        label_ca = translate_to_catalan(label_en)

        x1, y1, x2, y2 = box.xyxy[0]
        x = float(x1) / width
        y = float(y1) / height
        w = float(x2 - x1) / width
        h = float(y2 - y1) / height

        detections.append({
            "label": label_ca,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "confidence": conf
        })

    return {"detections": detections}
