import base64
import io
import os
from collections import Counter
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .detector import RoadDamageDetector

APP_TITLE = "ROAD-GuardNet API"
MAX_FILE_BYTES = int(os.getenv("MAX_UPLOAD_MB", "12")) * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(title=APP_TITLE, version="1.0.0",
              description="Road damage detection API for D00, D10, D20 and D40.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

detector = RoadDamageDetector()
session_history = []

@app.on_event("startup")
def startup_event():
    detector.load()

@app.get("/")
def root():
    return {"application": APP_TITLE, "status": "online", "docs": "/docs"}

@app.get("/health")
def health():
    return {
        "status": "ready" if detector.ready else "model_not_loaded",
        "model_ready": detector.ready,
        "weights_path": detector.weights_path,
        "message": "Model loaded." if detector.ready else detector.load_error
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...),
                  confidence: float = Query(0.25, ge=0.01, le=0.99),
                  use_cafa: bool = Query(True)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, or WEBP image.")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_FILE_BYTES // (1024*1024)} MB.")
    if not detector.ready:
        raise HTTPException(status_code=503, detail=detector.load_error or "Model weights are not available yet.")

    image_array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Please upload a valid image.")

    try:
        result = detector.predict(image, confidence=confidence, use_cafa=use_cafa)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    ok, encoded = cv2.imencode(".jpg", result.pop("annotated_image"))
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode annotated image.")

    result["annotated_image_base64"] = base64.b64encode(encoded.tobytes()).decode("ascii")
    result["filename"] = file.filename or "uploaded_image"
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    result["image_width"] = int(image.shape[1])
    result["image_height"] = int(image.shape[0])

    session_history.append({
        "timestamp": result["timestamp"],
        "filename": result["filename"],
        "total_detections": result["total_detections"],
        "inference_time_ms": result["inference_time_ms"],
        "classes": [d["class_name"] for d in result["detections"]]
    })
    return result

@app.get("/stats")
def stats():
    class_counts = Counter()
    for item in session_history:
        class_counts.update(item["classes"])
    total = sum(x["total_detections"] for x in session_history)
    avg_ms = (sum(x["inference_time_ms"] for x in session_history) / len(session_history)
              if session_history else 0.0)
    return {
        "images_processed": len(session_history),
        "total_detections": total,
        "average_inference_time_ms": round(avg_ms, 2),
        "class_counts": dict(class_counts),
        "history": session_history[-100:]
    }

@app.post("/reload-model")
def reload_model():
    loaded = detector.load()
    if not loaded:
        raise HTTPException(status_code=503, detail=detector.load_error)
    return {"status": "ready", "message": "Model loaded successfully."}
