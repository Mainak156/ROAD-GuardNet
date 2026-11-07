import base64
import logging
import os
import threading
from collections import Counter, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from .detector import RoadDamageDetector

APP_TITLE = "ROAD-GuardNet API"
logger = logging.getLogger("road_guardnet")

def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default

MAX_UPLOAD_MB = _positive_int_env("MAX_UPLOAD_MB", 12)
MAX_FILE_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_IMAGE_PIXELS = _positive_int_env("MAX_IMAGE_PIXELS", 20_000_000)
HISTORY_LIMIT = _positive_int_env("HISTORY_LIMIT", 100)
JPEG_QUALITY = min(95, max(60, _positive_int_env("OUTPUT_JPEG_QUALITY", 85)))

# Keep OpenCV from creating a large pool of CPU threads on small instances.
try:
    cv2.setNumThreads(1)
except Exception:
    logger.debug("Could not configure OpenCV thread count.", exc_info=True)

detector = RoadDamageDetector()

# Keep only a bounded number of detailed records while maintaining session totals.
session_history = deque(maxlen=HISTORY_LIMIT)
class_counts = Counter()
session_totals = {
    "images_processed": 0,
    "total_detections": 0,
    "inference_time_ms_total": 0.0,
}
history_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model loading happens once per process, not once per request.
    loaded = await run_in_threadpool(detector.load)
    if loaded:
        logger.info("ROAD-GuardNet model loaded successfully.")
    else:
        logger.error("ROAD-GuardNet model could not be loaded: %s", detector.load_error)
    yield


app = FastAPI(
    title=APP_TITLE,
    version="1.1.0",
    description="Road damage detection API for D00, D10, D20 and D40.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"application": APP_TITLE, "status": "online", "docs": "/docs"}


@app.get("/health")
def health():
    ready = detector.ready
    return {
        "status": "ready" if ready else "model_not_loaded",
        "model_ready": ready,
        "weights_path": detector.weights_path,
        "message": "Model loaded." if ready else (detector.load_error or "Model is not loaded."),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence: float = Query(0.25, ge=0.01, le=0.99),
    use_cafa: bool = Query(True),
):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPG, PNG, or WEBP image.",
        )

    try:
        # Read at most one byte beyond the limit to avoid unbounded uploads in RAM.
        raw = await file.read(MAX_FILE_BYTES + 1)
    finally:
        await file.close()

    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the {MAX_UPLOAD_MB} MB upload limit.",
        )

    if not detector.ready:
        raise HTTPException(
            status_code=503,
            detail=detector.load_error or "Model weights are not available.",
        )

    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image. Please upload a valid JPG, PNG, or WEBP image.",
        )

    height, width = image.shape[:2]
    if height * width > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=f"Image dimensions are too large. Maximum allowed is {MAX_IMAGE_PIXELS:,} pixels.",
        )

    try:
        # Crucial: run CPU-heavy YOLO inference outside the async event loop.
        # The detector serializes access to its single model instance.
        result = await run_in_threadpool(
            detector.predict,
            image,
            confidence=confidence,
            use_cafa=use_cafa,
        )
    except Exception:
        logger.exception("ROAD-GuardNet inference failed.")
        raise HTTPException(
            status_code=500,
            detail="Inference failed. Check the Render application logs for details.",
        )

    annotated_image = result.pop("annotated_image", None)
    if annotated_image is None:
        raise HTTPException(status_code=500, detail="The detector did not return an annotated image.")

    encoded_ok, encoded = cv2.imencode(
        ".jpg",
        annotated_image,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
    )
    if not encoded_ok:
        raise HTTPException(status_code=500, detail="Could not encode the annotated image.")

    timestamp = datetime.now(timezone.utc).isoformat()
    filename = file.filename or "uploaded_image"

    result.update({
        "annotated_image_base64": base64.b64encode(encoded).decode("ascii"),
        "filename": filename,
        "timestamp": timestamp,
        "image_width": int(width),
        "image_height": int(height),
    })

    history_item = {
        "timestamp": timestamp,
        "filename": filename,
        "total_detections": int(result["total_detections"]),
        "inference_time_ms": float(result["inference_time_ms"]),
        "classes": [item["class_name"] for item in result["detections"]],
    }

    with history_lock:
        session_history.append(history_item)
        session_totals["images_processed"] += 1
        session_totals["total_detections"] += history_item["total_detections"]
        session_totals["inference_time_ms_total"] += history_item["inference_time_ms"]
        class_counts.update(history_item["classes"])

    # Drop large temporary references before returning the JSON response.
    del annotated_image, encoded, image, raw
    return result


@app.get("/stats")
def stats():
    with history_lock:
        processed = session_totals["images_processed"]
        average_ms = (
            session_totals["inference_time_ms_total"] / processed
            if processed else 0.0
        )
        return {
            "images_processed": processed,
            "total_detections": session_totals["total_detections"],
            "average_inference_time_ms": round(average_ms, 2),
            "class_counts": dict(class_counts),
            "history": list(session_history),
        }


@app.post("/reload-model")
async def reload_model():
    # RoadDamageDetector.load() is idempotent: it will not create a second model
    # when one is already loaded. This endpoint only retries a failed initial load.
    loaded = await run_in_threadpool(detector.load)
    if not loaded:
        raise HTTPException(status_code=503, detail=detector.load_error or "Model load failed.")
    return {"status": "ready", "message": "Model is loaded."}
