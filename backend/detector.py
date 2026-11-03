import os
import time
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO
from .modules import cafa_lite, draw_detections

class RoadDamageDetector:
    def __init__(self, weights_path=None, conf=0.25, imgsz=640, device=None):
        self.weights_path = weights_path or os.getenv("MODEL_PATH", "weights/best.pt")
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.device = device or os.getenv("MODEL_DEVICE", "0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.load_error = None
        self.class_names = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}

    def load(self):
        path = Path(self.weights_path)
        if not path.exists():
            self.load_error = f"Model weights not found: {path.resolve()}"
            return False
        try:
            self.model = YOLO(str(path))
            names = self.model.names
            if isinstance(names, dict):
                self.class_names = {int(k): str(v) for k, v in names.items()}
            elif isinstance(names, list):
                self.class_names = {i: str(v) for i, v in enumerate(names)}
            self.load_error = None
            return True
        except Exception as exc:
            self.load_error = str(exc)
            self.model = None
            return False

    @property
    def ready(self):
        return self.model is not None

    def predict(self, image_bgr, confidence=None, use_cafa=True):
        if not self.ready:
            raise RuntimeError(self.load_error or "Model is not loaded.")
        start = time.perf_counter()
        processed = cafa_lite(image_bgr) if use_cafa else image_bgr.copy()
        results = self.model.predict(
            source=processed,
            conf=float(confidence if confidence is not None else self.conf),
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )
        detections = []
        result = results[0]
        if result.boxes is not None:
            for box in result.boxes:
                coords = box.xyxy[0].detach().cpu().numpy().tolist()
                cls_id = int(box.cls[0].detach().cpu())
                score = float(box.conf[0].detach().cpu())
                detections.append({
                    "class_id": cls_id,
                    "class_name": self.class_names.get(cls_id, str(cls_id)),
                    "confidence": round(score, 5),
                    "bbox": [round(float(v), 2) for v in coords]
                })
        elapsed_ms = (time.perf_counter() - start) * 1000
        annotated = draw_detections(processed, detections)
        return {
            "detections": detections,
            "total_detections": len(detections),
            "inference_time_ms": round(elapsed_ms, 2),
            "annotated_image": annotated
        }
