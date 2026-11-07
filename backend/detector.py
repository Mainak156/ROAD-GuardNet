import gc
import logging
import os
import threading
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from .modules import cafa_lite, draw_detections

logger = logging.getLogger("road_guardnet.detector")

# Reduce CPU thread oversubscription on small CPU-only hosting instances.
try:
    torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "1"))))
except (RuntimeError, ValueError):
    logger.debug("Could not set PyTorch intra-op thread count.", exc_info=True)

try:
    torch.set_num_interop_threads(max(1, int(os.getenv("TORCH_NUM_INTEROP_THREADS", "1"))))
except (RuntimeError, ValueError):
    # PyTorch only permits setting this before parallel work starts.
    logger.debug("Could not set PyTorch inter-op thread count.", exc_info=True)

try:
    cv2.setNumThreads(1)
except Exception:
    logger.debug("Could not set OpenCV thread count.", exc_info=True)


class RoadDamageDetector:
    """One YOLO model instance with serialized, inference-only prediction."""

    def __init__(self, weights_path=None, conf=0.25, imgsz=None, device=None):
        self.weights_path = str(
            weights_path or os.getenv("MODEL_PATH", "weights/best.pt")
        )
        self.conf = float(conf)
        self.imgsz = int(imgsz or os.getenv("MODEL_IMGSZ", "640"))
        # Render's ordinary web instances are CPU-only. Override only when needed.
        self.device = str(device or os.getenv("MODEL_DEVICE", "cpu"))
        self.model = None
        self.load_error = None
        self.class_names = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
        self._model_lock = threading.Lock()

    def load(self):
        """Load the model once. Repeated calls do not allocate another model."""
        with self._model_lock:
            if self.model is not None:
                return True

            path = Path(self.weights_path)
            if not path.is_file():
                self.load_error = f"Model weights not found: {path.resolve()}"
                logger.error(self.load_error)
                return False

            try:
                candidate = YOLO(str(path))
                names = candidate.names

                if isinstance(names, dict):
                    class_names = {int(key): str(value) for key, value in names.items()}
                elif isinstance(names, (list, tuple)):
                    class_names = {index: str(value) for index, value in enumerate(names)}
                else:
                    class_names = dict(self.class_names)

                self.model = candidate
                self.class_names = class_names
                self.load_error = None
                logger.info(
                    "Loaded YOLO weights from %s (device=%s, imgsz=%s).",
                    path,
                    self.device,
                    self.imgsz,
                )
                return True

            except Exception as exc:
                self.model = None
                self.load_error = f"{type(exc).__name__}: {exc}"
                logger.exception("Failed to load YOLO model.")
                return False

    @property
    def ready(self):
        return self.model is not None

    def predict(self, image_bgr, confidence=None, use_cafa=True):
        """
        Run one prediction. The lock prevents simultaneous calls from creating
        overlapping inference workloads against the same model instance.
        """
        with self._model_lock:
            model = self.model
            if model is None:
                raise RuntimeError(self.load_error or "Model is not loaded.")

            start = time.perf_counter()
            processed = None
            results = None
            annotated = None

            try:
                processed = cafa_lite(image_bgr) if use_cafa else image_bgr.copy()

                with torch.inference_mode():
                    results = model.predict(
                        source=processed,
                        conf=float(confidence if confidence is not None else self.conf),
                        imgsz=self.imgsz,
                        device=self.device,
                        verbose=False,
                        save=False,
                        stream=False,
                        augment=False,
                        max_det=100,
                    )

                detections = []
                if results:
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        xyxy = boxes.xyxy.detach().cpu().numpy()
                        class_ids = boxes.cls.detach().cpu().numpy().astype("int32")
                        scores = boxes.conf.detach().cpu().numpy()

                        for coords, class_id, score in zip(xyxy, class_ids, scores):
                            class_id = int(class_id)
                            detections.append({
                                "class_id": class_id,
                                "class_name": self.class_names.get(class_id, str(class_id)),
                                "confidence": round(float(score), 5),
                                "bbox": [round(float(value), 2) for value in coords],
                            })

                elapsed_ms = (time.perf_counter() - start) * 1000.0
                annotated = draw_detections(processed, detections)

                return {
                    "detections": detections,
                    "total_detections": len(detections),
                    "inference_time_ms": round(elapsed_ms, 2),
                    "annotated_image": annotated,
                }

            finally:
                # Ultralytics results contain tensors and image references.
                # Release request-local references after each inference.
                results = None
                processed = None
                gc.collect()
