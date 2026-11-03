
from ultralytics import YOLO
import cv2
from pathlib import Path

MODEL_PATH = "best.pt"
IMAGE_PATH = "Testing_images\\T2.jpg"

model = YOLO(MODEL_PATH)

print("Model classes:", model.names)

# Test with a very low confidence threshold
results = model.predict(
    source=IMAGE_PATH,
    imgsz=640,
    conf=0.01,
    iou=0.5,
    save=True,
    verbose=True
)

result = results[0]

print("\nNumber of detections:", len(result.boxes))

for box in result.boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])

    print(
        "Class:", model.names[class_id],
        "| Confidence:", round(confidence, 4)
    )

print("\nAnnotated image saved in runs/detect/predict/")
