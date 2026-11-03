import cv2
import numpy as np

def cafa_lite(image_bgr):
    """Condition-aware image-level brightness/contrast correction."""
    img = image_bgr.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if float(gray.mean()) < 85:
        gamma = 0.75
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).clip(0, 255).astype("uint8")
        img = cv2.LUT(img, lut)
    if float(gray.std()) < 42:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return img

def draw_detections(image_bgr, detections):
    output = image_bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        label = f'{d["class_name"]} {d["confidence"]:.2f}'
        cv2.rectangle(output, (x1, y1), (x2, y2), (40, 210, 80), 2)
        cv2.putText(output, label, (x1, max(20, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 210, 80), 2)
    return output
