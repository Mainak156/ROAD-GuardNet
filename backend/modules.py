import cv2
import numpy as np


def cafa_lite(image_bgr):
    """Condition-aware image-level brightness and contrast correction."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("CAFA-Lite received an empty image.")

    img = image_bgr.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if float(gray.mean()) < 85:
        gamma = 0.75
        lookup = np.clip(
            (np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255.0,
            0,
            255,
        ).astype(np.uint8)
        img = cv2.LUT(img, lookup)

    if float(gray.std()) < 42:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        lightness = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        ).apply(lightness)
        img = cv2.cvtColor(
            cv2.merge((lightness, channel_a, channel_b)),
            cv2.COLOR_LAB2BGR,
        )

    return img


def draw_detections(image_bgr, detections):
    """Draw predicted bounding boxes and labels on a copy of the image."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Cannot annotate an empty image.")

    output = image_bgr.copy()
    height, width = output.shape[:2]

    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["bbox"])
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))

        label = f'{detection["class_name"]} {detection["confidence"]:.2f}'
        cv2.rectangle(output, (x1, y1), (x2, y2), (40, 210, 80), 2)
        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 210, 80),
            2,
            cv2.LINE_AA,
        )

    return output
