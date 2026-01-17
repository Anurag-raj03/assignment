import cv2
import numpy as np

def detect_text_regions(binary_image):
    # Dilate to connect broken stencil characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    dilated = cv2.dilate(binary_image, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    h_img, w_img = binary_image.shape[:2]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Stencil-friendly heuristics
        if (
            w > 0.15 * w_img and   # wide text lines
            h > 20 and h < 120     # stencil text height
        ):
            boxes.append((x, y, x + w, y + h))

    # sort top to bottom
    boxes = sorted(boxes, key=lambda b: b[1])
    return boxes
