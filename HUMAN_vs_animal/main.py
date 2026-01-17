import os
import cv2
import time

from src.detection.model import load_detector
from src.detection.inference import detect_objects
from src.detection.visulaization import draw_detections
from src.detection.metrics import InferenceMetrics

TEST_IMAGES_DIR = "test_images"
OUTPUT_IMAGES_DIR = "outputs/images"

os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)

def main():
    model = load_detector()
    metrics = InferenceMetrics()

    for img_name in os.listdir(TEST_IMAGES_DIR):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(TEST_IMAGES_DIR, img_name)
        image = cv2.imread(img_path)

        start = time.time()
        detections = detect_objects(model, image)
        end = time.time()

        metrics.update(end - start, len(detections))

        result = draw_detections(image, detections)
        cv2.imwrite(
            os.path.join(OUTPUT_IMAGES_DIR, f"annotated_{img_name}"),
            result
        )

    print("\n=== INFERENCE METRICS ===")
    for k, v in metrics.summary().items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
