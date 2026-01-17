import cv2
import json
import os

from src.ocr.preprocessing import preprocess_image
from src.ocr.text_detection import detect_text_regions
from src.ocr.text_recognition import recognize_text
from src.ocr.postprocess import clean_results
from src.ocr.metrics import OCRMetrics

INPUT_DIR = "ocr_input"
OUTPUT_DIR = "ocr_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    metrics = OCRMetrics()

    for img_name in os.listdir(INPUT_DIR):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        image_path = os.path.join(INPUT_DIR, img_name)
        image = cv2.imread(image_path)

        binary = preprocess_image(image)
        boxes = detect_text_regions(binary)

        print(f"[DEBUG] Detected text regions: {len(boxes)}")

        raw_results = recognize_text(image, boxes)
        final_results = clean_results(raw_results)

        metrics.update(len(final_results))

        for r in final_results:
            x1, y1, x2, y2 = r["bbox"]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image, r["text"],
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )

        cv2.imwrite(
            os.path.join(OUTPUT_DIR, "annotated.jpg"),
            image
        )

        with open(os.path.join(OUTPUT_DIR, "result.json"), "w") as f:
            json.dump({"text_blocks": final_results}, f, indent=4)

    metrics.save(os.path.join(OUTPUT_DIR, "metrics.json"))

    print("\nOCR complete.")
    print(metrics.summary())

if __name__ == "__main__":
    main()
