import cv2

def classify_label(label_id):
    return "Human" if label_id == 1 else "Animal"

def draw_detections(image, detections):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = classify_label(det["label"])
        score = det["score"]

        color = (0, 255, 0) if label == "Human" else (0, 0, 255)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"{label} {score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
    return image
