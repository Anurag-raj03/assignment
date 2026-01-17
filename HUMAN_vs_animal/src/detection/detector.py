import torch
import torchvision
import cv2
from torchvision import transforms

# -------------------------------
# CONFIG
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCORE_THRESHOLD = 0.6

VALID_LABELS = [1, 16, 17, 18, 19, 20, 21]  # person + animals

# -------------------------------
# LOAD DETECTION MODEL
# -------------------------------
def load_detector():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT"
    )
    model.to(DEVICE)
    model.eval()
    return model

# -------------------------------
# IMAGE PREPROCESS
# -------------------------------
transform = transforms.Compose([
    transforms.ToTensor()
])

# -------------------------------
# RUN DETECTION
# -------------------------------
def detect_objects(model, image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image_rgb).to(DEVICE)

    with torch.no_grad():
        outputs = model([image_tensor])[0]

    boxes = outputs["boxes"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()

    detections = []
    for box, score, label in zip(boxes, scores, labels):
        if score >= SCORE_THRESHOLD and label in VALID_LABELS:
            detections.append({
                "box": box.astype(int),
                "score": float(score),
                "label": int(label)
            })

    return detections

# -------------------------------
# MAP LABELS
# -------------------------------
def classify_label(label_id):
    return "Human" if label_id == 1 else "Animal"

# -------------------------------
# DRAW
# -------------------------------
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

# -------------------------------
# TEST IMAGE
# -------------------------------
if __name__ == "__main__":
    import os

    os.makedirs("outputs/images", exist_ok=True)

    model = load_detector()
    img = cv2.imread(
        r"C:\Users\dell\fiftyone\open-images-v7\train\data\000c9121b17ee0ff.jpg"
    )

    detections = detect_objects(model, img)
    result = draw_detections(img, detections)

    cv2.imwrite("outputs/images/result1.jpg", result)
    print("[INFO] Detection complete")
