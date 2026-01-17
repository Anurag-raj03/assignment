import cv2
import torch
from torchvision import transforms
from src.detection.model import DEVICE

SCORE_THRESHOLD = 0.6
VALID_LABELS = [1, 16, 17, 18, 19, 20, 21]  # person + animals

transform = transforms.Compose([
    transforms.ToTensor()
])

def detect_objects(model, image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image_rgb).to(DEVICE)

    with torch.no_grad():
        outputs = model([image_tensor])[0]

    detections = []
    for box, score, label in zip(
        outputs["boxes"].cpu().numpy(),
        outputs["scores"].cpu().numpy(),
        outputs["labels"].cpu().numpy()
    ):
        if score >= SCORE_THRESHOLD and label in VALID_LABELS:
            detections.append({
                "box": box.astype(int),
                "score": float(score),
                "label": int(label)
            })

    return detections
