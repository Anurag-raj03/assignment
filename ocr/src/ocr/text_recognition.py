import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-printed"
)
model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-printed"
).to(DEVICE)

model.eval()

def recognize_text(image, boxes):
    results = []

    for (x1, y1, x2, y2) in boxes:
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        pil_img = Image.fromarray(crop).convert("RGB")

        pixel_values = processor(
            pil_img, return_tensors="pt"
        ).pixel_values.to(DEVICE)

        with torch.no_grad():
            generated_ids = model.generate(pixel_values)

        text = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        if text:
            results.append({
                "text": text,
                "bbox": [x1, y1, x2, y2]
            })

    return results
