
# 🧠 AI Technical Assignment

## Human & Animal Detection + Offline Industrial OCR

This project implements **two independent AI systems** designed for **offline deployment under real-world industrial constraints**.

* **Part A:** Human & Animal Detection in images and videos
* **Part B:** Offline OCR for industrial / military-style stenciled text

The emphasis of this assignment is on **system design, robustness, explainability, and realistic limitations**, rather than benchmark-driven accuracy.

---

## 📁 Project Structure

```
project/
├── src/
│   ├── detection/              # Part A – Object Detection
│   │   ├── model.py
│   │   ├── inference.py
│   │   ├── visualization.py
│   │   └── metrics.py
│   │
│   └── ocr/                    # Part B – Offline OCR
│       ├── preprocessing.py
│       ├── text_detection.py
│       ├── text_recognition.py
│       ├── postprocess.py
│       └── metrics.py
│
├── test_images/                # Part A image inputs
├── test_videos/                # Part A video inputs
├── outputs/                    # Part A outputs + metrics
│
├── ocr_inputs/                 # Part B OCR inputs
├── ocr_outputs/                # Part B OCR outputs + metrics
│
├── main.py                     # Entry point – Part A
├── main_ocr.py                 # Entry point – Part B
├── requirements.txt
└── README.md
```

---

# 🔹 Part A – Human & Animal Detection

## 🎯 Objective

Detect and classify **humans and animals** in images/videos using a **robust, offline pipeline** without relying on cloud services.

---

## 📊 Dataset

**Open Images V7 (Filtered Subset)**

* Chosen for real-world diversity and high-quality annotations
* Classes used:

  * `Person`
  * Animals: Dog, Cat, Horse, Cattle, Sheep, Bird
* Dataset downloaded once using FiftyOne and then used fully offline

---

## 🧠 Model

* **Faster R-CNN (ResNet-50 FPN)**
* Pretrained on COCO
* Used strictly in **inference-only mode**
* YOLO explicitly avoided (as per instructions)

### Why Faster R-CNN?

* Strong localization accuracy
* Industry-standard architecture
* Reliable offline performance

---

## 🔄 Detection Pipeline

```
Image / Video
   ↓
Faster R-CNN Detector
   ↓
Class Filtering (Person + Animals only)
   ↓
Semantic Grouping:
   - Person → Human
   - All animals → Animal
   ↓
Annotated Output
```

All videos placed in `test_videos/` are processed automatically.

---

## 📈 Metrics (Part A)

Since the model is not fine-tuned, **training metrics (loss, mAP)** are not recomputed.

Instead, **pipeline-level metrics** are recorded:

* Frames processed
* Average inference time per frame
* FPS
* Average detections per frame

Saved to:

```
outputs/metrics.json
```

Example:

```json
{
  "frames_processed": 49,
  "avg_inference_time_sec": 3.43,
  "fps": 0.29,
  "avg_detections_per_frame": 5.29
}
```

> The low FPS is expected for Faster R-CNN running on CPU and reflects a deliberate trade-off favoring robustness over real-time speed.

---

# 🔹 Part B – Offline OCR for Industrial / Stenciled Text

## 🎯 Objective

Extract text from **industrial and military-style containers** featuring:

* Stenciled paint
* Low contrast
* Surface wear
* Broken characters

The system must be **fully offline** and output **structured text data**.

---

## 🧠 OCR Design Philosophy

Industrial OCR is fundamentally different from document OCR.
This system prioritizes:

* Robustness over accuracy
* Safe failure (no hallucinated text)
* Explainable behavior

---

## 🧰 OCR Model

### Text Recognition

* **Microsoft TrOCR (Hugging Face)**
* Model: `microsoft/trocr-base-printed`
* Transformer-based OCR
* Used **locally, offline**
* No fine-tuning performed

### Why TrOCR?

* Significantly more robust than classical OCR on degraded text
* Handles broken glyphs better than Tesseract
* Fully open-source and offline

---

## 🔄 OCR Pipeline

```
Input Image
   ↓
Preprocessing (grayscale, contrast enhancement, thresholding)
   ↓
Stencil-aware text region detection
   ↓
Crop detected regions
   ↓
TrOCR (offline recognition)
   ↓
Post-processing & cleanup
   ↓
Structured JSON output
```

---

## 📈 Metrics (Part B)

OCR performance is evaluated using **pipeline-level metrics**, not character-level accuracy.

Metrics include:

* Number of detected text blocks
* Total inference time

Saved to:

```
ocr_outputs/metrics.json
```

Example:

```json
{
  "text_blocks_detected": 4,
  "total_inference_time_sec": 1.12
}
```

---

## ⚠️ Known Limitations (Important)

Industrial OCR is inherently difficult.

This system may **intentionally return no OCR output** for images where:

* Contrast is extremely low
* Stencil paint is heavily worn
* Text blends into the background

This behavior is **by design** to avoid false positives.

> Returning no text is preferable to hallucinating incorrect text in industrial systems.

---

## ▶️ How to Run

### Part A – Detection

```bash
python main.py
```

### Part B – OCR

```bash
python main.py
```

---

## 📤 Outputs

### Part A

* Annotated images/videos → `outputs/`
* Metrics → `outputs/metrics.json`

### Part B

* Annotated image → `ocr_outputs/annotated.jpg`
* Extracted text → `ocr_outputs/result.json`
* Metrics → `ocr_outputs/metrics.json`

