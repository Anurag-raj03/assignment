import time
import json

class OCRMetrics:
    def __init__(self):
        self.start = time.time()
        self.blocks = 0

    def update(self, count):
        self.blocks = count

    def summary(self):
        return {
            "text_blocks_detected": self.blocks,
            "total_inference_time_sec": round(time.time() - self.start, 3)
        }

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=4)
