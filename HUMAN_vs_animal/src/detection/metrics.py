import time

class InferenceMetrics:
    def __init__(self):
        self.total_time = 0.0
        self.frames = 0
        self.total_detections = 0

    def update(self, inference_time, detections_count):
        self.total_time += inference_time
        self.frames += 1
        self.total_detections += detections_count

    def summary(self):
        avg_time = self.total_time / max(1, self.frames)
        fps = 1.0 / avg_time if avg_time > 0 else 0

        return {
            "frames_processed": self.frames,
            "avg_inference_time_sec": round(avg_time, 4),
            "fps": round(fps, 2),
            "avg_detections_per_frame": round(
                self.total_detections / max(1, self.frames), 2
            )
        }
