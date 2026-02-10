import cv2
import numpy as np
from collections import deque

class DeepFakeDetector:
    def __init__(self, cnn_base, video_model):
        self.cnn_base = cnn_base
        self.video_model = video_model

        self.seq_len = 10
        self.buffer = deque(maxlen=self.seq_len)
        self.pred_history = deque(maxlen=20)

        self.fake_streak = 0

        self.FAKE_THRESHOLD = 0.6
        self.FAKE_TRIGGER = 5
        self.SUSPICIOUS_TRIGGER = 2

    def analyze_frame(self, frame):
        try:
            # Resize frame for Xception
            resized = cv2.resize(frame, (299, 299))
            resized = resized.astype("float32") / 255.0

            self.buffer.append(resized)

            # Not enough frames yet
            if len(self.buffer) < self.seq_len:
                return "COLLECTING", 0.0

            frames_array = np.array(self.buffer)

            # CNN feature extraction
            cnn_features = self.cnn_base.predict(frames_array, verbose=0)

            # LSTM inference
            lstm_input = np.expand_dims(cnn_features, axis=0)
            probs = self.video_model.predict(lstm_input, verbose=0)

            avg_prob = float(probs.mean())
            self.pred_history.append(avg_prob)

            # ---- Stabilization logic ----
            if avg_prob > self.FAKE_THRESHOLD:
                self.fake_streak += 1
            else:
                self.fake_streak = max(0, self.fake_streak - 1)

            if self.fake_streak >= self.FAKE_TRIGGER:
                status = "FAKE"
            elif self.fake_streak >= self.SUSPICIOUS_TRIGGER:
                status = "SUSPICIOUS"
            else:
                status = "REAL"

            stable_prob = np.mean(self.pred_history)

            confidence = (
                stable_prob if status == "FAKE"
                else (1 - stable_prob)
            ) * 100

            return status, round(confidence, 2)

        except Exception as e:
            print("❌ Detector error:", e)
            return "ERROR", 0.0
