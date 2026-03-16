import cv2
import numpy as np
from collections import deque
import os
import csv
import time

# Face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

class DeepFakeDetector:
    def __init__(self, cnn_base, video_model):
        self.cnn_base = cnn_base
        self.video_model = video_model

        # Increased sequence length
        self.seq_len = 20
        self.feature_buffer = deque(maxlen=self.seq_len)

        # Prediction smoothing history
        self.pred_history = deque(maxlen=30)

        self.fake_streak = 0

        # Tuned thresholds
        self.FAKE_THRESHOLD = 0.7
        self.FAKE_TRIGGER = 6
        self.SUSPICIOUS_TRIGGER = 3

        # Create log directory
        os.makedirs("logs", exist_ok=True)

        # Create CSV file if not exists
        self.log_file = "logs/predictions.csv"
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "status", "confidence"])

    def crop_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            return frame

        x, y, w, h = faces[0]
        face = frame[y:y+h, x:x+w]

        return face

    def log_prediction(self, status, confidence):
        try:
            with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([time.time(), status, confidence])
        except:
            pass

    def analyze_frame(self, frame):
        try:
            # ---- Face cropping ----
            frame = self.crop_face(frame)

            # Resize for Xception
            resized = cv2.resize(frame, (299, 299))
            resized = resized.astype("float32") / 255.0

            # Extract CNN feature for ONLY the new frame
            feature = self.cnn_base.predict(
                 np.expand_dims(resized, axis=0),
                 verbose=0
            )[0]

            self.feature_buffer.append(feature)

            if len(self.feature_buffer) < self.seq_len:
                return "COLLECTING", 0.0

            features_array = np.array(self.feature_buffer)

            # LSTM prediction
            lstm_input = np.expand_dims(features_array, axis=0)
            probs = self.video_model.predict(lstm_input, verbose=0)

            avg_prob = float(probs.mean())

            # ---- Smoothing ----
            self.pred_history.append(avg_prob)
            stable_prob = float(np.mean(self.pred_history))

            # ---- Stabilization Logic ----
            if stable_prob > self.FAKE_THRESHOLD:
                self.fake_streak += 1
            else:
                self.fake_streak = max(0, self.fake_streak - 1)

            if self.fake_streak >= self.FAKE_TRIGGER:
                status = "FAKE"
            elif self.fake_streak >= self.SUSPICIOUS_TRIGGER:
                status = "SUSPICIOUS"
            else:
                status = "REAL"

            confidence = (
                stable_prob if status == "FAKE"
                else (1 - stable_prob)
            ) * 100

            confidence = round(confidence, 2)

            # ---- Log prediction ----
            self.log_prediction(status, confidence)

            return status, confidence

        except Exception as e:
            print("❌ Detector error:", e)
            return "ERROR", 0.0