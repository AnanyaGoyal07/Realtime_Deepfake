import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

# ================= CONFIG =================
SEQ_LEN = 10
PREDICT_EVERY = 5
FAKE_THRESHOLD = 0.6
FAKE_TRIGGER = 5
SUSPICIOUS_TRIGGER = 2
# ==========================================

# Load models
cnn_base = load_model("models/cnn_base.keras")
video_model = load_model("models/video_lstm_best.keras")

print("✅ Models loaded")

# Buffers
frame_buffer = deque(maxlen=SEQ_LEN)
pred_history = deque(maxlen=20)

fake_streak = 0
frame_count = 0
current_state = "REAL"
last_confidence = 0.0
color = (0, 255, 0)

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Webcam not accessible")

print("🎥 Webcam started")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Resize for Xception
    resized = cv2.resize(frame, (299, 299))
    frame_buffer.append(resized)

    if len(frame_buffer) == SEQ_LEN and frame_count % PREDICT_EVERY == 0:
        frames_array = np.array(frame_buffer, dtype=np.float32) / 255.0

        cnn_features = cnn_base.predict(frames_array, verbose=0)
        lstm_input = np.expand_dims(cnn_features, axis=0)
        probs = video_model.predict(lstm_input, verbose=0)

        avg_prob = probs.mean()
        pred_history.append(avg_prob)

        if avg_prob > FAKE_THRESHOLD:
            fake_streak += 1
        else:
            fake_streak = max(0, fake_streak - 1)

        if fake_streak >= FAKE_TRIGGER:
            current_state = "FAKE"
            color = (0, 0, 255)
        elif fake_streak >= SUSPICIOUS_TRIGGER:
            current_state = "SUSPICIOUS"
            color = (0, 255, 255)
        else:
            current_state = "REAL"
            color = (0, 255, 0)

        stable_prob = np.mean(pred_history)
        last_confidence = (
            stable_prob * 100
            if current_state == "FAKE"
            else (1 - stable_prob) * 100
        )

    cv2.rectangle(frame, (10, 10), (420, 110), (0, 0, 0), -1)
    cv2.putText(frame, f"Status: {current_state}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Confidence: {last_confidence:.2f}%", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Real-Time DeepFake Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()