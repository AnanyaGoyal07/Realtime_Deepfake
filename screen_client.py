import cv2
import numpy as np
import requests
import mss

API_URL = "http://127.0.0.1:8000/analyze-frame"

print("🖥️ Screen capture started")

with mss.mss() as sct:

    monitor = sct.monitors[1]  # full screen

    while True:
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)

        # Convert BGRA → BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Resize (faster processing)
        small_frame = cv2.resize(frame, (640, 480))

        # Encode
        success, img_encoded = cv2.imencode(".jpg", small_frame)
        if not success:
            continue

        try:
            response = requests.post(
                API_URL,
                files={"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")},
                timeout=5
            )

            result = response.json()

        except Exception as e:
            print("❌ Error:", e)
            continue

        status = result.get("status", "...")
        confidence = result.get("confidence", 0.0)

        # Colors
        color = (0, 255, 0)
        if status == "FAKE":
            color = (0, 0, 255)
        elif status == "SUSPICIOUS":
            color = (0, 255, 255)

        # Overlay
        cv2.rectangle(frame, (20, 20), (500, 120), (0, 0, 0), -1)
        cv2.putText(frame, f"Status: {status}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}%", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Screen DeepFake Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()