import cv2
import requests
import time

API_URL = "http://127.0.0.1:8000/analyze-frame"

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Webcam not accessible")

print("🎥 Webcam client started")

# -------- Control Variables --------
frame_count = 0
SEND_EVERY = 5           # send every 5th frame (reduces API load)
last_status = "COLLECTING"
last_confidence = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # -------- Only send some frames to API --------
    if frame_count % SEND_EVERY == 0:

        success, img_encoded = cv2.imencode(".jpg", frame)
        if success:

            files = {
                "file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")
            }

            try:
                response = requests.post(API_URL, files=files, timeout=10)

                if response.status_code == 200:
                    result = response.json()
                    last_status = result.get("status", last_status)
                    last_confidence = result.get("confidence", last_confidence)

                else:
                    print("❌ API Error:", response.text)

            except Exception as e:
                print("❌ Request failed:", e)

    # -------- Use LAST known result (non-blocking display) --------
    status = last_status
    confidence = last_confidence

    # 🎨 Color logic
    color = (0, 255, 0)
    if status == "FAKE":
        color = (0, 0, 255)
    elif status == "SUSPICIOUS":
        color = (0, 255, 255)

    # 🖼 Overlay
    cv2.rectangle(frame, (10, 10), (420, 110), (0, 0, 0), -1)
    cv2.putText(frame, f"Status: {status}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Confidence: {confidence:.2f}%", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Webcam → API DeepFake Detection", frame)

    # Small sleep prevents CPU overload
    time.sleep(0.01)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
