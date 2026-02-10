import cv2
import requests

API_URL = "http://127.0.0.1:8000/analyze-frame"

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Webcam not accessible")

print("🎥 Webcam client started")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Encode frame as JPEG
    success, img_encoded = cv2.imencode(".jpg", frame)
    if not success:
        continue

    files = {
        "file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")
    }

    try:
        response = requests.post(API_URL, files=files, timeout=2)

        if response.status_code != 200:
            print("❌ API Error:", response.text)
            continue

        result = response.json()

    except Exception as e:
        print("❌ Request failed:", e)
        continue

    status = result.get("status", "COLLECTING")
    confidence = result.get("confidence", 0.0)

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

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
