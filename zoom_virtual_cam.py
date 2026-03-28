import cv2
import requests
import pyvirtualcam
import time

API_URL = "http://127.0.0.1:8000/analyze-frame"

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Webcam not accessible")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Starting virtual camera...")

with pyvirtualcam.Camera(width=width, height=height, fps=20) as cam:
    print(f"Virtual camera started: {cam.device}")

    last_status = "COLLECTING"
    last_confidence = 0.0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Send every 5th frame to API
        if frame_count % 5 == 0:
            try:
                success, img_encoded = cv2.imencode(".jpg", frame)
                if success:

                    files = {
                        "file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")
                    }

                    response = requests.post(API_URL, files=files, timeout=5)

                    if response.status_code == 200:
                        result = response.json()
                        last_status = result["status"]
                        last_confidence = result["confidence"]

            except:
                pass

        # Color logic
        color = (0,255,0)

        if last_status == "FAKE":
            color = (0,0,255)

        elif last_status == "SUSPICIOUS":
            color = (0,255,255)

        # Overlay detection info
        cv2.rectangle(frame,(10,10),(420,110),(0,0,0),-1)

        cv2.putText(
            frame,
            f"Status: {last_status}",
            (20,45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {last_confidence:.2f}%",
            (20,85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )

        # Convert to RGB for virtual camera
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cam.send(frame_rgb)
        cam.sleep_until_next_frame()

cap.release()