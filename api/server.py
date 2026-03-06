import os
from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import traceback

from api.utils import DeepFakeDetector
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import Xception

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CNN_PATH = os.path.join(BASE_DIR, "models", "cnn_base.keras")
LSTM_PATH = os.path.join(BASE_DIR, "models", "video_lstm_best.keras")

print("Loading CNN model...")
cnn_base = Xception(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(299, 299, 3)
)

print("Loading LSTM model...")
video_model = load_model(LSTM_PATH)

print("Initializing detector...")
detector = DeepFakeDetector(
    cnn_base=cnn_base,
    video_model=video_model
)

@app.get("/")
def home():
    return {"message": "DeepFake Detection API Running"}

@app.post("/analyze-frame")
async def analyze_frame(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        np_img = np.frombuffer(contents, np.uint8)

        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if frame is None:
            return {
                "status": "ERROR",
                "confidence": 0.0
            }

        status, confidence = detector.analyze_frame(frame)

        return {
            "status": status,
            "confidence": confidence
        }

    except Exception:
        print(traceback.format_exc())

        return {
            "status": "ERROR",
            "confidence": 0.0
        }