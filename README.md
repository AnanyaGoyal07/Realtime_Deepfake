Deepfake environment:
deepfake_env/Scripts/activate

Terminal 1:
uvicorn api.server:app --host 127.0.0.1 --port 8000

Terminal 2:
python zoom_virtual_cam.py
