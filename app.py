from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer


weights_path = './yolov11-seg2.pt'
model = YOLO(weights_path)
print("fatto")
webrtc_streamer(key="sample")
