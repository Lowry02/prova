from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO
import av

# Load the YOLOv8 model for Object Detection
model = YOLO("yolov8n.pt")

# Function to process each frame of the video stream
def process_frame(frame):
    # Read image from the frame with PyAV
    img = frame.to_ndarray(format="bgr24")

    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model.track(img, tracker="bytetrack.yaml")

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Return the annotated frame
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# Create a WebRTC video streamer with the process_frame callback
webrtc_streamer(key="streamer", video_frame_callback=process_frame, sendback_audio=False,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
                mode=WebRtcMode.SENDRECV,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                }
               )
