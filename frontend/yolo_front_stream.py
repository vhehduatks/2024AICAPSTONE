import streamlit as st
import requests
from PIL import Image
import cv2
import numpy as np
import base64
import time
import io

# FastAPI server address
FASTAPI_SERVER = "http://127.0.0.1:8000"

st.title("YOLO Real-Time Inference with Webcam")

# Placeholder for the video frames
frame_placeholder = st.empty()

# Initialize session state variables
if 'run' not in st.session_state:
    st.session_state['run'] = False
if 'cap' not in st.session_state:
    st.session_state['cap'] = None

def start_webcam():
    st.session_state['run'] = True
    st.session_state['cap'] = cv2.VideoCapture(0)
    if not st.session_state['cap'].isOpened():
        st.error("Unable to open webcam")
        st.session_state['run'] = False

def stop_webcam():
    st.session_state['run'] = False
    if st.session_state['cap'] is not None:
        st.session_state['cap'].release()
        st.session_state['cap'] = None

start_button = st.button("Start Webcam", on_click=start_webcam)
stop_button = st.button("Stop Webcam", on_click=stop_webcam)

if st.session_state['run']:
    # Read a single frame
    ret, frame = st.session_state['cap'].read()
    if not ret:
        st.error("Failed to read frame from camera")
        stop_webcam()
    else:
        # Encode frame to bytes
        retval, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Send frame to FastAPI server
        response = requests.post(
            f"{FASTAPI_SERVER}/process_image",
            files={"file": frame_bytes},
        )

        if response.status_code == 200:
            data = response.json()
            annotated_image_base64 = data.get("annotated_image")
            if annotated_image_base64:
                annotated_image_bytes = base64.b64decode(annotated_image_base64)
                annotated_image = Image.open(io.BytesIO(annotated_image_bytes))
                # Update the frame placeholder
                frame_placeholder.image(annotated_image, use_container_width=True)
        else:
            st.error("Error processing frame")

        # Sleep and rerun
        time.sleep(0.05)
        st.rerun()
else:
    st.write("Click 'Start Webcam' to begin")
