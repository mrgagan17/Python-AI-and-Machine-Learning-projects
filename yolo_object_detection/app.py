import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# 1. Page Configuration
st.set_page_config(page_title="YOLOv8 Object Detection", page_icon="👁️", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #0E0E10; color: #E2E2E2; }
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #34A853, #4285F4, #9B72CB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', sans-serif;
    }
    .metric-card { background-color: #1A1A1E; padding: 15px; border-radius: 12px; border: 1px solid #3C4043; text-align: center; margin-bottom: 20px;}
</style>
""", unsafe_allow_html=True)

# 2. Load YOLOv8 Model
@st.cache_resource
def load_yolo():
    # Automatically downloads the YOLOv8 Nano weights (~6MB) if not present locally
    return YOLO("yolov8n.pt")

model = load_yolo()

# 3. UI Layout
st.title("👁️ Real-Time YOLOv8 Detection")
st.markdown("Powered by **Ultralytics YOLOv8**. Toggle the camera to initialize real-time deep learning inference. The model can identify 80 different COCO classes (people, cars, phones, laptops) simultaneously.")
st.markdown("---")

col1, col2 = st.columns([1, 3])
with col1:
    st.markdown('<div class="metric-card">Hardware Access</div>', unsafe_allow_html=True)
    run_webcam = st.checkbox("🟢 Start Webcam")
    confidence_threshold = st.slider("Confidence", min_value=0.1, max_value=1.0, value=0.4, step=0.05)

with col2:
    frame_window = st.image([])

# 4. Real-Time Processing Loop
if run_webcam:
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        st.error("Error: Could not access the webcam.")
    else:
        while run_webcam:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to capture frame from camera.")
                break
                
            # Run YOLOv8 inference directly on the BGR frame
            # conf parameter filters out weak predictions based on your UI slider
            results = model(frame, conf=confidence_threshold, verbose=False)
            
            # Plot the bounding boxes and labels onto the frame
            annotated_frame = results[0].plot()
            
            # Convert BGR (OpenCV standard) to RGB (Streamlit standard)
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Update the UI
            frame_window.image(frame_rgb)
            
    camera.release()
else:
    frame_window.info("Webcam feed disabled. Check the box to start the YOLO engine.")