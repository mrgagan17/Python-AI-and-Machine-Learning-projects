import streamlit as st
import cv2
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Real-Time Face Detection", page_icon="👤", layout="centered")

# 2. Custom CSS for UI
st.markdown("""
<style>
    .stApp { background-color: #0E0E10; color: #E2E2E2; }
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #4285F4, #9B72CB, #D96570);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background: #1A1A1E;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #3C4043;
        margin-bottom: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("👤 Real-Time Face Detection")
st.markdown("Powered by **OpenCV** and **Haar Cascades**. Toggle the camera below to initialize real-time computer vision bounding boxes.")

# 3. Load Pre-trained Haar Cascade
@st.cache_resource
def load_cascade():
    # OpenCV includes pre-trained XML files for frontal face detection
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(cascade_path)

face_cascade = load_cascade()

# 4. Webcam UI Control
col1, col2 = st.columns([1, 3])
with col1:
    st.markdown('<div class="metric-card">Camera Control</div>', unsafe_allow_html=True)
    run_webcam = st.checkbox("🟢 Start Webcam")

with col2:
    # Placeholder to continuously update the video frames
    frame_window = st.image([])

# 5. Real-Time Processing Loop
if run_webcam:
    # Initialize local webcam (0 is usually the default built-in camera)
    camera = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not camera.isOpened():
        st.error("Error: Could not access the webcam. Ensure no other applications are using it.")
    else:
        while run_webcam:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to capture frame from camera.")
                break
                
            # OpenCV captures in BGR, Streamlit expects RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to grayscale for faster face detection math
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces: (image, scaleFactor, minNeighbors)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=6, minSize=(30, 30))
            
            # Draw bounding boxes around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (66, 133, 244), 3) # Google Blue Box
                cv2.putText(frame_rgb, 'Face Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (66, 133, 244), 2)

            # Push the annotated frame to the Streamlit UI
            frame_window.image(frame_rgb)
            
    # Release hardware when the checkbox is turned off
    camera.release()
else:
    frame_window.info("Webcam is currently disabled. Check the box to start detection.")