import os
os.environ["USE_TF"] = "0"

import streamlit as st
import cv2
import tempfile
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# 1. Page Configuration & Dark Mode UI
st.set_page_config(page_title="DeepFake Detector", page_icon="🕵️‍♂️", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #0E0E10; color: #E2E2E2; }
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #FF4B4B, #9B72CB, #4285F4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', sans-serif;
    }
    .metric-box { background-color: #1A1A1E; padding: 20px; border-radius: 10px; border: 1px solid #3C4043; text-align: center; margin-top: 20px;}
    .fake-alert { color: #FF4B4B; font-weight: bold; font-size: 24px;}
    .real-alert { color: #34A853; font-weight: bold; font-size: 24px;}
</style>
""", unsafe_allow_html=True)

# 2. Explicitly Load Model Components (Bypassing pipeline wrapper bugs)
@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained("dima806/deepfake_vs_real_image_detection")
    model = AutoModelForImageClassification.from_pretrained("dima806/deepfake_vs_real_image_detection")
    return processor, model

processor, model = load_model()

# 3. Dashboard Layout
st.title("🕵️‍♂️ DeepFake Forensic Analysis")
st.markdown("Upload a video or image. The AI will extract frames and analyze microscopic artifacts, lighting inconsistencies, and blending errors to determine authenticity.")
st.markdown("---")

uploaded_file = st.file_uploader("Upload Media...", type=["mp4", "jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    
    if st.button("Run Forensic Analysis ✦", use_container_width=True):
        with st.spinner("Initializing neural network and extracting features..."):
            try:
                frames_to_analyze = []
                
                # Handle Video Processing
                if file_extension == "mp4":
                    # Save uploaded video to a temporary file for OpenCV to read
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_file.read())
                    
                    cap = cv2.VideoCapture(tfile.name)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    # Extract 5 evenly spaced frames across the video
                    step = max(1, frame_count // 5)
                    current_frame = 0
                    
                    while cap.isOpened() and len(frames_to_analyze) < 5:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if current_frame % step == 0:
                            # Convert BGR (OpenCV) to RGB (Pillow)
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames_to_analyze.append(Image.fromarray(frame))
                        current_frame += 1
                        
                    cap.release()
                    os.unlink(tfile.name) # Clean up temp file
                    
                    st.image(frames_to_analyze[0], caption="Sample Extracted Frame", width="stretch")
                
                # Handle Image Processing
                else:
                    image = Image.open(uploaded_file).convert('RGB')
                    frames_to_analyze.append(image)
                    st.image(image, caption="Uploaded Media", width="stretch")

                # Inference Engine
                fake_scores = []
                for img in frames_to_analyze:
                    inputs = processor(images=img, return_tensors="pt")
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                        predicted_class_idx = logits.argmax(-1).item()
                        label = model.config.id2label[predicted_class_idx]
                        
                        # Assuming the model outputs labels like 'fake' or 'real'
                        if "fake" in label.lower():
                            fake_scores.append(1)
                        else:
                            fake_scores.append(0)
                
                # Calculate final consensus
                manipulation_percentage = (sum(fake_scores) / len(fake_scores)) * 100
                
                if manipulation_percentage >= 50:
                    result_html = f'<div class="fake-alert">⚠️ DEEPFAKE DETECTED ({manipulation_percentage:.1f}%)</div>'
                else:
                    result_html = f'<div class="real-alert">✅ AUTHENTIC MEDIA ({(100 - manipulation_percentage):.1f}%)</div>'
                
                st.markdown(f"""
                <div class="metric-box">
                    <h3 style="margin:0;">Forensic Result</h3>
                    <br>
                    {result_html}
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error during analysis: {e}")