import streamlit as st
import librosa
import numpy as np
import joblib
import os

# 1. Page Configuration
st.set_page_config(page_title="AI Music Classifier", page_icon="🎵", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #0E0E10; color: #E2E2E2; }
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #4285F4, #9B72CB, #D96570);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', sans-serif;
    }
    .genre-card { background-color: #1A1A1E; padding: 20px; border-radius: 10px; border: 1px solid #3C4043; text-align: center; margin-top: 20px;}
</style>
""", unsafe_allow_html=True)

# 2. Load Model & Scaler
@st.cache_resource
def load_system():
    model_path = 'model/music_mlp_model.pkl'
    scaler_path = 'model/audio_scaler.pkl'
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("Model files not found! Run 'python train_model.py' first.")
        st.stop()
    return joblib.load(model_path), joblib.load(scaler_path)

model, scaler = load_system()
genres = {0: "Rock 🎸", 1: "Jazz 🎷", 2: "Hip-Hop 🎤", 3: "Classical 🎻", 4: "Pop 🌟"}

# 3. UI Layout
st.title("🎵 AI Music Genre Classifier")
st.markdown("Upload a music file. The system uses **Librosa** to extract Mel-Frequency Cepstral Coefficients (MFCCs) and passes them through a **Deep Neural Network** to classify the genre.")
st.markdown("---")

uploaded_file = st.file_uploader("Upload an audio file (.wav, .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    # Display Streamlit Audio Player
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Extract Features & Classify ✦", use_container_width=True):
        with st.spinner("Processing audio signals..."):
            try:
                # Load audio using librosa (defaults to 22050 Hz)
                y, sr = librosa.load(uploaded_file, sr=22050)
                
                # Extract 20 MFCCs
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
                
                # Take the mean of the features across the time axis to create a 1D array
                mfccs_processed = np.mean(mfccs.T, axis=0).reshape(1, -1)
                
                # Scale features
                mfccs_scaled = scaler.transform(mfccs_processed)
                
                # Predict
                prediction = model.predict(mfccs_scaled)[0]
                probabilities = model.predict_proba(mfccs_scaled)[0]
                confidence = np.max(probabilities) * 100
                
                st.markdown(f"""
                <div class="genre-card">
                    <h3 style="color: #9B72CB; margin:0;">Detected Genre</h3>
                    <h1 style="margin: 10px 0 0 0;">{genres[prediction]}</h1>
                    <p style="margin-top: 10px; color: #9AA0A6;">Neural Network Confidence: {confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error processing audio file: {e}")