import streamlit as st
import joblib
import numpy as np
import os

# 1. Page Configuration
st.set_page_config(page_title="Iris AI Classifier", page_icon="✨", layout="centered")

# 2. Custom CSS for Gemini-style UI
st.markdown("""
<style>
    /* Main App Background */
    .stApp {
        background-color: #0E0E10;
        color: #E2E2E2;
    }
    
    /* Gradient Text for Headers (Gemini Vibe) */
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #4285F4, #9B72CB, #D96570);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', sans-serif;
    }
    
    /* Slider Styling */
    .stSlider > div > div > div > div {
        background-color: #9B72CB !important;
    }
    
    /* Custom Prediction Card */
    .prediction-card {
        background: linear-gradient(145deg, #1A1A1E 0%, #25252D 100%);
        padding: 25px;
        border-radius: 16px;
        border: 1px solid #3C4043;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        text-align: center;
        margin-top: 30px;
        animation: fadeIn 0.5s ease-in-out;
    }
    
    .species-title {
        font-size: 32px;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #A8C7FA, #E2E2E2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 10px 0;
    }
    
    .confidence-text {
        color: #9AA0A6;
        font-size: 14px;
        font-weight: 500;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# 3. Load the Model
@st.cache_resource
def load_model():
    model_path = 'model/iris_svm_model.pkl'
    if not os.path.exists(model_path):
        st.error("Model not found! Please run 'python train_model.py' first.")
        st.stop()
    return joblib.load(model_path)

model = load_model()
target_names = ['Setosa', 'Versicolor', 'Virginica']

# 4. UI Layout
st.title("✨ Iris AI Classifier")
st.markdown("Adjust the flower measurements below. The AI will analyze the parameters and classify the specific Iris species.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Sepal Features")
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)

with col2:
    st.markdown("### Petal Features")
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.3, 0.1)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3, 0.1)

# 5. Prediction Logic
if st.button("Generate Classification ✦", use_container_width=True):
    # Format input
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Predict
    prediction = model.predict(features)
    probabilities = model.predict_proba(features)
    
    # Get results
    species = target_names[prediction[0]]
    confidence = np.max(probabilities) * 100

    # Display Custom HTML Card
    st.markdown(f"""
    <div class="prediction-card">
        <p style='margin:0; color:#9AA0A6; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;'>AI Analysis Result</p>
        <div class="species-title">Iris {species}</div>
        <p class="confidence-text">Confidence Score: {confidence:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)