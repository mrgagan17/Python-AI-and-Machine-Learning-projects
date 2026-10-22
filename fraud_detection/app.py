import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import os

# 1. Page Configuration
st.set_page_config(page_title="Fraud Detection Engine", page_icon="🛡️", layout="centered")

# 2. Custom Dark Theme CSS
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
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #3C4043;
        text-align: center;
        margin-bottom: 20px;
    }
    .safe-alert { border-left: 5px solid #34A853; background: rgba(52, 168, 83, 0.1); padding: 20px; border-radius: 8px;}
    .fraud-alert { border-left: 5px solid #EA4335; background: rgba(234, 67, 53, 0.1); padding: 20px; border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

# 3. Load Model
@st.cache_resource
def load_model():
    model_path = 'model/smote_fraud_model.pkl'
    if not os.path.exists(model_path):
        st.error("Model not found! Run 'python train_model.py' first.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# 4. UI Header
st.title("🛡️ AI Fraud Detection Engine")
st.markdown("This system analyzes transaction metadata using a **Random Forest Classifier** trained on highly imbalanced data mitigated via **SMOTE** (Synthetic Minority Over-sampling Technique).")
st.markdown("---")

# 5. Interactive Dashboard
st.subheader("Live Transaction Terminal")

col1, col2 = st.columns(2)
with col1:
    amount = st.number_input("Transaction Amount (\$)", min_value=1.00, max_value=10000.00, value=250.00, step=10.0)
with col2:
    transaction_type = st.selectbox("Simulate Network Behavior", ["Standard Transaction", "Anomalous/Suspicious Behavior"])

if st.button("Intercept & Analyze Transaction", use_container_width=True):
    # Simulate backend PCA variables based on user selection
    if transaction_type == "Standard Transaction":
        # Generate safe-looking data (closer to mean 0)
        v_features = np.random.normal(0, 1, 11)
    else:
        # Generate anomalous data (statistical outliers)
        v_features = np.random.normal(3, 2, 11)
    
    # Construct the final feature array matching the training data
    input_data = np.array([amount] + list(v_features)).reshape(1, -1)
    
    # Simulate processing time for effect
    with st.spinner("Analyzing neural patterns and network anomalies..."):
        time.sleep(1.5)
        
        # Predict
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
    st.markdown("### Security Analysis Result")
    
    if prediction == 0:
        st.markdown(f"""
        <div class="safe-alert">
            <h3 style="color: #34A853; margin:0;">✅ Transaction Approved</h3>
            <p style="margin: 5px 0 0 0;">No anomalies detected. Confidence: {probabilities[0]*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="fraud-alert">
            <h3 style="color: #EA4335; margin:0;">🚨 FRAUD DETECTED</h3>
            <p style="margin: 5px 0 0 0;">Transaction blocked. Confidence: {probabilities[1]*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
    # Display the simulated metadata to show the user what the AI actually looked at
    with st.expander("View Raw Transaction Metadata (V1-V11)"):
        st.json({f"V{i+1}": round(v_features[i], 4) for i in range(11)})