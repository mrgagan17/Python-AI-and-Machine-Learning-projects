import streamlit as st
import pandas as pd
import joblib
import os

# 1. Page Configuration
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

# 2. Custom CSS for Gemini-style UI
st.markdown("""
<style>
    .stApp { background-color: #0E0E10; color: #E2E2E2; }
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #4285F4, #9B72CB, #D96570);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', sans-serif;
    }
    .stSelectbox > div > div, .stSlider > div > div { background-color: #1A1A1E !important; }
    .risk-high { border-left: 5px solid #EA4335; background: rgba(234, 67, 53, 0.1); padding: 20px; border-radius: 8px;}
    .risk-low { border-left: 5px solid #34A853; background: rgba(52, 168, 83, 0.1); padding: 20px; border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

# 3. Load Pipeline
@st.cache_resource
def load_pipeline():
    model_path = 'model/xgboost_churn_pipeline.pkl'
    if not os.path.exists(model_path):
        st.error("Model pipeline not found! Run 'python train_model.py' first.")
        st.stop()
    return joblib.load(model_path)

pipeline = load_pipeline()

# 4. UI Header
st.title("📉 Customer Churn Predictor")
st.markdown("Adjust the customer profile below. The **XGBoost Ensemble Model** will calculate the real-time probability of this customer canceling their subscription.")
st.markdown("---")

# 5. Interactive Customer Profile Input
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (Months with company)", min_value=1, max_value=72, value=12)
    monthly_charges = st.slider("Monthly Charges (\$)", min_value=20.0, max_value=120.0, value=75.0, step=0.5)

with col2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

if st.button("Predict Churn Risk ✦", use_container_width=True):
    # Construct a DataFrame exactly as the pipeline expects
    input_data = pd.DataFrame({
        'Tenure_Months': [tenure],
        'Monthly_Charges': [monthly_charges],
        'Contract_Type': [contract],
        'Internet_Service': [internet]
    })
    
    # Predict using the loaded pipeline (handles scaling and encoding automatically)
    probability = pipeline.predict_proba(input_data)[0][1] * 100
    
    st.markdown("### AI Retention Analysis")
    
    if probability > 50:
        st.markdown(f"""
        <div class="risk-high">
            <h3 style="color: #EA4335; margin:0;">⚠️ High Flight Risk</h3>
            <p style="margin: 5px 0 0 0;">There is a <b>{probability:.1f}%</b> probability this customer will cancel. Consider offering a promotional discount or switching them to an annual contract.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="risk-low">
            <h3 style="color: #34A853; margin:0;">✅ Safe / Retained</h3>
            <p style="margin: 5px 0 0 0;">There is only a <b>{probability:.1f}%</b> probability of cancellation. Customer shows strong loyalty metrics.</p>
        </div>
        """, unsafe_allow_html=True)