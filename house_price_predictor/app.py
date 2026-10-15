import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

st.set_page_config(
    page_title="Real-World Metropolitan Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Load Model & Columns safely
@st.cache_resource
def load_artifacts():
    if os.path.exists('model/model.pkl') and os.path.exists('model/columns.json'):
        model = joblib.load('model/model.pkl')
        with open('model/columns.json', 'r') as f:
            data_columns = json.load(f)['data_columns']
        return model, data_columns
    return None, None

model, data_columns = load_artifacts()

def format_inr(number):
    if number >= 10000000:
        return f"₹{number / 10000000:.2f} Crores"
    elif number >= 100000:
        return f"₹{number / 100000:.2f} Lakhs"
    else:
        return f"₹{number:,.2f}"

st.title("🏠 Real-World Metropolitan Housing Price Predictor")
st.markdown("Predict property valuations based on actual market trends across Bengaluru localities.")
st.divider()

if model is None or data_columns is None:
    st.error("⚠️ Model artifacts missing! Please run `python model/train_model.py` first.")
else:
    # Extract locations from columns schema
    locations = [col for col in data_columns if col not in ['total_sqft', 'bhk', 'bath']]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Property Specifications")
        
        location = st.selectbox("Select Locality / Area", options=sorted(locations), index=sorted(locations).index('Koramangala') if 'Koramangala' in locations else 0)
        
        # Set realistic default of 1,200 sq ft for a standard 2-3 BHK
        sqft = st.slider("Total Built-up Area (sq. ft.)", min_value=500, max_value=20000, value=1200, step=50)
        
        col_bhk, col_bath = st.columns(2)
        with col_bhk:
            bhk = st.selectbox("Bedrooms (BHK)", options=[1, 2, 3, 4, 5, 6, 7, 8], index=2) # Default 3 BHK
        with col_bath:
            bath = st.selectbox("Bathrooms", options=[1, 2, 3, 4, 5, 6, 7, 8], index=1) # Default 2 Baths
            
        predict_btn = st.button("Predict Market Price", type="primary", use_container_width=True)

    with col2:
        st.subheader("📊 Valuation Output")
        
        # Automatic calculation or on button click
        x_input = np.zeros(len(data_columns))
        x_input[data_columns.index('total_sqft')] = sqft
        x_input[data_columns.index('bhk')] = bhk
        x_input[data_columns.index('bath')] = bath
        
        if location in data_columns:
            loc_index = data_columns.index(location)
            x_input[loc_index] = 1

        prediction = model.predict([x_input])[0]
        rate_per_sqft = prediction / sqft
        
        st.metric(label="Estimated Valuation", value=format_inr(prediction))
        st.caption(f"Exact Value: ₹{prediction:,.2f} | Effective Rate: **₹{rate_per_sqft:,.2f} / sq. ft.**")
        
        st.success("Prediction computed successfully using trained Linear Regression weights.")
        
        st.info(f"**Configuration Summary:**\n"
                f"- **Location:** {location}\n"
                f"- **Area:** {sqft:,} sq. ft.\n"
                f"- **Layout:** {bhk} BHK | {bath} Bathrooms\n"
                f"- **Estimated Price Range:** {format_inr(prediction * 0.95)} – {format_inr(prediction * 1.05)}")

    st.divider()
    st.subheader("📈 Dataset Overview")
    if os.path.exists('data/bengaluru_housing.csv'):
        df = pd.read_csv('data/bengaluru_housing.csv')
        st.dataframe(df.head(10), use_container_width=True)