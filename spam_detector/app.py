import streamlit as st
import joblib
import os

# 1. Page Configuration
st.set_page_config(page_title="AI Spam Detector", page_icon="🛡️", layout="centered")

# 2. Custom CSS for Gemini-style UI
st.markdown("""
<style>
    .stApp {
        background-color: #0E0E10;
        color: #E2E2E2;
    }
    
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #4285F4, #9B72CB, #D96570);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', sans-serif;
    }
    
    .stTextArea textarea {
        background-color: #1A1A1E !important;
        color: #E2E2E2 !important;
        border: 1px solid #3C4043 !important;
        border-radius: 8px !important;
    }
    
    .prediction-card-spam {
        background: linear-gradient(145deg, #2D1A1A 0%, #3A1A1A 100%);
        padding: 25px;
        border-radius: 16px;
        border: 1px solid #5C2B2B;
        box-shadow: 0 8px 32px rgba(217, 101, 112, 0.2);
        text-align: center;
        margin-top: 30px;
        animation: fadeIn 0.5s ease-in-out;
    }

    .prediction-card-ham {
        background: linear-gradient(145deg, #1A2D22 0%, #1A3A25 100%);
        padding: 25px;
        border-radius: 16px;
        border: 1px solid #2B5C3B;
        box-shadow: 0 8px 32px rgba(66, 133, 244, 0.2);
        text-align: center;
        margin-top: 30px;
        animation: fadeIn 0.5s ease-in-out;
    }
    
    .result-title {
        font-size: 32px;
        font-weight: 800;
        margin: 10px 0;
    }
    
    .spam-text { background: -webkit-linear-gradient(45deg, #FF6B6B, #D96570); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .ham-text { background: -webkit-linear-gradient(45deg, #4285F4, #A8C7FA); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    
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

# 3. Load the Model and Vectorizer
@st.cache_resource
def load_assets():
    model_path = 'model/spam_nb_model.pkl'
    vec_path = 'model/count_vectorizer.pkl'
    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        st.error("Model files not found! Please run 'python train_model.py' first.")
        st.stop()
    return joblib.load(vec_path), joblib.load(model_path)

vectorizer, model = load_assets()

# 4. UI Layout
st.title("🛡️ AI Spam Detector")
st.markdown("Paste an email or text message below. The NLP model will analyze the vocabulary and classify it as **Spam** or **Safe (Ham)**.")

st.markdown("---")

user_input = st.text_area("Message Content", height=150, placeholder="Congratulations! You've won a $1,000 Walmart gift card. Click here to claim your prize now...")

# 5. Prediction Logic
if st.button("Run Threat Analysis ✦", use_container_width=True):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Transform input using the loaded CountVectorizer
        input_dtm = vectorizer.transform([user_input])
        
        # Predict
        prediction = model.predict(input_dtm)[0]
        probabilities = model.predict_proba(input_dtm)[0]
        
        confidence = probabilities[prediction] * 100

        # Output Results Formatting
        if prediction == 1:
            card_class = "prediction-card-spam"
            title_class = "spam-text"
            label = "🚨 SPAM DETECTED"
        else:
            card_class = "prediction-card-ham"
            title_class = "ham-text"
            label = "✅ SAFE (HAM)"

        # Display Custom HTML Card
        st.markdown(f"""
        <div class="{card_class}">
            <p style='margin:0; color:#9AA0A6; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;'>NLP Analysis Result</p>
            <div class="result-title {title_class}">{label}</div>
            <p class="confidence-text">Confidence Score: {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)