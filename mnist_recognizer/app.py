import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from PIL import Image
import scipy.ndimage
from streamlit_drawable_canvas import st_canvas

# 1. Page Configuration
st.set_page_config(page_title="AI Digit Recognizer", page_icon="✍️", layout="centered")

# 2. Custom CSS for Modern Dark UI
st.markdown("""
<style>
    .stApp { background-color: #0E0E10; color: #E2E2E2; }
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #4285F4, #9B72CB, #D96570);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', sans-serif;
    }
    .prediction-card {
        background: linear-gradient(145deg, #1A1A1E 0%, #25252D 100%);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #3C4043;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        text-align: center;
        margin-top: 10px;
        animation: fadeIn 0.4s ease-in-out;
    }
    .digit-title {
        font-size: 84px;
        font-weight: 800;
        line-height: 1.1;
        background: -webkit-linear-gradient(45deg, #A8C7FA, #E2E2E2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 5px 0;
    }
    .confidence-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        background: rgba(66, 133, 244, 0.15);
        border: 1px solid rgba(66, 133, 244, 0.4);
        color: #8AB4F8;
        font-size: 13px;
        font-weight: 600;
        margin-top: 6px;
    }
    .vision-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 15px;
        background: #16161A;
        border-radius: 12px;
        padding: 12px;
        border: 1px solid #2D2F36;
        margin-top: 15px;
    }
    .prob-row {
        display: flex;
        align-items: center;
        margin: 4px 0;
        font-size: 12px;
        font-family: 'Inter', sans-serif;
    }
    .prob-label {
        width: 20px;
        font-weight: 700;
        color: #C4C7C5;
    }
    .prob-bar-bg {
        flex-grow: 1;
        height: 8px;
        background: #25252D;
        border-radius: 4px;
        margin: 0 8px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #4285F4, #9B72CB);
        transition: width 0.3s ease;
    }
    .prob-bar-fill-active {
        background: linear-gradient(90deg, #34A853, #4285F4);
    }
    .prob-val {
        width: 44px;
        text-align: right;
        color: #9AA0A6;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# 3. Load Model
@st.cache_resource
def load_model():
    model_path = 'model/mnist_nn_model.pkl'
    if not os.path.exists(model_path):
        st.error("Model not found! Please run 'python train_model.py' first.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# 4. MNIST Preprocessing Pipeline
def preprocess_canvas_image(canvas_img_data):
    """
    Normalizes a user drawing to match the standard MNIST dataset format:
    1. Extract drawing mask from canvas RGBA image
    2. Filter out faint noise / antialiasing
    3. Find content bounding box
    4. Size normalize into a 20x20 box preserving aspect ratio
    5. Center within a 28x28 frame (leaving 4-pixel border)
    6. Translate using Center of Mass (official Yann LeCun MNIST specification)
    """
    if canvas_img_data is None:
        return None, None

    # Handle canvas RGBA numpy array
    if isinstance(canvas_img_data, np.ndarray):
        if canvas_img_data.ndim == 3 and canvas_img_data.shape[2] == 4:
            rgb = canvas_img_data[:, :, :3]
            alpha = canvas_img_data[:, :, 3]
            # Convert RGB luminance to grayscale
            gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            # If drawing occurred on alpha channel
            if gray.max() == 0 and alpha.max() > 0:
                gray = alpha
            img = Image.fromarray(gray)
        elif canvas_img_data.ndim == 3:
            img = Image.fromarray(canvas_img_data[:, :, 0].astype(np.uint8))
        else:
            img = Image.fromarray(canvas_img_data.astype(np.uint8))
    else:
        img = canvas_img_data.convert('L')

    img = img.convert('L')
    
    # Suppress low-intensity antialiasing artifacts
    arr = np.array(img)
    arr[arr < 35] = 0
    img = Image.fromarray(arr)

    # Locate bounding box of the drawn digit
    bbox = img.getbbox()
    if bbox is None:
        return None, None

    # Crop tightly to the digit
    cropped = img.crop(bbox)
    w, h = cropped.size

    # Fit into 20x20 box while strictly preserving aspect ratio
    if w > h:
        new_w = 20
        new_h = max(1, int(round((h * 20.0) / w)))
    else:
        new_h = 20
        new_w = max(1, int(round((w * 20.0) / h)))

    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Center within a 28x28 black canvas
    padded = Image.new('L', (28, 28), 0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    padded.paste(resized, (paste_x, paste_y))

    img_arr = np.array(padded, dtype=np.float32) / 255.0

    # Center of mass alignment (Yann LeCun's standard normalization for MNIST)
    total_mass = img_arr.sum()
    if total_mass > 0:
        cy, cx = scipy.ndimage.center_of_mass(img_arr)
        shift_y = np.clip(13.5 - cy, -4.0, 4.0)
        shift_x = np.clip(13.5 - cx, -4.0, 4.0)
        img_arr = scipy.ndimage.shift(img_arr, [shift_y, shift_x], mode='constant', cval=0.0)
        img_arr = np.clip(img_arr, 0.0, 1.0)

    # Visual 28x28 representation
    preprocessed_pil = Image.fromarray((img_arr * 255).astype(np.uint8))

    # Flatten for MLP input
    features = img_arr.flatten().reshape(1, -1)
    if hasattr(model, 'feature_names_in_'):
        features = pd.DataFrame(features, columns=model.feature_names_in_)

    return features, preprocessed_pil

# 5. UI Layout
st.title("✍️ AI Digit Recognizer")
st.markdown("Draw any digit (0–9) anywhere on the canvas. The AI automatically centers, normalizes, and recognizes your drawing using MNIST Neural Network standards.")

st.markdown("---")

col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.markdown("### Drawing Canvas")
    
    # Stroke width slider for user flexibility
    stroke_width = st.slider("Stroke Thickness", min_value=14, max_value=28, value=20, step=2, help="Adjust pen width for smooth drawing")
    
    # Drawable Canvas Component
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=stroke_width,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
        return_image_data=True,
    )
    
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        predict_clicked = st.button("Predict Digit ✦", use_container_width=True, type="primary")
    with col_btn2:
        auto_predict = st.checkbox("Live Auto-Predict", value=True, help="Update prediction instantly after each stroke")

with col2:
    st.markdown("### AI Prediction")
    
    has_drawing = (
        canvas_result.json_data is not None and 
        len(canvas_result.json_data.get("objects", [])) > 0
    )
    
    should_run_prediction = (predict_clicked or auto_predict) and has_drawing

    if should_run_prediction:
        try:
            features, vis_pil = preprocess_canvas_image(canvas_result.image_data)
            
            if features is not None:
                prediction = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]
                confidence = probabilities[prediction] * 100

                # Top Result Card
                st.markdown(f"""
                <div class="prediction-card">
                    <p style='margin:0; color:#9AA0A6; font-size: 12px; text-transform: uppercase; letter-spacing: 1.5px;'>Network Prediction</p>
                    <div class="digit-title">{prediction}</div>
                    <div class="confidence-badge">Confidence: {confidence:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Preprocessed AI View & Probability Breakdown
                st.markdown("<br>", unsafe_allow_html=True)
                tabs = st.tabs(["📊 Confidence Breakdown", "👁️ AI View (28×28)"])
                
                with tabs[0]:
                    prob_rows = []
                    for digit_idx in range(10):
                        prob = float(probabilities[digit_idx] * 100)
                        is_active = (digit_idx == prediction)
                        bar_class = "prob-bar-fill prob-bar-fill-active" if is_active else "prob-bar-fill"
                        weight_style = "font-weight: 700; color: #8AB4F8;" if is_active else "color: #9AA0A6;"
                        val_style = "color: #8AB4F8; font-weight: 700;" if is_active else "color: #9AA0A6;"
                        prob_rows.append(
                            f'<div class="prob-row">'
                            f'<span class="prob-label" style="{weight_style}">{digit_idx}</span>'
                            f'<div class="prob-bar-bg">'
                            f'<div class="{bar_class}" style="width: {prob:.1f}%;"></div>'
                            f'</div>'
                            f'<span class="prob-val" style="{val_style}">{prob:.1f}%</span>'
                            f'</div>'
                        )
                    st.markdown("".join(prob_rows), unsafe_allow_html=True)

                with tabs[1]:
                    st.caption("This is the size-normalized & center-of-mass aligned 28×28 image passed into the neural network:")
                    # Display the 28x28 image enlarged with pixel-perfect clarity
                    enlarged_vis = vis_pil.resize((140, 140), Image.Resampling.NEAREST)
                    st.image(enlarged_vis, caption="Normalized 28×28 MNIST Input", width=140)
            else:
                st.info("Canvas appears empty. Draw a digit inside the black box.")
        except Exception as e:
            st.error(f"Prediction error: {e}")
            
    elif not has_drawing:
        st.markdown("""
        <div class="prediction-card" style="opacity: 0.7;">
            <p style='margin:0; color:#9AA0A6; font-size: 13px;'>Canvas is currently empty</p>
            <div style="font-size: 40px; margin: 15px 0;">✏️</div>
            <p style='color:#757B82; font-size:13px; margin:0;'>Draw a digit (0–9) on the canvas to see the neural network predict it!</p>
        </div>
        """, unsafe_allow_html=True)