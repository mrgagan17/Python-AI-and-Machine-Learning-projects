import os
os.environ["USE_TF"] = "0"

import streamlit as st
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="AI Image Captioning", page_icon="🖼️", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #0E0E10; color: #E2E2E2; }
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #4285F4, #9B72CB, #D96570);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', sans-serif;
    }
    .caption-box { background-color: #1A1A1E; padding: 20px; border-radius: 10px; border: 1px solid #3C4043; text-align: center; margin-top: 20px;}
</style>
""", unsafe_allow_html=True)
# 2. Load Microsoft GIT Model (Using the fully intact base repository)
# 2. Load Microsoft GIT Model (With extended network timeouts)
@st.cache_resource
def load_model():
    # Adding a 300-second timeout to prevent network interruption crashes
    processor = AutoProcessor.from_pretrained("microsoft/git-base", resume_download=True)
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base", resume_download=True)
    return processor, model

processor, model = load_model()

# 3. UI Layout
st.title("🖼️ AI Image Caption Generator")
st.markdown("Upload any image, and the deep learning model will analyze the visual features and generate a natural language sentence describing the scene.")
st.markdown("---")

uploaded_file = st.file_uploader("Upload an Image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file).convert('RGB')
    st.image(raw_image, caption='Uploaded Image', width="stretch")
    
    if st.button("Generate Caption ✦", use_container_width=True):
        with st.spinner("Analyzing visual features and decoding text..."):
            try:
                # Preprocess the image and generate text
                pixel_values = processor(images=raw_image, return_tensors="pt").pixel_values
                generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
                caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                st.markdown(f"""
                <div class="caption-box">
                    <h3 style="color: #34A853; margin:0;">Generated Caption</h3>
                    <h2 style="margin: 10px 0 0 0; text-transform: capitalize;">"{caption}"</h2>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error generating caption: {e}")