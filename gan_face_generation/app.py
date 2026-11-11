import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from gan_architecture import Generator

st.set_page_config(page_title="GAN Face Generator", page_icon="👤", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #0E0E10; color: #E2E2E2; }
    h1, h2, h3 { background: -webkit-linear-gradient(45deg, #10B981, #3B82F6, #8B5CF6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-family: 'Inter', sans-serif; }
    .image-box { background-color: #1A1A1E; padding: 20px; border-radius: 10px; border: 1px solid #3C4043; text-align: center; margin-top: 20px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_generator():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = Generator(latent_dim=100, feature_maps=64, channels=3).to(device)
    weights_path = "generator_weights.pth"
    
    if os.path.exists(weights_path):
        netG.load_state_dict(torch.load(weights_path, map_location=device))
        netG.eval()
        return netG, device, True
    return netG, device, False

def tensor_to_image(tensor):
    tensor = tensor.detach().cpu().squeeze(0)
    tensor = (tensor + 1) / 2.0               
    tensor = tensor.clamp(0, 1)
    return transforms.ToPILImage()(tensor).resize((256, 256), Image.NEAREST)

netG, device, is_trained = load_generator()

st.title("👤 AI Face Generation (GAN)")
st.markdown("This application uses a trained Deep Convolutional Generative Adversarial Network (DCGAN).")
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("Generate Synthetic Face ✦", use_container_width=True):
        if not is_trained:
            st.error("Weights not found! Run train_gan.py to generate generator_weights.pth.")
        else:
            with st.spinner("Synthesizing latent vector..."):
                noise = torch.randn(1, 100, 1, 1, device=device)
                with torch.no_grad():
                    generated_tensor = netG(noise)
                
                synthetic_image = tensor_to_image(generated_tensor)
                
                st.markdown('<div class="image-box">', unsafe_allow_html=True)
                st.image(synthetic_image, caption="Neural Network Output", width="stretch")
                st.markdown('</div>', unsafe_allow_html=True)