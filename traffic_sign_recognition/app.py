import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# 1. Page Configuration
st.set_page_config(page_title="Traffic Sign AI", page_icon="🛑", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #0E0E10; color: #E2E2E2; }
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #4285F4, #9B72CB, #D96570);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', sans-serif;
    }
    .result-box { background-color: #1A1A1E; padding: 20px; border-radius: 10px; border: 1px solid #3C4043; text-align: center; margin-top: 20px;}
</style>
""", unsafe_allow_html=True)

# 2. Define CNN Architecture (Must match train_model.py exactly)
class TrafficSignCNN(nn.Module):
    def __init__(self):
        super(TrafficSignCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 43)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 3. Load Model Weights
@st.cache_resource
def load_model():
    model = TrafficSignCNN()
    model_path = 'model/traffic_cnn.pth'
    if not os.path.exists(model_path):
        st.error("Model weights not found! Run 'python train_model.py' first.")
        st.stop()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

model = load_model()

# 4. GTSRB Class Labels
classes = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)', 
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)', 
    7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)', 9: 'No passing', 
    11: 'Right-of-way at intersection', 12: 'Priority road', 13: 'Yield', 14: 'Stop', 
    15: 'No vehicles', 17: 'No entry', 18: 'General caution', 25: 'Road work', 
    31: 'Wild animals crossing', 33: 'Turn right ahead', 34: 'Turn left ahead', 35: 'Ahead only'
}

# 5. UI Layout
st.title("🛑 AI Traffic Sign Recognition")
st.markdown("Upload an image of a road sign, and the **Convolutional Neural Network (CNN)** will classify it using weights trained on the German Traffic Sign Recognition Benchmark.")
st.markdown("---")

uploaded_file = st.file_uploader("Choose a traffic sign image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', width=300)
    
    if st.button("Classify Sign ✦", use_container_width=True):
        with st.spinner("Analyzing convolutional feature maps..."):
            # Preprocess image to match training data
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            # Add batch dimension: [1, 3, 32, 32]
            image_tensor = transform(image).unsqueeze(0)
            
            # Predict
            with torch.no_grad():
                outputs = model(image_tensor)
                _, predicted = torch.max(outputs.data, 1)
                class_id = predicted.item()
                
            predicted_label = classes.get(class_id, f"Unknown Sign (Class ID: {class_id})")
            
            st.markdown(f"""
            <div class="result-box">
                <h3 style="color: #34A853; margin:0;">Classification Result</h3>
                <h2 style="margin: 10px 0 0 0;">{predicted_label}</h2>
            </div>
            """, unsafe_allow_html=True)