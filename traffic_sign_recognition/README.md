# 🛑 AI Traffic Sign Recognition (CNN)

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/Deep_Learning-PyTorch-EE4C2C?logo=pytorch)
![Computer Vision](https://img.shields.io/badge/Algorithm-CNN-9B72CB)

A deep learning web application that uses a Convolutional Neural Network (CNN) to automatically detect and classify road traffic signs. 

Trained on the German Traffic Sign Recognition Benchmark (GTSRB), the model processes raw RGB pixel matrices through multiple convolutional layers to extract spatial features, predicting the sign's class with high accuracy.

## 🚀 Key Features
*   **Deep Learning Pipeline:** Custom CNN architecture built from scratch using `torch.nn`.
*   **Image Transformations:** Utilizes `torchvision.transforms` for automated tensor conversion, normalization, and resizing.
*   **Dataset Automation:** Programmatically fetches and pipelines the GTSRB dataset via `torch.utils.data.DataLoader`.
*   **Interactive UI:** Streamlit frontend allowing users to upload local images and receive real-time classification inferences.

## 📂 Repository Structure
```text
├── app.py                 # Streamlit UI and inference logic
├── train_model.py         # PyTorch dataset fetching and CNN training loop
├── requirements.txt       # Project dependencies
└── model/                     
    └── traffic_cnn.pth    # Saved PyTorch model weights
