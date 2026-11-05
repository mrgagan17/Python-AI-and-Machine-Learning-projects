# 🕵️‍♂️ DeepFake Detection System

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/Deep_Learning-PyTorch-EE4C2C?logo=pytorch)
![OpenCV](https://img.shields.io/badge/Computer_Vision-OpenCV-5C3EE8)

An advanced Vision-Language and Forensic AI web application designed to differentiate between authentic media and AI-generated DeepFakes by analyzing microscopic digital artifacts.

## 📸 Output Demo
*The forensic neural network analyzing both authentic and manipulated media to determine authenticity.*

<p align="center">
  <img src="output1.png" width="49%" />
  <img src="output2.png" width="49%" />
</p>

## 🚀 Key Features
*   **Spatio-Temporal Architecture Skills:** Includes `cnn_forensic_architecture.py`, demonstrating the implementation of a custom ResNeXt50 + LSTM neural network built from scratch in PyTorch to track unnatural blinking and temporal blending errors.
*   **Video Frame Extraction:** Utilizes `OpenCV` to parse uploaded `.mp4` video files, dynamically mapping and extracting temporal frames for batched neural network processing.
*   **Production Inference:** Deploys a highly accurate Hugging Face Vision Transformer (ViT) via an interactive, dark-themed Streamlit dashboard.

## 📂 Repository Structure
```text
├── app.py                         # Streamlit UI and inference pipeline
├── cnn_forensic_architecture.py   # Pure PyTorch CNN (ResNeXt) + RNN (LSTM) code
└── requirements.txt               # Project dependencies
