# 👤 Real-Time Face Detection System

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?logo=opencv)

A real-time computer vision application that interfaces with local camera hardware to identify and track human faces using Streamlit and OpenCV.

This project utilizes **Haar Feature-based Cascade Classifiers**, a highly efficient machine learning object detection method. By operating on grayscale gradients and leveraging OpenCV's optimized C++ backend, it achieves real-time bounding box rendering on standard CPU hardware without requiring heavy deep learning frameworks like TensorFlow or PyTorch.

## 🚀 Key Features
*   **Real-Time hardware Access:** Captures and processes live video feeds using `cv2.VideoCapture(0)`.
*   **Haar Cascade Inference:** Applies pre-trained `haarcascade_frontalface_default.xml` for low-latency facial bounding.
*   **Color Space Transformation:** Dynamically handles BGR to RGB conversions for web rendering, and RGB to Grayscale for optimized matrix calculations.
*   **Interactive UI:** Streamlit frontend allowing users to securely toggle hardware access on and off.

## 📂 Repository Structure
```text
├── app.py                 # Streamlit UI, camera control loop, and bounding box logic
└── requirements.txt       # Project dependencies (opencv-python, streamlit)
