# 👁️ Real-Time Object Detection (YOLOv8)

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/Deep_Learning-PyTorch-EE4C2C?logo=pytorch)
![YOLO](https://img.shields.io/badge/Architecture-YOLOv8-34A853)

An advanced computer vision application deploying the **YOLOv8 (You Only Look Once)** architecture to identify, classify, and track multiple objects in real-time video feeds.

Trained on the COCO dataset, this deep learning model parses live hardware inputs via OpenCV, rendering high-speed bounding boxes and confidence metrics for 80 distinct entity classes (pedestrians, vehicles, electronics) simultaneously on CPU hardware.

## 🚀 Key Features
*   **State-of-the-Art Inference:** Integrates the `ultralytics` package for highly optimized YOLOv8 neural network processing.
*   **Live Hardware Processing:** Uses `cv2.VideoCapture(0)` to feed raw matrix data directly into the model without intermediate saving.
*   **Dynamic Thresholding:** Includes interactive Streamlit sliders to adjust non-maximum suppression (NMS) confidence thresholds on the fly.
*   **Automated Annotation:** Leverages the native `.plot()` method on Ultralytics result objects for mathematically precise bounding box overlays.


## 📸 Live Detection Demo
*The YOLOv8 model successfully rendering an 84% confidence bounding box around a cell phone in real-time.*

![YOLOv8 Real-Time Output](output.jpg)

## 📂 Repository Structure
```text
├── app.py                 # Streamlit UI, camera loop, and YOLO inference pipeline
├── requirements.txt       # Project dependencies
└── .gitignore             # Excluded directories and auto-downloaded weights
