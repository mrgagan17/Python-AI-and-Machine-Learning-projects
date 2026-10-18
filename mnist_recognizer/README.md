# ✍️ AI Digit Recognizer (MNIST)

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?logo=scikit-learn&logoColor=white)
![Neural Networks](https://img.shields.io/badge/Neural_Networks-MLPClassifier-9B72CB)

An interactive Machine Learning web application that recognizes handwritten digits (0-9) drawn directly on the screen by the user. 

This project explores the fundamentals of Image Classification and Artificial Neural Networks, utilizing the famous MNIST dataset and a Multi-Layer Perceptron (MLP) architecture. It features a sleek, dark-themed user interface with real-time canvas processing.

## 🚀 Key Features
*   **Interactive Drawing Canvas:** Integrated `streamlit-drawable-canvas` allows real-time user input via mouse or touch interface.
*   **Neural Network Architecture:** Utilizes a Scikit-Learn `MLPClassifier` with a 128-neuron hidden layer to process 784-pixel (28x28) image inputs.
*   **Automated Image Processing:** The backend automatically grayscales, downscales (LANCZOS resampling), and flattens canvas inputs to perfectly match the MNIST training structure.
*   **Modern UI/UX:** Custom dark mode CSS featuring gradient typography and glassmorphism prediction cards.

## 🛠️ Tech Stack
*   **Python 3.x**
*   **Scikit-Learn** (Model Training, MLPClassifier, OpenML Dataset Fetching)
*   **Streamlit** (Web UI Framework)
*   **Streamlit-Drawable-Canvas** (Interactive Drawing Component)
*   **Pillow (PIL) & NumPy** (Image Processing and Matrix Flattening)

## 📂 Repository Structure
```text
├── app.py                     # Streamlit frontend, canvas component, and prediction logic
├── train_model.py             # Script to fetch MNIST, train the Neural Network, and export
├── requirements.txt           # Project dependencies
├── .gitignore                 # Ignored files (venv, __pycache__)
└── model/
    └── mnist_nn_model.pkl     # Pre-trained Neural Network file
