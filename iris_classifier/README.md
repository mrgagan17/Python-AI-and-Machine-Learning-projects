# ✨ Iris AI Classifier

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?logo=scikit-learn&logoColor=white)

A modern, dark-themed Machine Learning web application that classifies Iris flowers into three distinct species (Setosa, Versicolor, Virginica) based on sepal and petal measurements. 

Designed with a sleek, glassmorphism UI inspired by modern AI interfaces, this project utilizes a Support Vector Machine (SVM) algorithm to provide real-time classification and confidence probability scoring.

## 🚀 Features
*   **Multi-class Classification:** Powered by Scikit-Learn's SVM with an RBF kernel.
*   **Probability Scoring:** Displays real-time AI confidence percentages for its predictions.
*   **Modern UI:** Custom dark mode CSS with floating result cards and gradient typography.
*   **Interactive Controls:** Real-time slider inputs for dynamic measurement adjustments.

## 📂 Repository Structure
```text
├── app.py                 # Streamlit frontend and UI configuration
├── train_model.py         # Script to train and export the SVM model
├── requirements.txt       # Project dependencies
└── model/
    └── iris_svm_model.pkl # Pre-trained model file
