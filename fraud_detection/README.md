# 🛡️ AI Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?logo=scikit-learn&logoColor=white)
![SMOTE](https://img.shields.io/badge/Imbalanced_Data-SMOTE-9B72CB)

An interactive Machine Learning security dashboard designed to detect anomalous and fraudulent credit card transactions in real-time. 

This project tackles one of the most notorious challenges in financial machine learning: **Highly Imbalanced Datasets** (where fraudulent transactions represent less than 2% of the data). To prevent the model from overfitting to the majority class, this system utilizes **SMOTE** (Synthetic Minority Over-sampling Technique) to mathematically synthesize realistic fraud patterns during training.

## 🚀 Key Features
*   **Imbalanced Data Handling:** Implements `imblearn.over_sampling.SMOTE` to perfectly balance the training distribution.
*   **Ensemble Learning:** Utilizes a highly parallelized `RandomForestClassifier` for robust non-linear anomaly detection.
*   **Real-Time Inference:** A Streamlit dashboard that allows users to simulate both safe and suspicious transactions, yielding immediate probability scoring.
*   **Automated Data Pipeline:** Programmatically generates a 25,000-row simulated PCA dataset, bypassing the need for manual CSV downloads.

## 📂 Repository Structure
```text
├── app.py                     # Streamlit frontend and dashboard logic
├── train_model.py             # Data generation, SMOTE application, and RF training
└── requirements.txt           # Project dependencies          
