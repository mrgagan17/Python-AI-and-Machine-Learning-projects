# 🛡️ AI Spam Email Detector

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?logo=scikit-learn&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Natural_Language_Processing-9B72CB)

A Natural Language Processing (NLP) web application designed to classify emails and text messages as either **Spam** or **Safe (Ham)**. 

Built with a sleek, dark-mode user interface, this project demonstrates core NLP concepts by utilizing a `CountVectorizer` for text tokenization and a **Multinomial Naive Bayes** classifier for binary threat prediction.

## 🚀 Features
*   **NLP Text Processing:** Transforms raw text into numerical features using Scikit-Learn's CountVectorizer.
*   **Binary Classification:** Fast and accurate spam detection using the Naive Bayes algorithm.
*   **Probability Scoring:** Real-time AI confidence percentages for threat analysis.
*   **Modern UI:** Custom dark theme with adaptive color-coded glassmorphism result cards (Red for Spam, Green for Safe).

## 📂 Repository Structure
```text
├── app.py                     # Streamlit frontend and UI configuration
├── train_model.py             # Script to fetch dataset, train NLP pipeline, and export
├── requirements.txt           # Project dependencies
└── model/
    ├── count_vectorizer.pkl   # Exported vocabulary vectorizer
    └── spam_nb_model.pkl      # Pre-trained Naive Bayes model file
