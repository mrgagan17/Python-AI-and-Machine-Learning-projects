# 💬 Rule-Based AI Support Chatbot

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP_Library-9B72CB)

A sleek, conversational AI customer support agent built using rule-based Natural Language Processing (NLP) pattern matching. 

This project demonstrates core NLP concepts by using the Natural Language Toolkit (NLTK) to parse user intents, handle conversational reflections, and trigger appropriate responses without relying on heavy machine learning models.

## 🚀 Features
*   **Regex Pattern Matching:** Fast and lightweight intent recognition using NLTK's `Chat` utility.
*   **Conversational Memory:** Utilizes Streamlit's session state to maintain chat history during the user's session.
*   **Modern UI/UX:** Built with Streamlit's native chat elements, augmented with a custom dark-mode, glassmorphism-inspired CSS theme.
*   **Expandable Logic:** Easily add new intents and responses to the `patterns` dictionary in the backend.

## 📂 Repository Structure
```text
├── app.py                 # Streamlit chat interface and session state management
├── bot_logic.py           # NLP logic, regex patterns, and NLTK integration
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
