# 🐦 Live Twitter Sentiment Analyzer

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-TextBlob-9B72CB)
![Tweepy](https://img.shields.io/badge/API-Twitter_v2-1DA1F2?logo=x)

A real-time Natural Language Processing (NLP) dashboard that scrapes public data from the Twitter (X) platform and classifies audience sentiment regarding brands, topics, or elections.

## 🚀 Key Features
*   **Live Data Ingestion:** Integrates the `Tweepy` client to query the Twitter API v2 in real-time.
*   **Regex Data Cleaning:** Utilizes Python Regular Expressions to strip raw social media text of hyperlinks, @mentions, and hashtags prior to processing.
*   **Lexicon-Based NLP:** Deploys `TextBlob` to assign positive, negative, or neutral polarity scores to unstructured text.
*   **Mock-Data Architecture:** Includes a seamless fallback simulation engine, ensuring the application remains functional for portfolio demonstrations even if API rate limits are exhausted.

## 📂 Repository Structure
```text
├── app.py                 # Streamlit UI, Tweepy API pipeline, and NLP logic
└── requirements.txt       # Project dependencies
