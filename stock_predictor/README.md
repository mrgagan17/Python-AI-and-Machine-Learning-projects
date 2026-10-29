# 📈 AI Stock Market Predictor

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/Deep_Learning-PyTorch-EE4C2C?logo=pytorch)
![yfinance](https://img.shields.io/badge/API-yfinance-34A853)

A deep learning financial dashboard that utilizes a **Long Short-Term Memory (LSTM)** recurrent neural network to perform time-series forecasting on global equities.

This application automatically pulls live historical market data from the Yahoo Finance API, scales the feature sets, and passes trailing 60-day sequence windows through the PyTorch model to forecast the subsequent day's closing price.

## 🚀 Key Features
*   **Recurrent Neural Networks:** Built using `torch.nn.LSTM` to mathematically preserve sequential time-series patterns and mitigate the vanishing gradient problem.
*   **Dynamic Data Ingestion:** Integrates the `yfinance` library for real-time, ticker-agnostic data fetching.
*   **Automated Preprocessing:** Utilizes `sklearn.preprocessing.MinMaxScaler` to dynamically normalize incoming data streams on the fly for standardized inference.
*   **Interactive Data Visualization:** Renders interactive, dark-mode financial charts utilizing `plotly.graph_objects`.

## 📂 Repository Structure
```text
├── app.py                     # Streamlit frontend, yfinance API logic, and Plotly charts
├── train_model.py             # PyTorch LSTM architecture and training loop
├── requirements.txt           # Project dependencies
└── model/                     
    └── stock_lstm.pth         # Compiled PyTorch model weights
