import streamlit as st
import torch
import torch.nn as nn
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import os

# 1. Page Configuration
st.set_page_config(page_title="LSTM Stock Predictor", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0E0E10; color: #E2E2E2; }
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #34A853, #4285F4, #9B72CB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', sans-serif;
    }
    .metric-card { background-color: #1A1A1E; padding: 20px; border-radius: 10px; border: 1px solid #3C4043; text-align: center; }
</style>
""", unsafe_allow_html=True)

# 2. Define LSTM Architecture (Must match training script)
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

# 3. Load Model
@st.cache_resource
def load_model():
    model = StockLSTM()
    model_path = 'model/stock_lstm.pth'
    if not os.path.exists(model_path):
        st.error("Model weights not found! Run 'python train_model.py' first.")
        st.stop()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

model = load_model()

# 4. UI Layout
st.title("📈 AI Stock Market Predictor")
st.markdown("Enter a stock ticker to fetch real-time market data. A **Long Short-Term Memory (LSTM)** neural network analyzes the last 60 days of trading sequences to forecast tomorrow's closing price.")
st.markdown("*Note: This is a machine learning portfolio demonstration, not financial advice.*")
st.markdown("---")

col1, col2 = st.columns([1, 3])

with col1:
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, NVDA, TSLA):", value="NVDA")
    analyze_btn = st.button("Run AI Forecast ✦", use_container_width=True)

if analyze_btn:
    with st.spinner(f"Fetching live market data for {ticker}..."):
        try:
            # Fetch 1 year of recent data
            data = yf.download(ticker, period="1y")
            if data.empty:
                st.error("Invalid ticker symbol or no data found.")
            else:
                # Prepare data
                close_prices = data[['Close']].values
                latest_close = close_prices[-1][0]
                
                # Scale data based on this specific stock's recent volatility
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(close_prices)
                
                # Extract the last 60 days to predict tomorrow
                last_60_days = scaled_data[-60:]
                X_test = np.array([last_60_days])
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                X_tensor = torch.tensor(X_test, dtype=torch.float32)
                
                # Run Inference
                with torch.no_grad():
                    predicted_scaled = model(X_tensor)
                    
                # Inverse transform back to real dollar amount
                predicted_price = scaler.inverse_transform(predicted_scaled.numpy())[0][0]
                
                price_change = predicted_price - latest_close
                percent_change = (price_change / latest_close) * 100
                color = "#34A853" if price_change >= 0 else "#EA4335"

                with col1:
                    st.markdown(f"""
                    <div class="metric-card" style="margin-top: 20px;">
                        <h3 style="color: #9AA0A6; margin:0;">Current Close</h3>
                        <h2 style="margin: 5px 0 15px 0;">\${latest_close:.2f}</h2>
                        <h3 style="color: #9B72CB; margin:0;">AI Forecast (Next Day)</h3>
                        <h2 style="margin: 5px 0 0 0; color: {color};">\${predicted_price:.2f}</h2>
                        <p style="color: {color}; font-weight: bold; margin:0;">{percent_change:+.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Plotly Interactive Chart
                    fig = go.Figure()
                    
                    # Historical Line
                    fig.add_trace(go.Scatter(
                        x=data.index[-90:], # Show last 90 days on chart
                        y=data['Close'].values.flatten()[-90:],
                        mode='lines',
                        name='Historical Close',
                        line=dict(color='#4285F4', width=2)
                    ))
                    
                    # Predicted Point
                    next_day = data.index[-1] + pd.Timedelta(days=1)
                    fig.add_trace(go.Scatter(
                        x=[data.index[-1], next_day],
                        y=[latest_close, predicted_price],
                        mode='lines+markers',
                        name='LSTM Forecast',
                        line=dict(color=color, width=3, dash='dash'),
                        marker=dict(size=8)
                    ))
                    
                    fig.update_layout(
                        title=f"{ticker} - 90 Day Trend & Next Day Forecast",
                        plot_bgcolor='#0E0E10',
                        paper_bgcolor='#0E0E10',
                        font=dict(color='#E2E2E2'),
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=True, gridcolor='#3C4043', tickprefix='\$'),
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error analyzing stock: {e}")