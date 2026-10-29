import torch
import torch.nn as nn
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

# 1. Define the LSTM Neural Network Architecture
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Fully connected layer to output the final price prediction
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        
        # Forward propagate LSTM
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

def train_stock_model():
    print("1. Fetching Historical Stock Data (AAPL)...")
    ticker = "AAPL"
    data = yf.download(ticker, start="2018-01-01", end="2024-01-01")
    
    # We only care about the 'Close' price for this predictor
    close_prices = data[['Close']].values

    print("2. Scaling and Sequencing Data...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Create sequences: Use 60 days of historical data to predict the 61st day
    sequence_length = 60
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
        
    X, y = np.array(X), np.array(y)
    
    # Reshape X to fit PyTorch LSTM input shape (Samples, TimeSteps, Features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    print("3. Training LSTM Neural Network (This will take a minute on CPU)...")
    model = StockLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 5 == 0:
            print(f'   Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

    print("4. Saving Trained Model...")
    os.makedirs('model', exist_ok=True)
    torch.save(model.state_dict(), 'model/stock_lstm.pth')
    print("✅ Model successfully saved to 'model/stock_lstm.pth'")

if __name__ == "__main__":
    train_stock_model()