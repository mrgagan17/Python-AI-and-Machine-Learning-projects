import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# 1. Define the Convolutional Neural Network (CNN)
class TrafficSignCNN(nn.Module):
    def __init__(self):
        super(TrafficSignCNN, self).__init__()
        # Input: 3 color channels (RGB), 32x32 image
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 43) # 43 traffic sign classes
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8) # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_cnn():
    print("1. Server offline. Generating Synthetic PyTorch Data for Portfolio Demonstration...")
    # Generate 1,000 synthetic 32x32 RGB images to bypass the broken dataset download
    synthetic_images = torch.randn(1000, 3, 32, 32)
    # Generate 1,000 corresponding labels (Classes 0 to 42)
    synthetic_labels = torch.randint(0, 43, (1000,))
    
    # Load into PyTorch
    train_data = TensorDataset(synthetic_images, synthetic_labels)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    print("2. Initializing Neural Network...")
    model = TrafficSignCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("3. Training Model (1 Epoch)...")
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f"   Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    print("4. Saving Trained Model...")
    os.makedirs('model', exist_ok=True)
    torch.save(model.state_dict(), 'model/traffic_cnn.pth')
    print("✅ Model successfully saved to 'model/traffic_cnn.pth'")

if __name__ == "__main__":
    train_cnn()