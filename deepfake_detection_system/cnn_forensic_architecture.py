import torch
import torch.nn as nn
import torchvision.models as models

class DeepFakeForensicModel(nn.Module):
    """
    A spatio-temporal Deep Neural Network designed to detect deepfakes.
    Uses a CNN (ResNeXt50) to extract frame-by-frame spatial artifacts (blending errors),
    and an LSTM to analyze temporal inconsistencies (unnatural blinking patterns).
    """
    def __init__(self, sequence_length=20, hidden_size=256):
        super(DeepFakeForensicModel, self).__init__()
        self.sequence_length = sequence_length
        
        # 1. Spatial Feature Extractor (CNN)
        # ResNeXt50 is highly effective at capturing microscopic digital artifacts
        resnext = models.resnext50_32x4d(pretrained=True)
        # Remove the final classification layer
        self.feature_extractor = nn.Sequential(*list(resnext.children())[:-1])
        
        # 2. Temporal Analyzer (RNN/LSTM)
        # Analyzes the sequence of features to detect frame-to-frame manipulation
        self.lstm = nn.LSTM(2048, hidden_size, num_layers=2, batch_first=True, dropout=0.3)
        
        # 3. Binary Classifier (Real vs Fake)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_length, c, h, w = x.size()
        
        # Flatten batch and sequence to pass through the CNN
        x = x.view(batch_size * seq_length, c, h, w)
        
        # Extract spatial features from each frame
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1) 
        
        # Reshape back to sequence format for the LSTM
        features = features.view(batch_size, seq_length, -1)
        
        # Process temporal sequence
        lstm_out, _ = self.lstm(features)
        
        # Extract the final temporal state
        last_frame_out = lstm_out[:, -1, :]
        
        # Classify the sequence
        confidence_score = self.fc(last_frame_out)
        return self.sigmoid(confidence_score)