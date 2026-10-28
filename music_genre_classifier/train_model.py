import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import os

def train_audio_model():
    print("1. Generating Synthetic Audio Features (MFCCs)...")
    # We use 20 MFCCs as this is the default for librosa.
    # Simulating 2000 audio tracks across 5 genres
    n_samples = 2000
    n_mfcc = 20 
    
    # Generate mock features simulating different audio frequencies
    X = np.random.randn(n_samples, n_mfcc)
    
    # Introduce patterns so the neural network can actually learn "genres"
    y = np.random.randint(0, 5, n_samples)
    for i in range(5):
        X[y == i] += np.random.randn(n_mfcc) * (i + 1)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("2. Scaling Audio Features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("3. Training Deep Neural Network (MLPClassifier)...")
    # MLPClassifier is highly effective for extracted audio features
    model = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500, random_state=42)
    model.fit(X_train_scaled, y_train)

    print("\n4. Model Evaluation:")
    genre_names = ["Rock", "Jazz", "Hip-Hop", "Classical", "Pop"]
    predictions = model.predict(X_test_scaled)
    print(classification_report(y_test, predictions, target_names=genre_names))

    print("5. Saving Model and Scaler...")
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/music_mlp_model.pkl')
    joblib.dump(scaler, 'model/audio_scaler.pkl')
    print("✅ Model saved to 'model/'")

if __name__ == "__main__":
    train_audio_model()