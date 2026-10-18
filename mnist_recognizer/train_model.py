import os
import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

def train_and_save_model():
    print("Downloading MNIST dataset (this may take a minute)...")
    # Fetch standard 28x28 pixel MNIST dataset
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    
    # Convert to numpy array and scale pixel values from 0-255 down to 0-1
    X = np.array(mnist.data, dtype=np.float32) / 255.0
    y = np.array(mnist.target, dtype=int)

    # To keep training fast on local machines, we use a 20,000 sample subset
    # (Change train_size if you want higher accuracy and have time to wait)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=20000, random_state=42)

    print("Training Basic Neural Network (MLPClassifier)...")
    # Configure a Neural Network with 1 hidden layer containing 128 neurons
    model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=30, 
                          solver='adam', verbose=True, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"Model Validation Accuracy: {model.score(X_test, y_test) * 100:.2f}%")

    # Create model directory and export
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/mnist_nn_model.pkl')
    print("✅ Model trained and saved successfully to 'model/mnist_nn_model.pkl'")

if __name__ == "__main__":
    train_and_save_model()