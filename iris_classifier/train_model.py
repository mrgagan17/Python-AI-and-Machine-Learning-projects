import os
import joblib
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def train_and_save_model():
    # Load the Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the SVM Classifier
    # probability=True is required to show confidence percentages in the UI
    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_train, y_train)

    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Save the trained model
    joblib.dump(model, 'model/iris_svm_model.pkl')
    print("✅ Model trained successfully and saved to 'model/iris_svm_model.pkl'")

if __name__ == "__main__":
    train_and_save_model()