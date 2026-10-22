import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib
import os

def build_fraud_model():
    print("1. Generating highly imbalanced transaction dataset (98% Safe / 2% Fraud)...")
    # Simulating the anonymized PCA features (V1-V11) and Amount found in real banking datasets
    X, y = make_classification(
        n_samples=25000, 
        n_features=12, 
        n_informative=6,
        n_redundant=2, 
        weights=[0.98, 0.02], # 98% majority class, 2% minority class
        random_state=42
    )

    columns = ['Amount', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11']
    df = pd.DataFrame(X, columns=columns)
    
    # Transform 'Amount' into realistic absolute dollar values
    df['Amount'] = np.abs(df['Amount']) * 150 

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
    
    print(f"   Original Training Distribution: {np.bincount(y_train)} (Safe: {np.bincount(y_train)[0]}, Fraud: {np.bincount(y_train)[1]})")

    print("\n2. Applying SMOTE to synthesize minority class data...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"   Resampled Training Distribution: {np.bincount(y_train_resampled)} (Perfectly Balanced!)")

    print("\n3. Training Random Forest Classifier on balanced data...")
    # n_jobs=-1 uses all CPU cores for faster training
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train_resampled, y_train_resampled)

    print("\n4. Testing the model on unseen, imbalanced test data...")
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions, target_names=["Safe", "Fraud"]))

    # Save the model
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/smote_fraud_model.pkl')
    print("✅ Model trained and saved to 'model/smote_fraud_model.pkl'")

if __name__ == "__main__":
    build_fraud_model()