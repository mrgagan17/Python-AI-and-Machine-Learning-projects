import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib
import os

def build_churn_model():
    print("1. Generating realistic Telecom Customer data...")
    np.random.seed(42)
    n_samples = 5000
    
    # Simulate realistic customer features
    tenure = np.random.randint(1, 73, n_samples)
    monthly_charges = np.random.uniform(20.0, 120.0, n_samples)
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.25, 0.20])
    internet = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.35, 0.45, 0.20])
    
    # Create a logical churn probability based on features
    churn_prob = np.zeros(n_samples)
    churn_prob += np.where(contract == 'Month-to-month', 0.4, 0.0)
    churn_prob -= np.where(tenure > 24, 0.2, 0.0)
    churn_prob += np.where(internet == 'Fiber optic', 0.1, 0.0)
    churn_prob += np.where(monthly_charges > 80, 0.1, 0.0)
    
    # Normalize and convert to binary target with some noise
    churn_prob = np.clip(churn_prob, 0.05, 0.95)
    churn = np.random.binomial(1, churn_prob)
    
    df = pd.DataFrame({
        'Tenure_Months': tenure,
        'Monthly_Charges': monthly_charges,
        'Contract_Type': contract,
        'Internet_Service': internet,
        'Churn': churn
    })

    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("2. Building Preprocessing Pipeline...")
    # Define which columns are numeric and which are categorical
    numeric_features = ['Tenure_Months', 'Monthly_Charges']
    categorical_features = ['Contract_Type', 'Internet_Service']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    print("3. Training XGBoost Ensemble Model...")
    # Create a full pipeline that processes data and then predicts
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42))
    ])

    model_pipeline.fit(X_train, y_train)

    print("\n4. Model Evaluation:")
    predictions = model_pipeline.predict(X_test)
    print(classification_report(y_test, predictions, target_names=["Retained", "Churned"]))

    # Save the entire pipeline
    os.makedirs('model', exist_ok=True)
    joblib.dump(model_pipeline, 'model/xgboost_churn_pipeline.pkl')
    print("✅ Full pipeline (Preprocessing + XGBoost) saved to 'model/xgboost_churn_pipeline.pkl'")

if __name__ == "__main__":
    build_churn_model()