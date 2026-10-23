# 📉 AI Customer Churn Predictor

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/Machine_Learning-XGBoost-9B72CB)
![Scikit-Learn](https://img.shields.io/badge/Data_Pipeline-Scikit_Learn-F7931E)

A business intelligence web application that leverages ensemble machine learning methods to predict whether a customer will cancel their subscription service (churn). 

By identifying at-risk customers in real-time, this tool allows retention teams to proactively intervene with targeted offers, directly impacting company revenue.

## 🚀 Key Features
*   **Ensemble Learning:** Utilizes `XGBoost` (Extreme Gradient Boosting) for high-accuracy tabular data classification.
*   **Automated Data Pipelines:** Implements `sklearn.pipeline.Pipeline` and `ColumnTransformer` to seamlessly handle standard scaling for numericals and one-hot encoding for categorical variables in a single integrated flow.
*   **Interactive Dashboard:** Streamlit UI allows stakeholders to manipulate customer parameters and view dynamic risk probabilities.
*   **Actionable Insights:** Translates raw ML probabilities into color-coded business alerts.

## 📂 Repository Structure
```text
├── app.py                     # Streamlit dashboard and risk analysis logic
├── train_model.py             # Data simulation, pipeline construction, and XGBoost training
├── requirements.txt           # Project dependencies
└── model/                     
    └── xgboost_churn_pipeline.pkl  # Compiled Scikit-Learn Pipeline
