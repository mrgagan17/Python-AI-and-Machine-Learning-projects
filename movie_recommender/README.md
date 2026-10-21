# 🍿 AI Movie Recommendation Engine

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?logo=scikit-learn&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Algorithm-Collaborative_Filtering-9B72CB)

A Machine Learning web application that recommends movies based on user rating patterns, replicating the core **Item-Based Collaborative Filtering** algorithms used by streaming platforms.

Built using the widely recognized MovieLens dataset, the AI leverages Cosine Similarity through a K-Nearest Neighbors (KNN) model to analyze multi-dimensional user preference vectors and surface highly correlated films.

## 🚀 Key Features
*   **Collaborative Filtering:** Recommends content based on the collective behavior of similar users rather than relying solely on metadata (like genres).
*   **Vector Similarity:** Calculates distance matrices utilizing the Cosine Similarity metric to find the closest movie matches.
*   **Pre-Trained Execution:** The repository includes pre-compiled `.pkl` model files, meaning the application runs instantly without requiring users to download or process the raw 100,000-row dataset.
*   **Modern UI:** Custom dark-themed user interface featuring glassmorphism recommendation cards and a searchable dropdown menu.

## 📂 Repository Structure
```text
├── app.py                     # Streamlit frontend and HTML/CSS UI components
├── train_model.py             # Script used to compile the matrix and train the KNN model
├── requirements.txt           # Project dependencies
└── model/                     
    ├── cf_knn_model.pkl       # Pre-trained Nearest Neighbors model
    ├── movie_pivot.pkl        # Compressed sparse matrix mapping
    └── movies_df.pkl          # Compressed movie metadata for UI rendering
