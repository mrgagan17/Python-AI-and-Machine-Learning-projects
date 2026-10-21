import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import joblib

def download_and_train_model():
    print("Processing Collaborative Filtering Matrix...")
    
    # Check if manual download is in the correct place
    if not os.path.exists('data/ml-latest-small/movies.csv'):
        print("❌ Error: Dataset not found. Please place 'ml-latest-small' inside the 'data' folder.")
        return

    # Load local data
    movies = pd.read_csv('data/ml-latest-small/movies.csv')
    ratings = pd.read_csv('data/ml-latest-small/ratings.csv')

    # Merge ratings with movie titles
    movie_data = pd.merge(ratings, movies, on='movieId')

    # Filter out movies with less than 20 ratings to reduce noise
    rating_counts = pd.DataFrame(movie_data.groupby('title')['rating'].count())
    popular_movies = rating_counts[rating_counts['rating'] >= 20].index
    filtered_data = movie_data[movie_data['title'].isin(popular_movies)]

    # Create a Pivot Table: Rows=Movie Titles, Columns=User IDs, Values=Ratings
    movie_pivot = filtered_data.pivot_table(index='title', columns='userId', values='rating').fillna(0)
    
    # Convert to a sparse matrix for memory efficiency
    movie_sparse_matrix = csr_matrix(movie_pivot.values)

    print("Training Nearest Neighbors Model...")
    # Initialize and fit the NearestNeighbors model
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(movie_sparse_matrix)

    # Export
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/cf_knn_model.pkl')
    movie_pivot.to_pickle('model/movie_pivot.pkl')
    movies.to_pickle('model/movies_df.pkl')
    
    print("✅ Recommender system trained and saved to 'model/'")

if __name__ == "__main__":
    download_and_train_model()