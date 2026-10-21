import streamlit as st
import joblib
import pandas as pd
import os

# 1. Page Configuration
st.set_page_config(page_title="AI Movie Recommender", page_icon="🍿", layout="centered")

# 2. Custom CSS for Gemini-style UI
st.markdown("""
<style>
    .stApp { background-color: #0E0E10; color: #E2E2E2; }
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #4285F4, #9B72CB, #D96570);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', sans-serif;
    }
    
    .stSelectbox > div > div { background-color: #1A1A1E !important; color: #E2E2E2 !important; border: 1px solid #3C4043 !important; }
    
    .movie-card {
        background: linear-gradient(145deg, #1A1A1E 0%, #25252D 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #3C4043;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
        margin-bottom: 15px;
        transition: transform 0.2s;
    }
    .movie-card:hover { transform: translateY(-3px); border-color: #9B72CB; }
    
    .movie-title { font-size: 20px; font-weight: 700; color: #E2E2E2; margin-bottom: 5px; }
    .movie-genres { color: #9AA0A6; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; }
    .match-score { color: #4285F4; font-size: 14px; font-weight: 600; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# 3. Load Model and Data
@st.cache_resource
def load_system():
    model_path = 'model/cf_knn_model.pkl'
    pivot_path = 'model/movie_pivot.pkl'
    movies_path = 'model/movies_df.pkl'
    
    if not os.path.exists(model_path):
        st.error("Model not found! Please run 'python train_model.py' first.")
        st.stop()
        
    return joblib.load(model_path), pd.read_pickle(pivot_path), pd.read_pickle(movies_path)

model, movie_pivot, movies_df = load_system()
movie_list = movie_pivot.index.tolist()

# 4. UI Layout
st.title("🍿 AI Movie Recommender")
st.markdown("Powered by **Collaborative Filtering**. Select a movie you love, and the AI will analyze the rating patterns of thousands of similar users to recommend what you should watch next.")
st.markdown("---")

selected_movie = st.selectbox("Search for a movie you like:", movie_list)

# 5. Recommendation Logic
if st.button("Generate Recommendations ✦", use_container_width=True):
    with st.spinner("Analyzing viewing patterns..."):
        # Find the vector for the selected movie
        movie_index = movie_pivot.index.get_loc(selected_movie)
        distances, indices = model.kneighbors(movie_pivot.iloc[movie_index, :].values.reshape(1, -1), n_neighbors=6)
        
        st.markdown("### Top AI Matches")
        
        # Skip the first item (index 0) because it is the movie the user selected
        for i in range(1, len(distances.flatten())):
            rec_title = movie_pivot.index[indices.flatten()[i]]
            similarity_score = (1 - distances.flatten()[i]) * 100
            
            # Fetch genres for the recommended movie
            # Movie titles in the dataset include the year of release in parentheses
            # Genres are a pipe-separated list
            genres = movies_df[movies_df['title'] == rec_title]['genres'].values[0].replace("|", " • ")
            
            # Render HTML Card
            st.markdown(f"""
            <div class="movie-card">
                <div class="movie-title">{rec_title}</div>
                <div class="movie-genres">{genres}</div>
                <div class="match-score">Match Profile: {similarity_score:.1f}% Similarity</div>
            </div>
            """, unsafe_allow_html=True)