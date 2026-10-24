import streamlit as st
import tweepy
from textblob import TextBlob
import pandas as pd
import re
import os
from dotenv import load_dotenv
import random

# 1. Page Configuration & Custom UI
st.set_page_config(page_title="Twitter Sentiment Analyzer", page_icon="🐦", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0E0E10; color: #E2E2E2; }
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #1DA1F2, #9B72CB, #D96570);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', sans-serif;
    }
    .metric-box { background-color: #1A1A1E; padding: 20px; border-radius: 10px; border: 1px solid #3C4043; text-align: center; }
</style>
""", unsafe_allow_html=True)

# 2. API Configuration
load_dotenv()
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

def clean_tweet(tweet):
    # Remove URLs, @mentions, and special characters using Regex
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    return tweet.strip()

def get_sentiment(text):
    # TextBlob returns a polarity score from -1.0 to 1.0
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.1:
        return "Positive", analysis.sentiment.polarity
    elif analysis.sentiment.polarity < -0.1:
        return "Negative", analysis.sentiment.polarity
    else:
        return "Neutral", analysis.sentiment.polarity

def fetch_tweets(query, count=50):
    # Simulated data fallback if no API key is provided
    if not BEARER_TOKEN or BEARER_TOKEN == "your_twitter_bearer_token_here":
        st.warning("⚠️ No valid Twitter Bearer Token found in .env file. Using simulated data for demonstration.")
        templates = [
            f"I absolutely love the new {query}! Best thing ever. 🚀",
            f"The recent update regarding {query} is terrible. So disappointed. 😡",
            f"Just saw the news about {query}. It's okay, nothing special.",
            f"{query} is completely changing the industry! Amazing work.",
            f"Why does {query} always fail when I need it most? Worst experience.",
            f"Reading some thoughts on {query} today."
        ]
        return [{"text": random.choice(templates)} for _ in range(count)]

    # Real Tweepy API Call
    try:
        client = tweepy.Client(bearer_token=BEARER_TOKEN)
        # Search recent tweets (excluding retweets to reduce noise)
        response = client.search_recent_tweets(query=f"{query} -is:retweet", max_results=count)
        return response.data if response.data else []
    except Exception as e:
        st.error(f"Twitter API Error: {e}")
        return []

# 3. UI Layout
st.title("🐦 Live Twitter Sentiment Analyzer")
st.markdown("Scrape real-time tweets via the Twitter v2 API and analyze public sentiment using Natural Language Processing (NLP).")
st.markdown("---")

col1, col2 = st.columns([3, 1])
with col1:
    search_query = st.text_input("Enter a Keyword, Brand, or Topic (e.g., Artificial Intelligence, Apple, Election):")
with col2:
    tweet_count = st.slider("Number of Tweets", min_value=10, max_value=100, value=50, step=10)

if st.button("Analyze Sentiment ✦", use_container_width=True) and search_query:
    with st.spinner(f"Scraping the Twitterverse for '{search_query}'..."):
        
        raw_tweets = fetch_tweets(search_query, tweet_count)
        
        if raw_tweets:
            # Process Data
            data = []
            for tweet in raw_tweets:
                clean_text = clean_tweet(tweet.text if hasattr(tweet, 'text') else tweet['text'])
                sentiment_label, polarity = get_sentiment(clean_text)
                data.append({"Tweet": clean_text, "Sentiment": sentiment_label, "Polarity Score": round(polarity, 2)})
            
            df = pd.DataFrame(data)
            
            # Calculate Metrics
            pos_count = len(df[df['Sentiment'] == 'Positive'])
            neg_count = len(df[df['Sentiment'] == 'Negative'])
            neu_count = len(df[df['Sentiment'] == 'Neutral'])
            
            # Display Metrics
            st.markdown("### Overall Sentiment Distribution")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="metric-box"><h2 style="color: #34A853;">{pos_count}</h2><p>Positive Tweets</p></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-box"><h2 style="color: #9AA0A6;">{neu_count}</h2><p>Neutral Tweets</p></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-box"><h2 style="color: #EA4335;">{neg_count}</h2><p>Negative Tweets</p></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### Processed Data")
            st.dataframe(df, use_container_width=True)