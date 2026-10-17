import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def train_and_save_model():
    print("Downloading dataset...")
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    
    # Load dataset directly from the URL
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
    
    # Convert labels: 'ham' -> 0, 'spam' -> 1
    df['label_num'] = df.label.map({'ham': 0, 'spam': 1})
    
    X = df['message']
    y = df['label_num']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize CountVectorizer and Naive Bayes
    vectorizer = CountVectorizer()
    model = MultinomialNB()

    print("Training NLP model...")
    # Transform text into numerical vectors and train
    X_train_dtm = vectorizer.fit_transform(X_train)
    model.fit(X_train_dtm, y_train)

    # Create model directory
    os.makedirs('model', exist_ok=True)
    
    # Save both the vectorizer (needed to process new text) and the trained model
    joblib.dump(vectorizer, 'model/count_vectorizer.pkl')
    joblib.dump(model, 'model/spam_nb_model.pkl')
    print("✅ Model and Vectorizer trained and saved successfully to 'model/'")

if __name__ == "__main__":
    train_and_save_model()