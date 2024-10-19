import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os


def train_model():
    # Get the absolute path to the dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, '..', 'spam_ham_dataset.csv')
    
    # Load the dataset
    dataset = pd.read_csv(dataset_path).drop(columns=['Unnamed: 0'])
    X = dataset['text']
    y = dataset['label_num']

    # Vectorize the text using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_tfidf = tfidf_vectorizer.fit_transform(X)

    # Train a Naive Bayes classifier
    model = MultinomialNB()
    model.fit(X_tfidf, y)

    # Save the model and vectorizer inside the app directory
    joblib.dump(model, os.path.join(base_dir, 'spam_model.pkl'))
    joblib.dump(tfidf_vectorizer, os.path.join(base_dir, 'tfidf_vectorizer.pkl'))

if __name__ == "__main__":
    train_model()
