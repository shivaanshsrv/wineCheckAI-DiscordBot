# Performs KMeans clustering on combined wine text (wine + description).

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import joblib

DATA_PATH = "data/cleaned_wine_food.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

VECT_PATH = os.path.join(MODEL_DIR, "tfidf_cluster.pkl")
KM_PATH = os.path.join(MODEL_DIR, "kmeans_model.pkl")
OUT_CSV = os.path.join(MODEL_DIR, "clusters.csv")

def main(n_clusters=8, random_state=42):
    df = pd.read_csv(DATA_PATH)
    df['text'] = (df['wine'].fillna('') + " " + df['description'].fillna('')).astype(str)
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=8000)
    X = vectorizer.fit_transform(df['text'])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    df['cluster'] = labels
    df.to_csv(OUT_CSV, index=False)
    joblib.dump(vectorizer, VECT_PATH)
    joblib.dump(kmeans, KM_PATH)
    print(f"Clustering done. clusters.csv saved ({len(df)} rows). Models saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()
