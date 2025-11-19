# 3_wine_recommender.py
"""
Given a food/dish name, return top wine recommendations.
Uses TF-IDF on 'food' text and cosine similarity.
"""
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import joblib

DATA_PATH = "data/cleaned_wine_food.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

VECT_PATH = os.path.join(MODEL_DIR, "tfidf_food.pkl")
DF_PATH = os.path.join(MODEL_DIR, "df_wine_index.pkl")

def prepare():
    df = pd.read_csv(DATA_PATH)
    df['food_text'] = (df['food'].fillna('') + " " + df['description'].fillna('')).astype(str)
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=5000)
    X = vectorizer.fit_transform(df['food_text'])
    joblib.dump(vectorizer, VECT_PATH)
    joblib.dump(df, DF_PATH)
    print("Wine recommender prepared. Models saved to models/")

def recommend_wine(food_name, top_n=5):
    food_name = str(food_name).strip()
    if not food_name:
        return []
    vectorizer = joblib.load(VECT_PATH)
    df = joblib.load(DF_PATH)
    X = vectorizer.transform(df['food_text'])
    query_vec = vectorizer.transform([food_name])
    sims = linear_kernel(query_vec, X).flatten()
    top_idx = sims.argsort()[::-1][:top_n*3]
    wines = []
    seen = set()
    for idx in top_idx:
        wine = str(df.iloc[idx]['wine']).strip()
        if wine and wine.lower() not in seen:
            wines.append(wine)
            seen.add(wine.lower())
        if len(wines) >= top_n:
            break
    return wines

if __name__ == "__main__":
    if not os.path.exists(VECT_PATH):
        print("Preparing model...")
        prepare()
    print(recommend_wine("Grilled Salmon", top_n=6))
