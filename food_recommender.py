import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import joblib

DATA_PATH = "data/cleaned_wine_food.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

VECT_PATH = os.path.join(MODEL_DIR, "tfidf_wine.pkl")
DF_PATH = os.path.join(MODEL_DIR, "df_food_index.pkl")

def prepare():
    df = pd.read_csv(DATA_PATH)
    # create a text field to vectorize
    df['wine_text'] = (df['wine'].fillna('') + " " + df['description'].fillna('')).astype(str)
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=5000)
    X = vectorizer.fit_transform(df['wine_text'])
    joblib.dump(vectorizer, VECT_PATH)
    joblib.dump(df, DF_PATH)
    print("Food recommender prepared. Models saved to models/")

def recommend_food(wine_name, top_n=5):
    wine_name = str(wine_name).strip()
    if not wine_name:
        return []
    # load
    vectorizer = joblib.load(VECT_PATH)
    df = joblib.load(DF_PATH)
    X = vectorizer.transform(df['wine_text'])
    query_vec = vectorizer.transform([wine_name])
    sims = linear_kernel(query_vec, X).flatten()
    top_idx = sims.argsort()[::-1][:top_n*3] 
    foods = []
    seen = set()
    for idx in top_idx:
        food = str(df.iloc[idx]['food']).strip()
        if food and food.lower() not in seen:
            foods.append(food)
            seen.add(food.lower())
        if len(foods) >= top_n:
            break
    return foods

if __name__ == "__main__":
    if not os.path.exists(VECT_PATH):
        print("Preparing model...")
        prepare()
    print(recommend_food("Pinot Noir", top_n=6))
