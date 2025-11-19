# 1_data_cleaning.py
"""
Loads /mnt/data/wine_food_pairings.csv, attempts to detect wine/food text columns,
cleans text, deduplicates, and writes cleaned_wine_food.csv to ./data/
"""
import os
import pandas as pd
import re

RAW_PATH = r"C:\Users\hp\Downloads\Diamonds\shivansh\WineCheck DC\wine_food_pairings.csv"
OUT_DIR = "data"
OUT_PATH = os.path.join(OUT_DIR, "cleaned_wine_food.csv")

os.makedirs(OUT_DIR, exist_ok=True)

def detect_columns(df):
    """Try to find wine and food columns by common names."""
    cols = [c.lower() for c in df.columns]
    wine_candidates = [c for c in df.columns if any(k in c.lower() for k in ["wine", "variet", "variety", "bottle", "label"])]
    food_candidates = [c for c in df.columns if any(k in c.lower() for k in ["food", "pair", "pairing", "dish", "serve", "serve_with"])]
    desc_candidates = [c for c in df.columns if "description" in c.lower() or "note" in c.lower() or "tasting" in c.lower()]
    return wine_candidates, food_candidates, desc_candidates

def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)
    return s

def main():
    print("Loading raw dataset:", RAW_PATH)
    df = pd.read_csv(RAW_PATH)
    print("Columns found:", df.columns.tolist())

    wine_cols, food_cols, desc_cols = detect_columns(df)
    print("Detected wine columns:", wine_cols)
    print("Detected food columns:", food_cols)
    print("Detected desc columns:", desc_cols)

    # Create canonical columns if possible
    if wine_cols:
        df['wine'] = df[wine_cols[0]].astype(str)
    else:
        # fallback: use first column
        df['wine'] = df.iloc[:, 0].astype(str)

    if food_cols:
        df['food'] = df[food_cols[0]].astype(str)
    else:
        # fallback: try second column if exists
        if df.shape[1] > 1:
            df['food'] = df.iloc[:, 1].astype(str)
        else:
            df['food'] = ""

    # optional description
    if desc_cols:
        df['description'] = df[desc_cols[0]].astype(str)
    else:
        # create combined description from all text-like columns excluding wine/food
        text_cols = [c for c in df.columns if df[c].dtype == object and c not in ('wine', 'food', 'description')]
        df['description'] = df[text_cols].astype(str).agg(' '.join, axis=1) if text_cols else ""

    # Clean
    df['wine'] = df['wine'].apply(clean_text).str.title()
    df['food'] = df['food'].apply(clean_text)
    df['description'] = df['description'].apply(clean_text)

    # Drop rows missing both wine and food
    df = df[~((df['wine'].str.len()==0) & (df['food'].str.len()==0))].copy()

    # Remove duplicates (case-insensitive)
    df['wine_lower'] = df['wine'].str.lower()
    df['food_lower'] = df['food'].str.lower()
    df = df.drop_duplicates(subset=['wine_lower','food_lower'])
    df = df.drop(columns=['wine_lower','food_lower'], errors='ignore')

    # Reset index and save
    df = df.reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Cleaned data saved to {OUT_PATH}. Rows: {len(df)}")

if __name__ == "__main__":
    main()
