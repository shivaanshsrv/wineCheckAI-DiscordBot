<div align="center">

<h1>🍷 Wine & Food AI — Discord Bot</h1>

<p>
A complete AI-powered wine & food pairing system built with  
<strong>Machine Learning · Recommender Systems · Clustering · Discord Slash Commands</strong>
</p>

<br/>

<img src="https://img.shields.io/badge/Python-3.10+-blue.svg" />
<img src="https://img.shields.io/badge/Discord.py-2.x-purple.svg" />
<img src="https://img.shields.io/badge/ML-TF--IDF%20|%20KMeans-green.svg" />
<img src="https://img.shields.io/badge/Platform-Discord-black.svg" />

<br/><br/>

<a href="#-setup-guide"><img src="https://img.shields.io/badge/⬇️ Setup Guide-blue?style=for-the-badge" /></a>
<a href="#-features"><img src="https://img.shields.io/badge/✨ Features-purple?style=for-the-badge" /></a>
<a href="#-slash-commands"><img src="https://img.shields.io/badge/💬 Slash Commands-green?style=for-the-badge" /></a>

<br/><br/>

</div>

---

<div align="center">
<h2>🚀 What This Project Does</h2>
</div>

This AI system provides:
- **Food recommendations for any wine**  
- **Wine recommendations for any food**  
- **Wine cluster analysis using ML**  
- **Beautiful Discord slash commands**  
- Full ML pipeline: cleaning → training → clustering → bot

---

## 📁 Project Structure

```plaintext
wine-food-ai/
│
├── 1_data_cleaning.py
├── 2_food_recommender.py
├── 3_wine_recommender.py
├── 4_clustering_model.py
├── 5_chatbot_assistant.py  # SLASH COMMAND VERSION
│
├── data/
│   └── cleaned_wine_food.csv
│
├── models/
│   ├── tfidf_wine.pkl
│   ├── tfidf_food.pkl
│   ├── tfidf_cluster.pkl
│   ├── kmeans_model.pkl
│   └── clusters.csv
│
├── .env
├── requirements.txt
└── README.md
