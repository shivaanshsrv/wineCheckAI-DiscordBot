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
```

<div align="center"> <h2>✨ Features</h2> </div> <div align="center"> <table> <tr> <td>🍽️ Recommend Food</td> <td>🍷 Recommend Wine</td> <td>📊 Wine Clustering</td> </tr> <tr> <td>✔ TF-IDF Similarity</td> <td>✔ KMeans Groups</td> <td>✔ Clean Dataset Pipeline</td> </tr> <tr> <td>⚡ Slash Commands</td> <td>🧠 Content-Based Filtering</td> <td>📦 Fully Modular Code</td> </tr> </table> </div>

💬 Slash Commands
/food Pinot Noir
/wine Butter Chicken
/cluster Merlot


All commands return a styled Embed with your ML model predictions.

🌐 Discord Bot Token Setup (HTML Styled)
<ol>
  <li>Go to <a href="https://discord.com/developers/applications">Discord Developers Portal</a></li>
  <li>Create a new Application → Add Bot</li>
  <li>Enable:
    <ul>
      <li>Server Members Intent</li>
      <li>Message Content Intent (optional)</li>
    </ul>
  </li>
  <li>Click <strong>Reset Token</strong> → Copy Token</li>
  <li>Create a file named <code>.env</code> in project root with:</li>
</ol>

DISCORD_TOKEN=YOUR_TOKEN_HERE

<h2>⬇️ Setup Guide</h2>

<h3>1️⃣ Install Dependencies</h3>

<pre><code>pip install -r requirements.txt
</code></pre>

<h3>2️⃣ Run the ML Pipeline</h3>

<pre><code>python data_cleaning.py
python food_recommender.py
python wine_recommender.py
python clustering_model.py
</code></pre>

<h3>3️⃣ Run the Discord Bot</h3>

<pre><code>python chatbot_assistant.py
</code></pre>


<div align="center"> <h2>⭐ Enjoy the Project!</h2> <p>If you found this useful, consider starring the repo 🌟</p> </div> 
