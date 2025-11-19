# 5_chatbot_assistant.py
"""
Discord bot with SLASH COMMANDS:
/food <wine>
/wine <food>
/cluster <wine>

Uses your existing models from:
- 2_food_recommender.py
- 3_wine_recommender.py
- 4_clustering_model.py
"""

import os
import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
import joblib
import pandas as pd
import importlib.util

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
if not TOKEN:
    raise RuntimeError("DISCORD_TOKEN not found in .env")

MODEL_DIR = "models"
DATA_PATH = "data/cleaned_wine_food.csv"

# Function to load Python scripts dynamically
def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Load recommenders
food_mod = load_module("2_food_recommender.py", "food_reco")
wine_mod = load_module("3_wine_recommender.py", "wine_reco")

# Load clustering model + vectorizer
CLUSTER_MODEL = None
CLUSTER_VECTOR = None
CLUSTER_MODEL_PATH = os.path.join(MODEL_DIR, "kmeans_model.pkl")
CLUSTER_VECT_PATH = os.path.join(MODEL_DIR, "tfidf_cluster.pkl")

if os.path.exists(CLUSTER_MODEL_PATH):
    CLUSTER_MODEL = joblib.load(CLUSTER_MODEL_PATH)

if os.path.exists(CLUSTER_VECT_PATH):
    CLUSTER_VECTOR = joblib.load(CLUSTER_VECT_PATH)

# Load dataset
df = pd.read_csv(DATA_PATH)

intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)


# -------------- SLASH COMMAND SETUP ---------------- #

@bot.event
async def on_ready():
    print(f"Bot connected as {bot.user}")
    print("Syncing slash commands...")
    try:
        synced = await bot.tree.sync()
        print(f"Slash commands synced: {len(synced)}")
    except Exception as e:
        print("Error syncing commands:", e)



# ------------------- SLASH COMMANDS ------------------- #

@bot.tree.command(name="food", description="Recommend foods for a given wine.")
async def food(interaction: discord.Interaction, wine: str):
    await interaction.response.defer()

    recs = food_mod.recommend_food(wine, top_n=6)

    if not recs:
        await interaction.followup.send(f"❌ No food recommendations found for **{wine}**.")
        return

    embed = discord.Embed(
        title=f"Food Pairings for {wine.title()}",
        description="\n".join(f"• {r}" for r in recs),
        color=0x00ff99
    )
    await interaction.followup.send(embed=embed)



@bot.tree.command(name="wine", description="Recommend wines for a given food.")
async def wine(interaction: discord.Interaction, food: str):
    await interaction.response.defer()

    recs = wine_mod.recommend_wine(food, top_n=6)

    if not recs:
        await interaction.followup.send(f"❌ No wines found for **{food}**.")
        return

    embed = discord.Embed(
        title=f"Wines for {food.title()}",
        description="\n".join(f"• {r}" for r in recs),
        color=0x0099ff
    )
    await interaction.followup.send(embed=embed)



@bot.tree.command(name="cluster", description="Get the cluster and similar wines.")
async def cluster(interaction: discord.Interaction, wine: str):
    await interaction.response.defer()

    if CLUSTER_MODEL is None or CLUSTER_VECTOR is None:
        await interaction.followup.send("❌ Clustering model not found. Run 4_clustering_model.py first.")
        return

    vec = CLUSTER_VECTOR.transform([wine])
    label = CLUSTER_MODEL.predict(vec)[0]

    # Wines in same cluster
    similar = df[df["cluster"] == label]["wine"].unique().tolist()[:10]

    embed = discord.Embed(
        title=f"Cluster Result for {wine.title()}",
        description=f"Cluster ID: **{label}**",
        color=0xffcc00
    )

    if similar:
        embed.add_field(
            name="Similar Wines",
            value="\n".join(f"• {w}" for w in similar),
            inline=False
        )
    else:
        embed.add_field(
            name="Similar Wines",
            value="No similar wines found.",
            inline=False
        )

    await interaction.followup.send(embed=embed)



# ------------------- RUN BOT ------------------- #

bot.run(TOKEN)
