import pandas as pd
from sklearn.model_selection import train_test_split
import os

os.makedirs("data/processed", exist_ok=True)

# Charger les sources
liar = pd.read_csv("data/processed/liar_clean.csv")[["text", "label"]]
liar["source"] = "liar_en"

bluesky = pd.read_csv("data/labeled/bluesky_clean.csv")[["text", "label"]]
bluesky["source"] = "bluesky_real"

# Fusionner
df = pd.concat([liar, bluesky], ignore_index=True)
df = df.dropna(subset=["text", "label"])
df["label"] = df["label"].astype(int)
df["text"] = df["text"].astype(str)

print(f"Dataset total : {len(df)} exemples")
print(df["label"].value_counts())
print(df["source"].value_counts())

# Les posts Bluesky reels → test set exclusif
test_df = bluesky.copy()

# LIAR → train + validation
train_df, val_df = train_test_split(
    liar,
    test_size=0.15,
    random_state=42,
    stratify=liar["label"]
)

train_df.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/val.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print(f"\nTrain   : {len(train_df)} exemples")
print(f"Val     : {len(val_df)} exemples")
print(f"Test    : {len(test_df)} exemples (posts Bluesky reels)")
print("\nSauvegarde -> data/processed/")