import pandas as pd
from sklearn.model_selection import train_test_split
import os

os.makedirs("data/processed", exist_ok=True)

liar = pd.read_csv("data/processed/liar_clean.csv")[["text", "label"]]
liar["source"] = "liar_en"

synthetic = pd.read_csv("data/processed/synthetic_french.csv")[["text", "label"]]
synthetic["source"] = "synthetic_fr"

more_fake = pd.read_csv("data/processed/more_fake_news.csv")[["text", "label"]]
more_fake["source"] = "synthetic_fr"

bluesky = pd.read_csv("data/labeled/bluesky_clean.csv")[["text", "label"]]
bluesky["source"] = "bluesky_real"

train_source = pd.concat([liar, synthetic, more_fake], ignore_index=True)
train_source = train_source.dropna(subset=["text", "label"])
train_source["label"] = train_source["label"].astype(int)
train_source["text"] = train_source["text"].astype(str)

train_df, val_df = train_test_split(
    train_source, test_size=0.15, random_state=42, stratify=train_source["label"]
)

test_df = bluesky.copy()
test_df["label"] = test_df["label"].astype(int)

train_df.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/val.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print(f"Train   : {len(train_df)} exemples")
print(f"Val     : {len(val_df)} exemples")
print(f"Test    : {len(test_df)} exemples")
print(f"\nDistribution train:")
print(train_df["label"].value_counts())
print(f"\nSources:")
print(train_df["source"].value_counts())