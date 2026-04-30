import pandas as pd
import os

os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv("data/raw/liar_raw.csv")
print(f"Dataset brut : {len(df)} exemples")

# Mapping 6 labels -> 3 labels
label_mapping = {
    "true": 0,
    "mostly-true": 0,
    "half-true": 1,
    "barely-true": 1,
    "false": 2,
    "pants-fire": 2
}

df["label"] = df["label"].map(label_mapping)
df = df.dropna(subset=["label", "text"])
df["label"] = df["label"].astype(int)

# Garder uniquement ce qu'il faut
df_clean = df[["text", "label", "subject"]].copy()

# Filtrer sujets pertinents
sujets = [
    "health", "science", "conspiracy", "elections",
    "environment", "immigration", "coronavirus", "vaccines",
    "economy", "technology", "media"
]

mask = df_clean["subject"].str.contains(
    "|".join(sujets), case=False, na=False
)
df_filtered = df_clean[mask].copy()

# Equilibrer les classes
min_count = df_filtered["label"].value_counts().min()
df_balanced = df_filtered.groupby("label").apply(
    lambda x: x.sample(min_count, random_state=42)
).reset_index(drop=True)

print(f"Dataset filtre : {len(df_balanced)} exemples")
print(df_balanced["label"].value_counts())

df_balanced.to_csv("data/processed/liar_clean.csv", index=False)
print("Sauvegarde -> data/processed/liar_clean.csv")