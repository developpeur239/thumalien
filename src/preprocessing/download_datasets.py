import pandas as pd
import requests, zipfile, io, os

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

print("Telechargement LIAR dataset officiel...")

url = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall("data/raw/liar_zip")
print("Archive extraite")

columns = [
    "id", "label", "text", "subject", "speaker",
    "job", "state", "party", "barely_true", "false_count",
    "half_true", "mostly_true", "pants_fire", "context"
]

dfs = []
for filename in ["train.tsv", "test.tsv", "valid.tsv"]:
    path = f"data/raw/liar_zip/{filename}"
    df = pd.read_csv(path, sep="\t", header=None, names=columns)
    dfs.append(df)
    print(f"  {filename} : {len(df)} exemples")

df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal : {len(df)} exemples")
print(df["label"].value_counts())

df.to_csv("data/raw/liar_raw.csv", index=False)
print("Sauvegarde -> data/raw/liar_raw.csv")