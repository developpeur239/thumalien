import anthropic
import pandas as pd
import os
import time
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

df = pd.read_csv("data/labeled/bluesky_labeled.csv")
print(f"Dataset charge : {len(df)} posts")
print(f"Distribution initiale :")
print(df["label"].value_counts())

# ─── ETAPE 1 : Supprimer les posts inutilisables ───────────────────────────

# Posts trop courts
mask_court = df["text"].str.len() < 20
# Posts qui ne sont que des hashtags
mask_hashtags = df["text"].str.match(r'^[#@\s]+$', na=False)

df_clean = df[~mask_court & ~mask_hashtags].copy()
print(f"\nApres nettoyage basique : {len(df_clean)} posts")

# ─── ETAPE 2 : Recorriger les cas ambigus ──────────────────────────────────
# Les posts collectes avec "complot" ne sont pas tous des fake news
# On reanalysé ceux qui semblent mal classes

def relabel_ambiguous(text: str, current_label: int, query: str) -> int:
    """Reclassifie les posts potentiellement mal labels"""
    try:
        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=10,
            messages=[{
                "role": "user",
                "content": f"""Expert en detection de fake news. Analyse ce post.

CONTEXTE : Ce post a ete collecte avec le mot-cle "{query}".

REGLES STRICTES :
- 0 = Le post RAPPORTE ou ANALYSE un sujet sans affirmer de fausses informations
- 1 = Le post est ambigu, exagere ou manque de sources
- 2 = Le post AFFIRME activement quelque chose de faux ou promote un complot

Post: {text[:400]}

Reponds UNIQUEMENT par 0, 1 ou 2."""
            }]
        )
        result = message.content[0].text.strip()
        if result in ["0", "1", "2"]:
            return int(result)
        return current_label
    except Exception as e:
        print(f"Erreur : {e}")
        return current_label

# Recorriger uniquement les posts suspects :
# - Labels 0 ou 2 collectes avec "complot" (souvent mal classes)
# - Posts tres longs (Claude a peut-etre mal lu)
mask_suspects = (
    (df_clean["query"] == "complot") & (df_clean["label"] != 1)
) | (
    df_clean["text"].str.len() > 400
)

suspects = df_clean[mask_suspects]
print(f"\nPosts a recorriger : {len(suspects)}")

corrections = 0
for idx, row in suspects.iterrows():
    new_label = relabel_ambiguous(
        str(row["text"]), 
        int(row["label"]), 
        str(row["query"])
    )
    if new_label != row["label"]:
        df_clean.at[idx, "label"] = new_label
        corrections += 1
    time.sleep(0.3)

print(f"Corrections apportees : {corrections} posts")

# ─── ETAPE 3 : Statistiques finales ───────────────────────────────────────

print(f"\nDistribution finale :")
print(df_clean["label"].value_counts())
print(f"\nPourcentages :")
print((df_clean["label"].value_counts(normalize=True) * 100).round(1))

# ─── ETAPE 4 : Sauvegarder ────────────────────────────────────────────────

os.makedirs("data/labeled", exist_ok=True)
df_clean.to_csv("data/labeled/bluesky_clean.csv", index=False)
print(f"\nSauvegarde -> data/labeled/bluesky_clean.csv")
print(f"Total final : {len(df_clean)} posts de qualite")