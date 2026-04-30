import anthropic
import pandas as pd
import os
import time
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

df = pd.read_csv("data/labeled/bluesky_to_label.csv")
print(f"Posts a labelliser : {len(df)}")

def label_post(text: str) -> int:
    try:
        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=10,
            messages=[{
                "role": "user",
                "content": f"""Tu es un expert en detection de fake news.
Analyse ce post et reponds UNIQUEMENT par 0, 1 ou 2.

0 = Credible (factuel, source verifiable, ton neutre)
1 = Douteux (exagere, source floue, emotion sans preuve)
2 = Fake news (faux, complot, manipulation evidente)

Post: {text[:500]}

Reponds uniquement par le chiffre 0, 1 ou 2."""
            }]
        )
        result = message.content[0].text.strip()
        if result in ["0", "1", "2"]:
            return int(result)
        return 1
    except Exception as e:
        print(f"Erreur : {e}")
        return 1

labels = []
for i, row in df.iterrows():
    label = label_post(str(row["text"]))
    labels.append(label)
    if i % 10 == 0:
        print(f"Progress : {i}/{len(df)} posts labellises")
    time.sleep(0.5)

df["label"] = labels

print(f"\nDistribution des labels :")
print(df["label"].value_counts())

df.to_csv("data/labeled/bluesky_labeled.csv", index=False)
print("Sauvegarde -> data/labeled/bluesky_labeled.csv")