import anthropic
import pandas as pd
import json
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

os.makedirs("data/processed", exist_ok=True)

def generate_examples(label: int, description: str, n: int = 50):
    prompts = {
        0: f"Genere {n} posts Bluesky CREDIBLES en français. Posts factuels, ton neutre, sujets : sante, science, politique, environnement. Chaque post fait 1-3 phrases.",
        1: f"Genere {n} posts Bluesky DOUTEUX en français. Exagerations, sources floues, emotions fortes sans preuves. Sujets : vaccins, politique, medias. Chaque post fait 1-3 phrases.",
        2: f"Genere {n} posts Bluesky qui sont des FAKE NEWS en français. Affirmations fausses, theories du complot, manipulation evidente. Sujets : 5G, vaccins, elections, WEF. Chaque post fait 1-3 phrases."
    }

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=8000,
        messages=[{
            "role": "user",
            "content": f"""{prompts[label]}

IMPORTANT : Reponds UNIQUEMENT avec du JSON brut, sans markdown, sans backticks, sans explication.
Format exact : {{"examples": ["post1", "post2", "post3"]}}"""
        }]
    )

    text = message.content[0].text.strip()

    # Nettoyer les backticks si presents
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    print(f"  Reponse recue ({len(text)} chars)")
    data = json.loads(text)
    return [{"text": ex, "label": label, "source": "synthetic_fr"} for ex in data["examples"]]


all_examples = []

for label, description in [(0, "credible"), (1, "douteux"), (2, "fake news")]:
    print(f"Generation exemples '{description}'...")
    examples = generate_examples(label, description, n=50)
    all_examples.extend(examples)
    print(f"OK : {len(examples)} exemples generes")

df = pd.DataFrame(all_examples)
df.to_csv("data/processed/synthetic_french.csv", index=False)
print(f"\nTotal : {len(df)} exemples français sauvegardes")
print(df["label"].value_counts())