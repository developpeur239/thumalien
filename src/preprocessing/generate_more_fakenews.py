import anthropic
import json
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

all_examples = []

for batch in range(4):
    print(f"Batch {batch+1}/4...")
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": f"Genere exactement 40 posts Bluesky FAKE NEWS courts (max 100 caracteres chacun) en français. Sujets : 5G, vaccins, WEF, chemtrails, QAnon. JSON sans markdown : {{\"examples\": [\"post1\", \"post2\"]}}"
        }]
    )

    text = message.content[0].text.strip()

    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    print(f"  Reponse : {len(text)} chars")
    data = json.loads(text)
    all_examples.extend(data["examples"])
    print(f"  OK : {len(data['examples'])} exemples")

df = pd.DataFrame([
    {"text": ex, "label": 2, "source": "synthetic_fr"}
    for ex in all_examples
])

df.to_csv("data/processed/more_fake_news.csv", index=False)
print(f"\nTotal : {len(df)} fake news sauvegardees")