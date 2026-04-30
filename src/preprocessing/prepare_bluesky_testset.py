from supabase import create_client
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

os.makedirs("data/labeled", exist_ok=True)

db = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

result = db.table("posts").select("id, text, author, likes, reposts, query").execute()
df = pd.DataFrame(result.data)

print(f"OK : {len(df)} posts Bluesky recuperes")

df["label"] = None
df["source"] = "bluesky_real"

df.to_csv("data/labeled/bluesky_to_label.csv", index=False)
print("Sauvegarde -> data/labeled/bluesky_to_label.csv")
print(f"\nProchaine etape : labellise manuellement 50 posts dans ce fichier")
print("0 = credible | 1 = douteux | 2 = fake news")