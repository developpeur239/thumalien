from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import uvicorn

app = FastAPI(title="Thumalien API", description="Detection de Fake News sur Bluesky")

print("Chargement du modele...")
tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForSequenceClassification.from_pretrained("models/camembert_thumalien")
model.eval()
print("Modele charge")

LABELS = {0: "Credible", 1: "Douteux", 2: "Fake News"}

class TextInput(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok", "model": "thumalien-fake-news"}

@app.post("/analyze")
def analyze(input: TextInput):
    inputs = tokenizer(input.text[:512], return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0].tolist()
    label_id = probs.index(max(probs))
    return {
        "text": input.text[:100],
        "label": LABELS[label_id],
        "confidence": round(max(probs), 3),
        "scores": {
            "credible": round(probs[0], 3),
            "douteux": round(probs[1], 3),
            "fake_news": round(probs[2], 3)
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)