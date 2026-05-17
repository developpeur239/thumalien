import os
os.system("pip install sentencepiece")

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import uvicorn

app = FastAPI(title="Thumalien API")

print("Chargement modeles...")
tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForSequenceClassification.from_pretrained(
    "thumalien/thumalien-fake-news"
)
model.eval()

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)
print("Modeles charges")

LABELS = {0: "Credible", 1: "Douteux", 2: "Fake News"}
EMOTION_MAP = {
    "anger":   {"fr": "Colere",    "emoji": "😡"},
    "disgust": {"fr": "Degout",    "emoji": "🤢"},
    "fear":    {"fr": "Peur",      "emoji": "😨"},
    "joy":     {"fr": "Joie",      "emoji": "😊"},
    "neutral": {"fr": "Neutre",    "emoji": "😐"},
    "sadness": {"fr": "Tristesse", "emoji": "😢"},
    "surprise":{"fr": "Surprise",  "emoji": "😲"},
}

class TextInput(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
def analyze(input: TextInput):
    text = input.text[:512]

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0].tolist()
    label_id = probs.index(max(probs))

    try:
        translated = translator(text[:400], max_length=512)[0]["translation_text"]
        emotion_results = emotion_classifier(translated)[0]
        best_emotion = max(emotion_results, key=lambda x: x["score"])
        emotion_key = best_emotion["label"].lower()
        emotion_info = EMOTION_MAP.get(emotion_key, EMOTION_MAP["neutral"])
        emotion_data = {
            "emotion": emotion_key,
            "emotion_fr": emotion_info["fr"],
            "emoji": emotion_info["emoji"],
            "confidence": round(best_emotion["score"], 3)
        }
    except Exception:
        emotion_data = {"emotion": "neutral", "emotion_fr": "Neutre", "emoji": "😐", "confidence": 0.5}

    return {
        "text": text[:100],
        "label": LABELS[label_id],
        "confidence": round(max(probs), 3),
        "scores": {
            "credible": round(probs[0], 3),
            "douteux": round(probs[1], 3),
            "fake_news": round(probs[2], 3)
        },
        "emotion": emotion_data
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)