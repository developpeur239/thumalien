from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from codecarbon import EmissionsTracker
from datetime import datetime
import torch
import uvicorn
import json
import os

app = FastAPI(
    title="Thumalien API",
    description="Detection de Fake News sur Bluesky avec analyse emotionnelle et suivi energetique",
    version="1.0.0"
)

# ─── CHARGEMENT DES MODELES ────────────────────────────────────────────────────
print("Chargement des modeles...")

tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForSequenceClassification.from_pretrained("models/camembert_thumalien")
model.eval()

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

print("Tous les modeles charges")

# ─── CONSTANTES ────────────────────────────────────────────────────────────────
LABELS = {0: "Credible", 1: "Douteux", 2: "Fake News"}

EMOTION_MAP = {
    "anger":   {"fr": "Colere",    "emoji": "😡", "color": "#FF4545"},
    "disgust": {"fr": "Degout",    "emoji": "🤢", "color": "#8B5CF6"},
    "fear":    {"fr": "Peur",      "emoji": "😨", "color": "#F59E0B"},
    "joy":     {"fr": "Joie",      "emoji": "😊", "color": "#22C55E"},
    "neutral": {"fr": "Neutre",    "emoji": "😐", "color": "#6B7280"},
    "sadness": {"fr": "Tristesse", "emoji": "😢", "color": "#3B82F6"},
    "surprise":{"fr": "Surprise",  "emoji": "😲", "color": "#EC4899"},
}

ENERGY_LOG = "data/energy_log.json"
os.makedirs("data", exist_ok=True)

# ─── UTILS ─────────────────────────────────────────────────────────────────────
def log_energy(emissions: float, text_length: int, label: str):
    """Sauvegarde les emissions CO2 de chaque analyse"""
    log = []
    if os.path.exists(ENERGY_LOG):
        try:
            with open(ENERGY_LOG, "r") as f:
                log = json.load(f)
        except Exception:
            log = []

    log.append({
        "timestamp": datetime.now().isoformat(),
        "emissions_kg": round(emissions or 0, 10),
        "text_length": text_length,
        "label": label
    })

    # Garder les 1000 derniers logs
    with open(ENERGY_LOG, "w") as f:
        json.dump(log[-1000:], f, indent=2)


# ─── SCHEMAS ───────────────────────────────────────────────────────────────────
class TextInput(BaseModel):
    text: str


# ─── ENDPOINTS ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "Thumalien API",
        "version": "1.0.0",
        "description": "Detection de Fake News sur Bluesky",
        "endpoints": ["/health", "/analyze", "/energy", "/docs"]
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "camembert-base-thumalien",
        "version": "1.0.0"
    }


@app.post("/analyze")
def analyze(input: TextInput):
    text = input.text[:512]

    # ── Suivi energetique ──
    tracker = EmissionsTracker(
        save_to_file=False,
        log_level="error",
        measure_power_secs=1
    )
    tracker.start()

    # ── Detection fake news ──
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0].tolist()
    label_id = probs.index(max(probs))
    label = LABELS[label_id]

    # ── Analyse emotionnelle ──
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
            "color": emotion_info["color"],
            "confidence": round(best_emotion["score"], 3)
        }
    except Exception:
        emotion_data = {
            "emotion": "neutral",
            "emotion_fr": "Neutre",
            "emoji": "😐",
            "color": "#6B7280",
            "confidence": 0.5
        }

    # ── Stopper le tracker ──
    emissions = tracker.stop()
    log_energy(emissions or 0, len(text), label)

    return {
        "text": text[:100],
        "label": label,
        "confidence": round(max(probs), 3),
        "scores": {
            "credible":  round(probs[0], 3),
            "douteux":   round(probs[1], 3),
            "fake_news": round(probs[2], 3)
        },
        "emotion": emotion_data,
        "energy": {
            "emissions_kg": round(emissions or 0, 10),
            "emissions_g": round((emissions or 0) * 1000, 7)
        }
    }


@app.get("/energy")
def energy_stats():
    """Statistiques de consommation energetique du projet"""
    if not os.path.exists(ENERGY_LOG):
        return {
            "total_analyses": 0,
            "total_emissions_kg": 0,
            "avg_per_analysis_g": 0,
            "equivalent_km_voiture": 0
        }

    with open(ENERGY_LOG, "r") as f:
        log = json.load(f)

    if not log:
        return {"total_analyses": 0, "total_emissions_kg": 0}

    total_emissions = sum(e["emissions_kg"] for e in log)
    avg_emissions   = total_emissions / len(log)

    # 1 km en voiture = ~0.000171 kg CO2
    km_equivalent = total_emissions / 0.000171

    labels_count = {}
    for e in log:
        labels_count[e.get("label", "unknown")] = labels_count.get(e.get("label", "unknown"), 0) + 1

    return {
        "total_analyses":      len(log),
        "total_emissions_kg":  round(total_emissions, 8),
        "avg_per_analysis_g":  round(avg_emissions * 1000, 7),
        "equivalent_km_voiture": round(km_equivalent, 4),
        "labels_distribution": labels_count,
        "last_analysis":       log[-1]["timestamp"] if log else None
    }


# ─── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)