from transformers import pipeline
import os

print("Chargement modeles...")

# Traducteur FR -> EN
translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-fr-en"
)

# Detecteur d'emotions
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

EMOTION_MAP = {
    "anger":   {"fr": "Colere",    "emoji": "😡", "color": "#FF4545"},
    "disgust": {"fr": "Degout",    "emoji": "🤢", "color": "#8B5CF6"},
    "fear":    {"fr": "Peur",      "emoji": "😨", "color": "#F59E0B"},
    "joy":     {"fr": "Joie",      "emoji": "😊", "color": "#22C55E"},
    "neutral": {"fr": "Neutre",    "emoji": "😐", "color": "#6B7280"},
    "sadness": {"fr": "Tristesse", "emoji": "😢", "color": "#3B82F6"},
    "surprise":{"fr": "Surprise",  "emoji": "😲", "color": "#EC4899"},
}

def analyze_emotion(text: str) -> dict:
    # Traduire en anglais d'abord
    translated = translator(text[:400], max_length=512)[0]["translation_text"]
    print(f"  Traduction : {translated[:80]}")

    results = emotion_classifier(translated)[0]
    best = max(results, key=lambda x: x["score"])
    emotion_key = best["label"].lower()
    emotion_info = EMOTION_MAP.get(emotion_key, EMOTION_MAP["neutral"])

    return {
        "emotion": emotion_key,
        "emotion_fr": emotion_info["fr"],
        "emoji": emotion_info["emoji"],
        "color": emotion_info["color"],
        "confidence": round(best["score"], 3),
        "all_scores": {
            EMOTION_MAP[r["label"].lower()]["fr"]: round(r["score"], 3)
            for r in results if r["label"].lower() in EMOTION_MAP
        }
    }

if __name__ == "__main__":
    tests = [
        "La 5G nous surveille et le gouvernement cache tout !",
        "Une nouvelle etude publiee dans Nature confirme les effets du rechauffement.",
        "SCANDALE : ils empoisonnent nos enfants et personne ne dit rien !!!"
    ]
    for t in tests:
        print(f"\nTexte : {t[:60]}")
        result = analyze_emotion(t)
        print(f"Emotion : {result['emoji']} {result['emotion_fr']} ({result['confidence']*100:.1f}%)")