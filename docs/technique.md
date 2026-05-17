# Documentation technique — Thumalien

## Vue d'ensemble

Thumalien est un pipeline de détection de fake news construit autour de trois blocs principaux : la collecte de données, l'analyse NLP, et la restitution via API et dashboard.

Ce document décrit les choix techniques, l'architecture et les instructions pour reproduire le projet.

---

## Architecture générale

```
[Bluesky API]
      ↓
[Collecteur Python]  →  [Supabase PostgreSQL]
      ↓
[Pipeline NLP]
  - Nettoyage (spaCy)
  - Classification (CamemBERT)
  - Émotions (DistilRoBERTa)
  - Traduction FR→EN (Helsinki-NLP)
      ↓
[API FastAPI]  →  [HuggingFace Spaces]
      ↓
[Dashboard Streamlit]  →  [Streamlit Cloud]
```

---

## Collecte des données

**Fichier :** `src/collector/bluesky_collector.py`

On utilise la librairie `atproto` pour se connecter à l'API officielle Bluesky et récupérer des posts par mots-clés. Les posts sont stockés directement dans Supabase via le client Python officiel.

Mots-clés utilisés pour la collecte :
- complot, fake news, désinformation
- vaccin danger, 5G, élection fraude

Chaque post stocké contient : id, texte, auteur, date, likes, reposts, query source.

---

## Construction du dataset

**Fichiers :** `src/preprocessing/`

Le dataset final combine trois sources :

| Source | Exemples | Langue |
|---|---|---|
| LIAR dataset (filtré) | 3493 | Anglais |
| Données synthétiques Claude | 151 | Français |
| Posts Bluesky réels | 280 | Français |

Les labels sont sur 3 classes :
- `0` — Crédible
- `1` — Douteux
- `2` — Fake News

Le split final :
- Train : 3621 exemples
- Val : 640 exemples
- Test : 280 posts Bluesky réels (jamais vus pendant l'entraînement)

---

## Modèle de détection

**Modèle de base :** `camembert-base` (Facebook, 110M paramètres)

CamemBERT a été choisi pour sa spécialisation sur le français. Il a été pré-entraîné sur 138 Go de texte français, ce qui lui permet de comprendre les nuances du langage bien mieux qu'un modèle multilingue générique.

**Fine-tuning :**
- Entraîné sur Google Colab (GPU T4 gratuit)
- 5 epochs avec early stopping (patience=3)
- Learning rate : 3e-5
- Batch size : 8
- Framework : HuggingFace Transformers + Trainer API

**Performances sur le test set (280 posts Bluesky réels) :**

| Classe | Précision | Rappel | F1 |
|---|---|---|---|
| Crédible | 0.90 | 0.57 | 0.70 |
| Douteux | 0.65 | 0.87 | 0.75 |
| Fake News | 0.33 | 0.23 | 0.27 |
| **Weighted avg** | **0.70** | **0.68** | **0.67** |

**Limite principale :** la classe Fake News est sous-représentée dans les données françaises. C'est un problème connu dans la littérature sur la détection de désinformation en langue autre que l'anglais.

---

## Analyse émotionnelle

On traduit d'abord le texte du français vers l'anglais avec `Helsinki-NLP/opus-mt-fr-en`, puis on applique `j-hartmann/emotion-english-distilroberta-base` qui détecte 7 émotions : colère, dégoût, peur, joie, neutre, tristesse, surprise.

La traduction intermédiaire est nécessaire car peu de modèles d'émotions existent en français avec des performances comparables.

---

## Suivi énergétique (Green IT)

Chaque appel à `/analyze` est encapsulé dans un `EmissionsTracker` CodeCarbon qui mesure la consommation CPU/GPU en temps réel et calcule les émissions CO2 correspondantes.

Les données sont loggées dans `data/energy_log.json` et consultables via `/energy`.

Résultat mesuré : **~0.011g CO2 par analyse**, soit environ 180 fois moins qu'une requête équivalente à GPT-4.

---

## API FastAPI

**Fichier :** `src/api/main.py`

### Endpoints

```
GET  /          → infos générales
GET  /health    → status de l'API
POST /analyze   → analyser un texte
GET  /energy    → statistiques énergétiques
GET  /docs      → documentation Swagger auto-générée
```

### Exemple de requête

```bash
curl -X POST "https://thumalien-thumalien-api.hf.space/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "La 5G implante des puces dans notre cerveau"}'
```

### Exemple de réponse

```json
{
  "text": "La 5G implante des puces dans notre cerveau",
  "label": "Fake News",
  "confidence": 0.889,
  "scores": {
    "credible": 0.056,
    "douteux": 0.055,
    "fake_news": 0.889
  },
  "emotion": {
    "emotion": "anger",
    "emotion_fr": "Colere",
    "emoji": "😡",
    "confidence": 0.83
  },
  "energy": {
    "emissions_kg": 0.0000114,
    "emissions_g": 0.011
  }
}
```

---

## Déploiement

| Service | Plateforme | URL |
|---|---|---|
| API FastAPI | HuggingFace Spaces (Docker) | thumalien-thumalien-api.hf.space |
| Dashboard | Streamlit Cloud | thumalien.streamlit.app |
| Modèle | HuggingFace Hub | huggingface.co/thumalien/thumalien-fake-news |
| Code | GitHub | github.com/developpeur239/thumalien |
| Base de données | Supabase | cloud privé |

---

## Pistes d'amélioration

- Augmenter le dataset français pour améliorer la détection des fake news
- Ajouter SHAP pour expliquer visuellement quels mots ont influencé le verdict
- Mettre en place une collecte continue automatisée (cron job)
- Ajouter un système d'alertes pour les posts viraux avec score faible
- Évaluer sur d'autres réseaux sociaux (Twitter/X, Mastodon)
