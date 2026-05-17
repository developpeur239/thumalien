# Thumalien

Détecteur de fake news sur Bluesky, développé dans le cadre du Mastère Big Data & IA à SUP DE VINCI (2025).

L'idée de base : automatiser ce qu'un fact-checker humain ferait à la main — lire un post, évaluer sa crédibilité, identifier l'émotion qu'il cherche à provoquer.

---

## Ce que ça fait

Tu colles un texte ou un post Bluesky, le système te dit :
- si c'est crédible, douteux ou une fake news
- l'émotion dominante du contenu (colère, peur, joie...)
- le niveau de confiance du modèle
- la consommation CO2 de l'analyse

---

## Liens

- Dashboard : https://thumalien.streamlit.app
- API : https://thumalien-thumalien-api.hf.space/docs
- Modèle : https://huggingface.co/thumalien/thumalien-fake-news

---

## Stack

- Collecte des données : API officielle Bluesky (atproto)
- Stockage : Supabase (PostgreSQL)
- Modèle principal : CamemBERT fine-tuné sur LIAR dataset + posts Bluesky réels
- Analyse émotionnelle : DistilRoBERTa via HuggingFace
- Traduction FR→EN : Helsinki-NLP
- API : FastAPI déployée sur HuggingFace Spaces
- Dashboard : Streamlit déployé sur Streamlit Cloud
- Green IT : CodeCarbon (suivi CO2 par analyse)

---

## Performances

Le modèle a été entraîné sur 3621 exemples et évalué sur 280 vrais posts Bluesky.

| Classe | F1 |
|---|---|
| Crédible | 0.75 |
| Douteux | 0.79 |
| Fake News | 0.27 |
| **Weighted F1** | **0.67** |

La détection des fake news reste le point le plus difficile — les données en français sur ce sujet sont rares, c'est une limite connue du projet.

---

## Lancer en local

```bash
git clone https://github.com/developpeur239/thumalien
cd thumalien
pip install -r requirements.txt

# Configurer les variables d'environnement
cp .env.example .env

# API
python src/api/main.py

# Dashboard (dans un autre terminal)
streamlit run dashboard/app.py
```

Variables à remplir dans `.env` :
```
BLUESKY_HANDLE=...
BLUESKY_PASSWORD=...
SUPABASE_URL=...
SUPABASE_KEY=...
HF_TOKEN=...
ANTHROPIC_API_KEY=...
```

---

## Structure

```
src/
  collector/      → collecte les posts Bluesky
  preprocessing/  → nettoyage, labellisation, dataset
  emotion/        → analyse émotionnelle
  api/            → FastAPI + suivi énergétique

dashboard/        → interface Streamlit
data/             → données brutes, traitées, labellisées
models/           → modèle CamemBERT en local
```

---

## Green IT

Chaque analyse consomme environ 0.011g de CO2.
C'est ~180 fois moins qu'une requête GPT-4.
Les émissions sont loggées et consultables via `/energy`.

---

Mastère Big Data & IA — SUP DE VINCI — 2025