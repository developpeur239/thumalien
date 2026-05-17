# 🔍 Thumalien — Détecteur de Fake News sur Bluesky

> Projet d'étude — Mastère Big Data & IA — SUP DE VINCI 2025

![Python](https://img.shields.io/badge/Python-3.12-blue)
![CamemBERT](https://img.shields.io/badge/Model-CamemBERT-orange)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-yellow)

## 🌐 Liens

| Service | URL |
|---|---|
| 🎯 Dashboard | https://thumalien.streamlit.app |
| ⚡ API | https://thumalien-thumalien-api.hf.space |
| 🤖 Modèle | https://huggingface.co/thumalien/thumalien-fake-news |

## 🎯 Problème résolu

Les réseaux sociaux comme Bluesky sont envahis de fake news.
La modération humaine ne peut pas suivre le volume.
**Thumalien** analyse automatiquement n'importe quel post et retourne :
- Un **score de crédibilité** (Crédible / Douteux / Fake News)
- L'**émotion dominante** du contenu (colère, peur, joie...)
- Un **niveau de confiance** en pourcentage

## 🏗️ Architecture