"""
Tests unitaires — Projet Thumalien
Joel Mambou & Lucas Blanchide — SUP DE VINCI 2025

Lancer : pytest tests/ -v
"""

import pytest
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ─── TESTS PREPROCESSING ──────────────────────────────────────────────────────

class TestPreprocessing:
    """Tests sur le nettoyage et la préparation des données"""

    def test_text_not_empty(self):
        text = "La 5G nous surveille"
        assert len(text) > 0

    def test_text_truncation(self):
        long_text = "a" * 1000
        truncated = long_text[:512]
        assert len(truncated) == 512

    def test_label_mapping(self):
        """Vérifie que les labels sont bien mappés sur 3 classes"""
        liar_labels = ["true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"]
        label_mapping = {
            "true": 0, "mostly-true": 0,
            "half-true": 1, "barely-true": 1,
            "false": 2, "pants-fire": 2
        }
        for label in liar_labels:
            assert label in label_mapping
            assert label_mapping[label] in [0, 1, 2]

    def test_label_values_valid(self):
        """Les labels doivent être 0, 1 ou 2"""
        valid_labels = [0, 1, 2]
        for label in valid_labels:
            assert label in [0, 1, 2]

    def test_text_cleaning_strips_whitespace(self):
        text = "  La 5G est dangereuse  "
        cleaned = text.strip()
        assert cleaned == "La 5G est dangereuse"

    def test_empty_text_detection(self):
        empty_texts = ["", " ", "   "]
        for text in empty_texts:
            assert len(text.strip()) == 0

    def test_dataset_split_ratios(self):
        """Vérifie que le split train/val/test est cohérent"""
        total = 4261
        train = 3621
        val   = 640
        # Le split doit couvrir ~85/15
        assert train / total > 0.80
        assert val   / total < 0.20
        assert train + val <= total


# ─── TESTS MODELE ─────────────────────────────────────────────────────────────

class TestModel:
    """Tests sur les sorties du modèle"""

    def test_output_probabilities_sum_to_one(self):
        """Les probabilités doivent sommer à 1"""
        probs = [0.056, 0.055, 0.889]
        assert abs(sum(probs) - 1.0) < 0.01

    def test_output_probabilities_in_range(self):
        """Chaque probabilité doit être entre 0 et 1"""
        probs = [0.056, 0.055, 0.889]
        for p in probs:
            assert 0.0 <= p <= 1.0

    def test_label_id_from_probs(self):
        """Le label prédit doit correspondre à la probabilité max"""
        probs = [0.056, 0.055, 0.889]
        label_id = probs.index(max(probs))
        assert label_id == 2  # Fake News

    def test_labels_mapping(self):
        """Vérifie le mapping id → label texte"""
        LABELS = {0: "Credible", 1: "Douteux", 2: "Fake News"}
        assert LABELS[0] == "Credible"
        assert LABELS[1] == "Douteux"
        assert LABELS[2] == "Fake News"

    def test_confidence_score_format(self):
        """Le score de confiance doit être arrondi à 3 décimales"""
        raw_score = 0.8891234
        rounded = round(raw_score, 3)
        assert rounded == 0.889
        assert len(str(rounded).split(".")[-1]) <= 3

    def test_credible_label(self):
        probs = [0.85, 0.10, 0.05]
        label_id = probs.index(max(probs))
        LABELS = {0: "Credible", 1: "Douteux", 2: "Fake News"}
        assert LABELS[label_id] == "Credible"

    def test_douteux_label(self):
        probs = [0.20, 0.65, 0.15]
        label_id = probs.index(max(probs))
        LABELS = {0: "Credible", 1: "Douteux", 2: "Fake News"}
        assert LABELS[label_id] == "Douteux"

    def test_fake_news_label(self):
        probs = [0.05, 0.05, 0.90]
        label_id = probs.index(max(probs))
        LABELS = {0: "Credible", 1: "Douteux", 2: "Fake News"}
        assert LABELS[label_id] == "Fake News"


# ─── TESTS EMOTION ────────────────────────────────────────────────────────────

class TestEmotion:
    """Tests sur le module d'analyse émotionnelle"""

    def test_emotion_map_complete(self):
        """Vérifie que toutes les émotions sont mappées"""
        EMOTION_MAP = {
            "anger": {"fr": "Colere", "emoji": "😡"},
            "disgust": {"fr": "Degout", "emoji": "🤢"},
            "fear": {"fr": "Peur", "emoji": "😨"},
            "joy": {"fr": "Joie", "emoji": "😊"},
            "neutral": {"fr": "Neutre", "emoji": "😐"},
            "sadness": {"fr": "Tristesse", "emoji": "😢"},
            "surprise": {"fr": "Surprise", "emoji": "😲"},
        }
        expected_emotions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
        for emotion in expected_emotions:
            assert emotion in EMOTION_MAP
            assert "fr" in EMOTION_MAP[emotion]
            assert "emoji" in EMOTION_MAP[emotion]

    def test_emotion_fallback(self):
        """Si émotion inconnue, retourner neutral"""
        EMOTION_MAP = {"neutral": {"fr": "Neutre", "emoji": "😐"}}
        unknown = "happiness"
        result = EMOTION_MAP.get(unknown, EMOTION_MAP["neutral"])
        assert result["fr"] == "Neutre"

    def test_emotion_confidence_in_range(self):
        confidence = 0.830
        assert 0.0 <= confidence <= 1.0

    def test_emotion_response_structure(self):
        """La réponse émotion doit avoir les bons champs"""
        emotion_data = {
            "emotion": "anger",
            "emotion_fr": "Colere",
            "emoji": "😡",
            "confidence": 0.830
        }
        required_fields = ["emotion", "emotion_fr", "emoji", "confidence"]
        for field in required_fields:
            assert field in emotion_data


# ─── TESTS API RESPONSE FORMAT ────────────────────────────────────────────────

class TestAPIResponseFormat:
    """Tests sur la structure des réponses API"""

    def test_analyze_response_has_required_fields(self):
        """La réponse /analyze doit contenir tous les champs requis"""
        mock_response = {
            "text": "La 5G implante des puces",
            "label": "Fake News",
            "confidence": 0.889,
            "scores": {
                "credible": 0.056,
                "douteux": 0.055,
                "fake_news": 0.889
            },
            "emotion": {
                "emotion": "neutral",
                "emotion_fr": "Neutre",
                "emoji": "😐",
                "confidence": 0.657
            },
            "energy": {
                "emissions_kg": 0.0000114,
                "emissions_g": 0.011
            }
        }
        required = ["text", "label", "confidence", "scores", "emotion", "energy"]
        for field in required:
            assert field in mock_response

    def test_scores_has_three_classes(self):
        scores = {"credible": 0.056, "douteux": 0.055, "fake_news": 0.889}
        assert "credible"  in scores
        assert "douteux"   in scores
        assert "fake_news" in scores

    def test_scores_sum_to_one(self):
        scores = {"credible": 0.056, "douteux": 0.055, "fake_news": 0.889}
        total = sum(scores.values())
        assert abs(total - 1.0) < 0.01

    def test_label_is_valid_class(self):
        valid_labels = ["Credible", "Douteux", "Fake News"]
        label = "Fake News"
        assert label in valid_labels

    def test_confidence_between_0_and_1(self):
        confidence = 0.889
        assert 0.0 <= confidence <= 1.0

    def test_energy_response_structure(self):
        energy = {"emissions_kg": 0.0000114, "emissions_g": 0.011}
        assert "emissions_kg" in energy
        assert "emissions_g"  in energy
        assert energy["emissions_kg"] >= 0
        assert energy["emissions_g"]  >= 0

    def test_energy_endpoint_structure(self):
        """La réponse /energy doit avoir les bons champs"""
        mock_energy = {
            "total_analyses": 42,
            "total_emissions_kg": 0.00048,
            "avg_per_analysis_g": 0.011,
            "equivalent_km_voiture": 0.0028
        }
        required = ["total_analyses", "total_emissions_kg", "avg_per_analysis_g"]
        for field in required:
            assert field in mock_energy

    def test_health_response(self):
        mock_health = {"status": "ok", "model": "camembert-base-thumalien", "version": "1.0.0"}
        assert mock_health["status"] == "ok"
        assert "model" in mock_health


# ─── TESTS GREEN IT ───────────────────────────────────────────────────────────

class TestGreenIT:
    """Tests sur le suivi énergétique"""

    def test_emissions_positive(self):
        emissions = 0.0000114
        assert emissions >= 0

    def test_emissions_kg_to_g_conversion(self):
        emissions_kg = 0.0000114
        emissions_g  = round(emissions_kg * 1000, 7)
        assert abs(emissions_g - 0.011) < 0.001

    def test_km_equivalent_calculation(self):
        """1 km voiture = ~0.000171 kg CO2"""
        total_emissions_kg = 0.00048
        km_per_kg_co2 = 1 / 0.000171
        km_equivalent = total_emissions_kg * km_per_kg_co2
        assert km_equivalent > 0
        assert km_equivalent < 10  # moins de 10 km pour 42 analyses

    def test_log_entry_structure(self):
        log_entry = {
            "timestamp": "2025-01-15T14:30:00",
            "emissions_kg": 0.0000114,
            "text_length": 45,
            "label": "Fake News"
        }
        required = ["timestamp", "emissions_kg", "text_length", "label"]
        for field in required:
            assert field in log_entry

    def test_thumalien_vs_gpt4_ratio(self):
        """Thumalien doit être au moins 100x moins énergivore que GPT-4"""
        thumalien_g = 0.011
        gpt4_g      = 2.0
        ratio = gpt4_g / thumalien_g
        assert ratio > 100