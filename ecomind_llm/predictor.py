"""
EcoMind Predictor — Inference Engine
======================================
Loads the trained EcoMindNet and makes predictions on new product descriptions.

Usage:
    from ecomind_llm.predictor import EcoMindPredictor
    predictor = EcoMindPredictor()
    result = predictor.predict({
        "name": "Organic Cotton Tee",
        "brand": "Patagonia",
        "category": "Clothing",
        "materials": "organic_cotton",
        "sustainability_cert": "GOTS",
        "description": "Classic organic cotton t-shirt, GOTS certified."
    })
    print(result)
    # {
    #   "eco_score": 9.2,
    #   "ethics_score": 8.8,
    #   "carbon_level": 0,
    #   "carbon_label": "ultra_low",
    #   "tags": ["organic", "natural"],
    #   "confidence": 91.3,
    #   "overall_score": 9.0
    # }
"""

import os
import numpy as np

from ecomind_llm.neural_net   import EcoMindNet
from ecomind_llm.preprocessor import (
    TFIDFVectorizer, build_input_text,
    denormalise_score, class_to_carbon_label,
    decode_tags, TAG_NAMES, CARBON_LABELS
)

MODEL_PATH      = os.path.join(os.path.dirname(__file__), "saved_model", "ecomind_net.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "saved_model", "tfidf_vectorizer.pkl")


class EcoMindPredictor:
    """
    Singleton-style predictor.
    Loads model once, predicts many times efficiently.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model      = None
            cls._instance.vectorizer = None
            cls._instance.ready      = False
        return cls._instance

    def load(self):
        """Load trained weights and vectorizer from disk."""
        if self.ready:
            return self

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}.\n"
                "Please run:  python ecomind_llm/train.py  first."
            )
        if not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError(
                f"Vectorizer not found at {VECTORIZER_PATH}.\n"
                "Please run:  python ecomind_llm/train.py  first."
            )

        # Load vectorizer first to get vocab size
        self.vectorizer = TFIDFVectorizer()
        self.vectorizer.load(VECTORIZER_PATH)

        # Build model with correct input dim and load weights
        self.model = EcoMindNet(
            input_dim=self.vectorizer.vocab_size,
            hidden1=256, hidden2=128, hidden3=64
        )
        self.model.load(MODEL_PATH)
        self.model._set_training_mode(False)  # inference mode
        self.ready = True
        print("[EcoMindPredictor] Model ready for inference.")
        return self

    def predict(self, product_dict):
        """
        Predict ethical scores for a single product.

        Args:
            product_dict: dict with keys: name, brand, category,
                          materials, sustainability_cert, description

        Returns:
            dict with eco_score, ethics_score, carbon_level,
                  carbon_label, tags, tag_names, confidence, overall_score
        """
        if not self.ready:
            self.load()

        # Build and vectorise text
        text = build_input_text(product_dict)
        X    = self.vectorizer.transform([text])   # (1, vocab_size)

        # Forward pass
        self.model._set_training_mode(False)
        eco_p, eth_p, car_p, tag_p = self.model.forward(X)

        # Decode outputs
        eco_score    = round(float(np.clip(denormalise_score(eco_p[0, 0]), 0, 10)), 1)
        ethics_score = round(float(np.clip(denormalise_score(eth_p[0, 0]), 0, 10)), 1)
        carbon_level = int(np.argmax(car_p[0]))
        carbon_label = class_to_carbon_label(carbon_level)
        carbon_conf  = round(float(car_p[0, carbon_level]) * 100, 1)
        tag_probs    = tag_p[0]
        tags         = decode_tags(tag_probs, threshold=0.4)

        # Overall ethical score (weighted average)
        overall = round((eco_score * 0.45 + ethics_score * 0.45 +
                         (3 - carbon_level) * 0.5 +           # low carbon → bonus
                         len(tags) * 0.1), 1)
        overall = round(min(overall, 10.0), 1)

        # Confidence: average of eco/ethics normalised + carbon softmax prob
        confidence = round((
            min(eco_score / 10, 1.0) * 0.35 +
            min(ethics_score / 10, 1.0) * 0.35 +
            carbon_conf / 100 * 0.30
        ) * 100, 1)

        return {
            "eco_score":     eco_score,
            "ethics_score":  ethics_score,
            "carbon_level":  carbon_level,
            "carbon_label":  carbon_label,
            "carbon_conf":   carbon_conf,
            "tag_probs":     {TAG_NAMES[i]: round(float(tag_probs[i]), 3)
                              for i in range(len(TAG_NAMES))},
            "tags":          tags,
            "confidence":    confidence,
            "overall_score": overall,
        }

    def predict_batch(self, product_list):
        """Predict for a list of product dicts efficiently."""
        if not self.ready:
            self.load()

        texts = [build_input_text(p) for p in product_list]
        X     = self.vectorizer.transform(texts)

        self.model._set_training_mode(False)
        eco_p, eth_p, car_p, tag_p = self.model.forward(X)

        results = []
        for i in range(len(product_list)):
            eco_score    = round(float(np.clip(denormalise_score(eco_p[i, 0]), 0, 10)), 1)
            ethics_score = round(float(np.clip(denormalise_score(eth_p[i, 0]), 0, 10)), 1)
            carbon_level = int(np.argmax(car_p[i]))
            carbon_label = class_to_carbon_label(carbon_level)
            tag_probs    = tag_p[i]
            tags         = decode_tags(tag_probs, threshold=0.4)
            overall      = round(min(eco_score*0.45 + ethics_score*0.45 +
                                     (3-carbon_level)*0.5 + len(tags)*0.1, 10.0), 1)
            confidence   = round((min(eco_score/10,1)*0.35 +
                                  min(ethics_score/10,1)*0.35 +
                                  float(car_p[i,carbon_level])*0.30)*100, 1)
            results.append({
                "eco_score": eco_score, "ethics_score": ethics_score,
                "carbon_level": carbon_level, "carbon_label": carbon_label,
                "tags": tags, "confidence": confidence, "overall_score": overall,
            })
        return results

    def predict_from_text(self, text):
        """
        Predict from raw natural language text.
        e.g. "organic cotton shirt from fair trade factory, GOTS certified"
        """
        return self.predict({
            "name": "", "brand": "", "category": "",
            "materials": "", "sustainability_cert": "",
            "description": text
        })


# ── Singleton instance ────────────────────────────────────────
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = EcoMindPredictor()
        _predictor.load()
    return _predictor
