"""
EcoMind RNN Intent Predictor
Loads trained model once, classifies any user message instantly.
"""

import numpy as np
import os, json

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved_model')

INTENT_CLASSES = [
    "product_search",
    "greeting",
    "help",
    "ethical_question",
    "political",
    "vulgar_abusive",
    "off_topic",
    "out_of_scope",
]

# Intent → response type
INTENT_RESPONSES = {
    "product_search":    "search",
    "greeting":          "greeting",
    "help":              "help",
    "ethical_question":  "ethical",
    "political":         "political",
    "vulgar_abusive":    "vulgar",
    "off_topic":         "off_topic",
    "out_of_scope":      "out_of_scope",
}

_predictor_instance = None


def get_intent_predictor():
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = IntentPredictor()
    return _predictor_instance


class IntentPredictor:

    def __init__(self):
        self.ready = False
        self._load()

    def _load(self):
        weights_path  = os.path.join(MODEL_DIR, 'rnn_weights.json')
        tokenizer_path = os.path.join(MODEL_DIR, 'tokenizer.json')
        config_path   = os.path.join(MODEL_DIR, 'config.json')

        if not all(os.path.exists(p) for p in [weights_path, tokenizer_path, config_path]):
            print("[RNN] Saved model not found. Run: python ecomind_rnn/train_rnn.py")
            self.ready = False
            return

        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from ecomind_rnn.rnn_model import IntentClassifier, Tokenizer

        with open(config_path) as f:
            cfg = json.load(f)

        self.tok = Tokenizer.load(tokenizer_path)
        self.model = IntentClassifier(
            vocab_size  = cfg['vocab_size'],
            embed_dim   = cfg.get('embed_dim',  32),
            hidden_dim  = cfg.get('hidden_dim', 64),
            num_classes = cfg['num_classes'],
            max_len     = cfg.get('max_len',    20),
        )
        self.model.load(weights_path)
        self.intents = cfg.get('intent_classes', INTENT_CLASSES)
        self.ready   = True
        print(f"[RNN] ✅ Intent classifier loaded. Val acc: {cfg.get('best_val_acc','?')}")

    def predict(self, text):
        """
        Returns dict:
          intent      - intent class name
          confidence  - float 0-1
          all_probs   - dict of all class probabilities
          response_type - what kind of response to give
        """
        if not self.ready:
            # Fallback — treat as product search
            return {
                'intent': 'product_search',
                'confidence': 0.5,
                'all_probs': {},
                'response_type': 'search',
                'fallback': True,
            }

        token_ids = self.tok.encode(text).reshape(1, -1)
        probs     = self.model.forward(token_ids, training=False)[0]
        top_idx   = int(probs.argmax())
        confidence = float(probs[top_idx])

        intent = self.intents[top_idx] if top_idx < len(self.intents) else 'out_of_scope'

        # Low confidence → treat as out_of_scope
        if confidence < 0.35:
            intent = 'out_of_scope'

        return {
            'intent':        intent,
            'confidence':    round(confidence, 3),
            'all_probs':     {self.intents[i]: round(float(p), 3)
                              for i, p in enumerate(probs)},
            'response_type': INTENT_RESPONSES.get(intent, 'search'),
        }
