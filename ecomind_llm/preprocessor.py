"""
EcoMind Text Preprocessor
==========================
Converts raw product text → numerical features for the neural network.

Pipeline:
  1. Clean & tokenise text
  2. Build vocabulary from training corpus
  3. TF-IDF vectorisation (sparse → dense)
  4. Label encoding for all targets
"""

import numpy as np
import re
import pickle
from collections import Counter


# ════════════════════════════════════════════════════════════
# 1.  TEXT CLEANING
# ════════════════════════════════════════════════════════════
STOP_WORDS = {
    'a','an','the','and','or','but','in','on','at','to','for',
    'of','with','is','are','was','were','be','been','have','has',
    'it','its','this','that','from','by','as','not','no','so',
    'they','their','we','our','i','my','you','your','he','she',
    'made','make','making','use','used','using','can','will',
    'also','all','each','any','per','into','than','more','most',
}

def clean_text(text):
    """Lowercase, remove punctuation, remove stopwords."""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return ' '.join(tokens)


def build_input_text(row):
    """
    Combine all product fields into one rich text string for the model.
    The more context, the better the predictions.
    """
    parts = [
        str(row.get('name', '')),
        str(row.get('brand', '')),
        str(row.get('category', '')),
        str(row.get('materials', '')).replace(',', ' '),
        str(row.get('sustainability_cert', '')),
        str(row.get('description', '')),
    ]
    return clean_text(' '.join(parts))


# ════════════════════════════════════════════════════════════
# 2.  TF-IDF VECTORISER  (built from scratch)
# ════════════════════════════════════════════════════════════
class TFIDFVectorizer:
    """
    Term Frequency – Inverse Document Frequency vectoriser.
    Converts a list of text documents into a dense feature matrix.
    """

    def __init__(self, max_features=512, ngram_max=2):
        self.max_features = max_features
        self.ngram_max    = ngram_max
        self.vocab        = {}        # token → index
        self.idf          = None      # IDF weights
        self.vocab_size   = 0

    def _get_ngrams(self, tokens):
        """Unigrams + bigrams."""
        ngrams = list(tokens)
        if self.ngram_max >= 2:
            ngrams += [tokens[i] + '_' + tokens[i+1]
                       for i in range(len(tokens)-1)]
        return ngrams

    def fit(self, texts):
        """Build vocabulary and IDF from training texts."""
        tokenised = []
        df_counts = Counter()

        for text in texts:
            tokens  = text.split()
            ngrams  = list(set(self._get_ngrams(tokens)))  # unique per doc
            tokenised.append(tokens)
            for ng in ngrams:
                df_counts[ng] += 1

        # Keep top max_features by document frequency
        top_tokens = [tok for tok, _ in df_counts.most_common(self.max_features)]
        self.vocab  = {tok: i for i, tok in enumerate(top_tokens)}
        self.vocab_size = len(self.vocab)

        # Compute IDF
        N = len(texts)
        self.idf = np.zeros(self.vocab_size)
        for tok, idx in self.vocab.items():
            df = df_counts.get(tok, 0)
            self.idf[idx] = np.log((N + 1) / (df + 1)) + 1  # smooth IDF

        print(f"[TFIDFVectorizer] Vocab size: {self.vocab_size}, N-gram max: {self.ngram_max}")
        return self

    def transform(self, texts):
        """Convert texts to TF-IDF matrix (N, vocab_size)."""
        X = np.zeros((len(texts), self.vocab_size), dtype=np.float32)

        for i, text in enumerate(texts):
            tokens = text.split()
            ngrams = self._get_ngrams(tokens)
            tf_counts = Counter(ngrams)
            total = max(len(ngrams), 1)

            for tok, count in tf_counts.items():
                if tok in self.vocab:
                    idx = self.vocab[tok]
                    tf  = count / total
                    X[i, idx] = tf * self.idf[idx]

        # L2 normalise each row
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return X / norms

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'vocab': self.vocab, 'idf': self.idf,
                         'max_features': self.max_features,
                         'ngram_max': self.ngram_max}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        self.vocab        = d['vocab']
        self.idf          = d['idf']
        self.max_features = d['max_features']
        self.ngram_max    = d['ngram_max']
        self.vocab_size   = len(self.vocab)
        return self


# ════════════════════════════════════════════════════════════
# 3.  LABEL ENCODERS FOR TARGETS
# ════════════════════════════════════════════════════════════

CARBON_BINS = [0, 0.8, 2.0, 4.0, 999]
CARBON_LABELS = ['ultra_low', 'low', 'moderate', 'high']

def carbon_to_class(carbon_fp):
    """Convert carbon footprint float → class index 0-3."""
    for i in range(len(CARBON_BINS)-1):
        if float(carbon_fp) <= CARBON_BINS[i+1]:
            return i
    return len(CARBON_LABELS) - 1

def class_to_carbon_label(cls):
    return CARBON_LABELS[int(cls)]


# Certification → tag multi-label encoding
TAG_KEYWORDS = {
    0: ['organic', 'usda', 'cosmos', 'gots', 'certified organic'],
    1: ['fair trade', 'fairtrade'],
    2: ['vegan', 'peta', 'plant based'],
    3: ['recycled', 'upcycled', 'recycling'],
    4: ['natural', 'bamboo', 'hemp', 'cork', 'linen', 'botanical', 'herb'],
}
TAG_NAMES = ['organic', 'fair_trade', 'vegan', 'recycled', 'natural']

def encode_tags(cert, materials, description):
    """Encode product into 5-dim binary tag vector."""
    text = (str(cert) + ' ' + str(materials) + ' ' + str(description)).lower()
    vec = np.zeros(5, dtype=np.float32)
    for i, keywords in TAG_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            vec[i] = 1.0
    return vec

def decode_tags(tag_probs, threshold=0.4):
    """Convert tag probability vector → list of tag names."""
    return [TAG_NAMES[i] for i, p in enumerate(tag_probs) if p >= threshold]


# ════════════════════════════════════════════════════════════
# 4.  SCORE NORMALISER (0-10 → 0-1 for training, back for display)
# ════════════════════════════════════════════════════════════
def normalise_score(score):
    return float(score) / 10.0

def denormalise_score(norm_score):
    return round(float(norm_score) * 10.0, 2)
