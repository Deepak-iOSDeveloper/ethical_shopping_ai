"""
EcoMind RNN Intent Classifier
Built from scratch using NumPy only — no PyTorch, no TensorFlow.

Architecture:
  Input tokens (max_len=20)
      ↓
  Embedding Layer (vocab_size × embed_dim=32)
      ↓
  GRU Layer (hidden_dim=64) — understands word sequences
      ↓
  Attention Layer (weights each time step)
      ↓
  Dense (64 → 32) + ReLU
      ↓
  Output (32 → 8 classes) + Softmax
"""

import numpy as np
import json
import os


class Tokenizer:
    """Simple word tokenizer with vocabulary building."""

    def __init__(self, max_vocab=3000, max_len=20):
        self.max_vocab = max_vocab
        self.max_len   = max_len
        self.word2idx  = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word  = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2

    def _tokenize(self, text):
        """Lowercase, remove punctuation, split to words."""
        import re
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()

    def build_vocab(self, texts):
        """Build vocabulary from list of texts."""
        from collections import Counter
        all_words = []
        for text in texts:
            all_words.extend(self._tokenize(text))
        freq = Counter(all_words)
        # Keep most common words
        common = [w for w, _ in freq.most_common(self.max_vocab - 2)]
        for word in common:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx]  = word
        self.vocab_size = len(self.word2idx)

    def encode(self, text):
        """Convert text to padded integer sequence."""
        tokens = self._tokenize(text)[:self.max_len]
        ids    = [self.word2idx.get(t, 1) for t in tokens]  # 1 = UNK
        # Pad or truncate to max_len
        ids = ids + [0] * (self.max_len - len(ids))
        return np.array(ids, dtype=np.int32)

    def encode_batch(self, texts):
        return np.array([self.encode(t) for t in texts])

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'word2idx': self.word2idx,
                       'max_len': self.max_len,
                       'vocab_size': self.vocab_size}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        tok = cls(max_len=data['max_len'])
        tok.word2idx  = data['word2idx']
        tok.idx2word  = {int(v): k for k, v in data['word2idx'].items()}
        tok.vocab_size = data['vocab_size']
        return tok


class EmbeddingLayer:
    """Learnable word embeddings."""

    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim
        # He init scaled for embeddings
        self.E = np.random.randn(vocab_size, embed_dim) * 0.1
        self.dE = np.zeros_like(self.E)

    def forward(self, token_ids):
        """token_ids: (batch, seq_len) → output: (batch, seq_len, embed_dim)"""
        self._token_ids = token_ids
        return self.E[token_ids]

    def backward(self, dOut):
        """dOut: (batch, seq_len, embed_dim)"""
        self.dE = np.zeros_like(self.E)
        np.add.at(self.dE, self._token_ids, dOut)

    def params(self):
        uid = str(id(self))[-4:]
        return [(f'E_{uid}', self.E, self.dE)]


class GRULayer:
    """
    Gated Recurrent Unit — processes sequences, remembers context.
    
    Unlike simple RNN, GRU has:
    - Update gate z: decides how much of old hidden state to keep
    - Reset gate r: decides how much of old state influences new candidate
    This helps with longer sequences and avoids vanishing gradients.
    """

    def __init__(self, input_dim, hidden_dim):
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        h = hidden_dim
        d = input_dim

        # Xavier init for gates
        scale_z = np.sqrt(2.0 / (d + h))
        scale_h = np.sqrt(2.0 / (d + h))

        # Update gate z
        self.Wz = np.random.randn(d, h) * scale_z
        self.Uz = np.random.randn(h, h) * scale_z
        self.bz = np.zeros(h)

        # Reset gate r
        self.Wr = np.random.randn(d, h) * scale_z
        self.Ur = np.random.randn(h, h) * scale_z
        self.br = np.zeros(h)

        # Candidate hidden state
        self.Wh = np.random.randn(d, h) * scale_h
        self.Uh = np.random.randn(h, h) * scale_h
        self.bh = np.zeros(h)

        # Gradients
        self.dWz = np.zeros_like(self.Wz)
        self.dUz = np.zeros_like(self.Uz)
        self.dbz = np.zeros(h)
        self.dWr = np.zeros_like(self.Wr)
        self.dUr = np.zeros_like(self.Ur)
        self.dbr = np.zeros(h)
        self.dWh = np.zeros_like(self.Wh)
        self.dUh = np.zeros_like(self.Uh)
        self.dbh = np.zeros(h)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

    def _tanh(self, x):
        return np.tanh(np.clip(x, -15, 15))

    def forward(self, X):
        """
        X: (batch, seq_len, input_dim)
        Returns all_hidden: (batch, seq_len, hidden_dim)
        """
        batch, seq_len, _ = X.shape
        h = np.zeros((batch, self.hidden_dim))

        self._X  = X
        self._hs = np.zeros((batch, seq_len + 1, self.hidden_dim))
        self._zs = np.zeros((batch, seq_len, self.hidden_dim))
        self._rs = np.zeros((batch, seq_len, self.hidden_dim))
        self._hs_tilde = np.zeros((batch, seq_len, self.hidden_dim))
        self._hs[: , 0, :] = h

        for t in range(seq_len):
            x_t = X[:, t, :]
            z = self._sigmoid(x_t @ self.Wz + h @ self.Uz + self.bz)
            r = self._sigmoid(x_t @ self.Wr + h @ self.Ur + self.br)
            h_tilde = self._tanh(x_t @ self.Wh + (r * h) @ self.Uh + self.bh)
            h = (1 - z) * h + z * h_tilde
            self._hs[:, t+1, :] = h
            self._zs[:, t, :]   = z
            self._rs[:, t, :]   = r
            self._hs_tilde[:, t, :] = h_tilde

        return self._hs[:, 1:, :]  # (batch, seq_len, hidden_dim)

    def backward(self, dAll):
        """dAll: (batch, seq_len, hidden_dim)"""
        batch, seq_len, _ = dAll.shape
        dh_next = np.zeros((batch, self.hidden_dim))

        self.dWz = np.zeros_like(self.Wz)
        self.dUz = np.zeros_like(self.Uz)
        self.dbz = np.zeros(self.hidden_dim)
        self.dWr = np.zeros_like(self.Wr)
        self.dUr = np.zeros_like(self.Ur)
        self.dbr = np.zeros(self.hidden_dim)
        self.dWh = np.zeros_like(self.Wh)
        self.dUh = np.zeros_like(self.Uh)
        self.dbh = np.zeros(self.hidden_dim)
        dX = np.zeros_like(self._X)

        for t in reversed(range(seq_len)):
            dh = dAll[:, t, :] + dh_next
            z  = self._zs[:, t, :]
            r  = self._rs[:, t, :]
            h_tilde = self._hs_tilde[:, t, :]
            h_prev  = self._hs[:, t, :]
            x_t     = self._X[:, t, :]

            # d h_tilde
            dh_tilde = dh * z
            dtanh    = dh_tilde * (1 - h_tilde ** 2)

            self.dWh += x_t.T @ dtanh
            self.dUh += (r * h_prev).T @ dtanh
            self.dbh += dtanh.sum(axis=0)
            dX[:, t, :] += dtanh @ self.Wh.T
            dr_h   = dtanh @ self.Uh.T
            dr     = dr_h * h_prev
            dh_prev_from_r = dr_h * r

            # d z gate
            dz   = dh * (h_tilde - h_prev)
            dsig_z = dz * z * (1 - z)
            self.dWz += x_t.T @ dsig_z
            self.dUz += h_prev.T @ dsig_z
            self.dbz += dsig_z.sum(axis=0)
            dX[:, t, :] += dsig_z @ self.Wz.T
            dh_prev_from_z = dsig_z @ self.Uz.T

            # d r gate
            dsig_r = dr * r * (1 - r)
            self.dWr += x_t.T @ dsig_r
            self.dUr += h_prev.T @ dsig_r
            self.dbr += dsig_r.sum(axis=0)
            dX[:, t, :] += dsig_r @ self.Wr.T
            dh_prev_from_r2 = dsig_r @ self.Ur.T

            dh_next = (dh * (1 - z) + dh_prev_from_z +
                       dh_prev_from_r + dh_prev_from_r2)

        return dX

    def params(self):
        return [
            ('Wz', self.Wz, self.dWz), ('Uz', self.Uz, self.dUz), ('bz', self.bz, self.dbz),
            ('Wr', self.Wr, self.dWr), ('Ur', self.Ur, self.dUr), ('br', self.br, self.dbr),
            ('Wh', self.Wh, self.dWh), ('Uh', self.Uh, self.dUh), ('bh', self.bh, self.dbh),
        ]


class AttentionLayer:
    """
    Soft attention — learns which words in the sentence matter most.
    Computes a weighted sum of all GRU hidden states.
    """

    def __init__(self, hidden_dim):
        self.W = np.random.randn(hidden_dim, 1) * 0.1
        self.dW = np.zeros_like(self.W)

    def _softmax(self, x):
        x = x - x.max(axis=1, keepdims=True)
        e = np.exp(x)
        return e / (e.sum(axis=1, keepdims=True) + 1e-9)

    def forward(self, H):
        """
        H: (batch, seq_len, hidden_dim)
        Returns context: (batch, hidden_dim)
        """
        self._H = H
        scores = H @ self.W              # (batch, seq_len, 1)
        scores = scores.squeeze(-1)      # (batch, seq_len)
        self._alpha = self._softmax(scores)  # (batch, seq_len)
        context = (self._alpha[:, :, None] * H).sum(axis=1)  # (batch, hidden)
        return context

    def backward(self, d_context):
        """d_context: (batch, hidden_dim)"""
        batch, seq_len, hidden = self._H.shape
        # d_alpha
        d_alpha = (d_context[:, None, :] * self._H).sum(axis=-1)  # (batch, seq_len)
        # softmax backward
        alpha = self._alpha
        d_scores = alpha * (d_alpha - (d_alpha * alpha).sum(axis=1, keepdims=True))
        d_scores = d_scores[:, :, None]  # (batch, seq_len, 1)
        self.dW = (self._H.transpose(0, 2, 1) @ d_scores).sum(axis=0)  # (hidden, 1)
        dH_from_scores = d_scores @ self.W.T  # (batch, seq_len, hidden)
        dH_from_context = self._alpha[:, :, None] * d_context[:, None, :]
        return dH_from_scores + dH_from_context

    def params(self):
        uid = str(id(self))[-4:]
        return [(f'W_attn_{uid}', self.W, self.dW)]


class DenseLayer:
    """Fully connected layer with optional activation."""

    def __init__(self, in_dim, out_dim, activation='relu'):
        self.activation = activation
        scale = np.sqrt(2.0 / in_dim)
        self.W  = np.random.randn(in_dim, out_dim) * scale
        self.b  = np.zeros(out_dim)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def _relu(self, x):      return np.maximum(0, x)
    def _relu_back(self, x): return (x > 0).astype(float)

    def forward(self, x):
        self._x = x
        self._z = x @ self.W + self.b
        if self.activation == 'relu':
            self._out = self._relu(self._z)
        else:
            self._out = self._z
        return self._out

    def backward(self, dout):
        if self.activation == 'relu':
            dout = dout * self._relu_back(self._z)
        self.dW = self._x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T

    def params(self):
        uid = str(id(self))[-4:]
        return [(f'W_{uid}', self.W, self.dW), (f'b_{uid}', self.b, self.db)]


class DropoutLayer:
    def __init__(self, rate=0.3):
        self.rate = rate
        self.training = True

    def forward(self, x):
        if self.training:
            self._mask = (np.random.rand(*x.shape) > self.rate) / (1 - self.rate)
            return x * self._mask
        return x

    def backward(self, dout):
        return dout * self._mask if self.training else dout


class IntentClassifier:
    """
    Full RNN intent classifier model.
    
    Pipeline:
      Tokenizer → Embedding → GRU → Attention → Dense(ReLU) → Dropout → Output(Softmax)
    """

    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64,
                 num_classes=8, max_len=20):
        self.embed  = EmbeddingLayer(vocab_size, embed_dim)
        self.gru    = GRULayer(embed_dim, hidden_dim)
        self.attn   = AttentionLayer(hidden_dim)
        self.dense1 = DenseLayer(hidden_dim, 32, activation='relu')
        self.drop   = DropoutLayer(rate=0.3)
        self.dense2 = DenseLayer(32, num_classes, activation='none')
        self.num_classes = num_classes
        self.layers = [self.embed, self.gru, self.attn,
                       self.dense1, self.drop, self.dense2]

    def _softmax(self, x):
        x = x - x.max(axis=1, keepdims=True)
        e = np.exp(x)
        return e / (e.sum(axis=1, keepdims=True) + 1e-9)

    def forward(self, token_ids, training=True):
        self.drop.training = training
        x = self.embed.forward(token_ids)  # (B, L, E)
        h = self.gru.forward(x)            # (B, L, H)
        c = self.attn.forward(h)           # (B, H)
        d = self.dense1.forward(c)         # (B, 32)
        d = self.drop.forward(d)           # (B, 32)
        o = self.dense2.forward(d)         # (B, num_classes)
        probs = self._softmax(o)           # (B, num_classes)
        self._probs = probs
        return probs

    def backward(self, y_true):
        """Cross-entropy loss backward."""
        batch = y_true.shape[0]
        dlogits = self._probs.copy()
        dlogits[np.arange(batch), y_true] -= 1
        dlogits /= batch

        d = self.dense2.backward(dlogits)
        d = self.drop.backward(d)
        d = self.dense1.backward(d)
        d = self.attn.backward(d)
        d = self.gru.backward(d)
        self.embed.backward(d)

    def all_params(self):
        params = []
        for layer in [self.embed, self.gru, self.attn, self.dense1, self.dense2]:
            params.extend(layer.params())
        return params

    def cross_entropy_loss(self, probs, y_true):
        eps = 1e-9
        return -np.mean(np.log(probs[np.arange(len(y_true)), y_true] + eps))

    def accuracy(self, probs, y_true):
        return (probs.argmax(axis=1) == y_true).mean()

    def set_training(self, training):
        self.drop.training = training

    def save(self, path):
        # Use positional index as key — avoids name collision issues
        weights = {str(i): arr.tolist()
                   for i, (_, arr, _) in enumerate(self.all_params())}
        with open(path, 'w') as f:
            json.dump(weights, f)

    def load(self, path):
        with open(path) as f:
            weights = json.load(f)
        for i, (_, arr, _) in enumerate(self.all_params()):
            key = str(i)
            if key in weights:
                loaded = np.array(weights[key])
                if loaded.shape == arr.shape:
                    arr[:] = loaded


class AdamOptimizer:
    """Adam optimizer — same as used in GPT/BERT."""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.t     = 0
        self.m     = {}
        self.v     = {}

    def step(self, params):
        self.t += 1
        for name, arr, grad in params:
            if name not in self.m:
                self.m[name] = np.zeros_like(arr)
                self.v[name] = np.zeros_like(arr)
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grad**2
            m_hat = self.m[name] / (1 - self.beta1**self.t)
            v_hat = self.v[name] / (1 - self.beta2**self.t)
            arr  -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
