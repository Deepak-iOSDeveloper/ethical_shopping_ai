"""
EcoMind Neural Network — Built from Scratch with NumPy
=======================================================
A multi-output text classifier that predicts:
  - eco_score        (0–10)
  - ethics_score     (0–10)
  - carbon_level     (0=ultra_low, 1=low, 2=moderate, 3=high)
  - certification tags (multi-label: organic, fair_trade, vegan, recycled, natural)

Architecture:
  Text → TF-IDF Features → Embedding Layer → Hidden Layers → Multi-Output Heads
  [vocab_size] → [128] → [64] → [32] → [regression + classification heads]
"""

import numpy as np
import pickle
import os

# ── Reproducibility ──────────────────────────────────────────
np.random.seed(42)


# ════════════════════════════════════════════════════════════
# 1.  ACTIVATION FUNCTIONS
# ════════════════════════════════════════════════════════════
def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_grad(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)


# ════════════════════════════════════════════════════════════
# 2.  LOSS FUNCTIONS
# ════════════════════════════════════════════════════════════
def mse_loss(pred, target):
    return np.mean((pred - target) ** 2)

def mse_grad(pred, target):
    return 2 * (pred - target) / pred.shape[0]

def bce_loss(pred, target, eps=1e-7):
    pred = np.clip(pred, eps, 1 - eps)
    return -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))

def bce_grad(pred, target, eps=1e-7):
    pred = np.clip(pred, eps, 1 - eps)
    return (pred - target) / (pred * (1 - pred) * pred.shape[0])


# ════════════════════════════════════════════════════════════
# 3.  DENSE LAYER
# ════════════════════════════════════════════════════════════
class DenseLayer:
    def __init__(self, in_dim, out_dim, activation='relu'):
        # He initialisation for ReLU families
        scale = np.sqrt(2.0 / in_dim)
        self.W = np.random.randn(in_dim, out_dim) * scale
        self.b = np.zeros(out_dim)
        self.activation = activation
        # Adam state
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)
        self.t  = 0
        # Cache for backprop
        self.cache_x = None
        self.cache_z = None

    def forward(self, x):
        self.cache_x = x
        z = x @ self.W + self.b
        self.cache_z = z
        if self.activation == 'relu':
            return relu(z)
        elif self.activation == 'leaky_relu':
            return leaky_relu(z)
        elif self.activation == 'sigmoid':
            return sigmoid(z)
        elif self.activation == 'linear':
            return z
        return relu(z)

    def backward(self, d_out):
        if self.activation == 'relu':
            d_z = d_out * relu_grad(self.cache_z)
        elif self.activation == 'leaky_relu':
            d_z = d_out * leaky_relu_grad(self.cache_z)
        elif self.activation == 'sigmoid':
            d_z = d_out * sigmoid_grad(self.cache_z)
        else:
            d_z = d_out

        self.dW = self.cache_x.T @ d_z
        self.db = d_z.sum(axis=0)
        return d_z @ self.W.T

    def adam_update(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        # Weights
        self.mW = beta1 * self.mW + (1 - beta1) * self.dW
        self.vW = beta2 * self.vW + (1 - beta2) * self.dW ** 2
        mW_hat  = self.mW / (1 - beta1 ** self.t)
        vW_hat  = self.vW / (1 - beta2 ** self.t)
        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        # Biases
        self.mb = beta1 * self.mb + (1 - beta1) * self.db
        self.vb = beta2 * self.vb + (1 - beta2) * self.db ** 2
        mb_hat  = self.mb / (1 - beta1 ** self.t)
        vb_hat  = self.vb / (1 - beta2 ** self.t)
        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)


# ════════════════════════════════════════════════════════════
# 4.  BATCH NORM LAYER
# ════════════════════════════════════════════════════════════
class BatchNorm:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.gamma   = np.ones(dim)
        self.beta    = np.zeros(dim)
        self.eps     = eps
        self.momentum= momentum
        self.running_mean = np.zeros(dim)
        self.running_var  = np.ones(dim)
        self.training = True
        self.cache    = None
        # Adam state
        self.mg = np.zeros(dim); self.vg = np.zeros(dim)
        self.mb = np.zeros(dim); self.vb = np.zeros(dim)
        self.t  = 0
        self.dg = None; self.db_ = None

    def forward(self, x):
        if self.training:
            mean = x.mean(axis=0)
            var  = x.var(axis=0)
            self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mean
            self.running_var  = (1-self.momentum)*self.running_var  + self.momentum*var
        else:
            mean = self.running_mean
            var  = self.running_var
        x_hat = (x - mean) / np.sqrt(var + self.eps)
        self.cache = (x, x_hat, mean, var)
        return self.gamma * x_hat + self.beta

    def backward(self, d_out):
        x, x_hat, mean, var = self.cache
        N = x.shape[0]
        self.dg  = (d_out * x_hat).sum(axis=0)
        self.db_ = d_out.sum(axis=0)
        dx_hat   = d_out * self.gamma
        dvar  = (dx_hat * (x - mean) * -0.5 * (var + self.eps)**-1.5).sum(axis=0)
        dmean = (dx_hat * -1/np.sqrt(var + self.eps)).sum(axis=0)
        return dx_hat/np.sqrt(var+self.eps) + dvar*2*(x-mean)/N + dmean/N

    def adam_update(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for attr, grad, m_attr, v_attr in [
            ('gamma', self.dg, 'mg', 'vg'),
            ('beta',  self.db_,'mb', 'vb'),
        ]:
            m = getattr(self, m_attr)
            v = getattr(self, v_attr)
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            setattr(self, m_attr, m)
            setattr(self, v_attr, v)
            m_hat = m / (1 - beta1 ** self.t)
            v_hat = v / (1 - beta2 ** self.t)
            setattr(self, attr, getattr(self, attr) - lr * m_hat / (np.sqrt(v_hat) + eps))


# ════════════════════════════════════════════════════════════
# 5.  DROPOUT LAYER
# ════════════════════════════════════════════════════════════
class Dropout:
    def __init__(self, rate=0.3):
        self.rate    = rate
        self.mask    = None
        self.training = True

    def forward(self, x):
        if self.training and self.rate > 0:
            self.mask = (np.random.rand(*x.shape) > self.rate) / (1 - self.rate)
            return x * self.mask
        return x

    def backward(self, d_out):
        if self.training and self.mask is not None:
            return d_out * self.mask
        return d_out

    def adam_update(self, **kwargs):
        pass  # no params


# ════════════════════════════════════════════════════════════
# 6.  ECOMIND NEURAL NETWORK
# ════════════════════════════════════════════════════════════
class EcoMindNet:
    """
    Multi-output neural network for ethical product scoring.

    Inputs : TF-IDF text features (vocab_size,)
    Outputs:
      - eco_score    : float 0-10  (regression)
      - ethics_score : float 0-10  (regression)
      - carbon_level : int  0-3    (4-class classification)
      - tags         : 5-dim binary (multi-label: organic, fair_trade, vegan, recycled, natural)
    """

    def __init__(self, input_dim, hidden1=256, hidden2=128, hidden3=64):
        print(f"[EcoMindNet] Building network: {input_dim} → {hidden1} → {hidden2} → {hidden3} → outputs")

        # ── Shared backbone ──────────────────────────────────
        self.l1      = DenseLayer(input_dim, hidden1, 'leaky_relu')
        self.bn1     = BatchNorm(hidden1)
        self.drop1   = Dropout(0.3)

        self.l2      = DenseLayer(hidden1, hidden2, 'leaky_relu')
        self.bn2     = BatchNorm(hidden2)
        self.drop2   = Dropout(0.2)

        self.l3      = DenseLayer(hidden2, hidden3, 'leaky_relu')
        self.bn3     = BatchNorm(hidden3)
        self.drop3   = Dropout(0.1)

        # ── Output heads ─────────────────────────────────────
        self.eco_head    = DenseLayer(hidden3, 1,  'linear')   # regression
        self.eth_head    = DenseLayer(hidden3, 1,  'linear')   # regression
        self.carbon_head = DenseLayer(hidden3, 4,  'linear')   # 4-class softmax
        self.tag_head    = DenseLayer(hidden3, 5,  'sigmoid')  # multi-label

        self.all_layers = [
            self.l1, self.bn1, self.drop1,
            self.l2, self.bn2, self.drop2,
            self.l3, self.bn3, self.drop3,
            self.eco_head, self.eth_head,
            self.carbon_head, self.tag_head,
        ]

        self.training = True
        self._set_training_mode(True)

    def _set_training_mode(self, mode):
        self.training = mode
        for layer in self.all_layers:
            if hasattr(layer, 'training'):
                layer.training = mode

    def forward(self, x):
        # Backbone
        h = self.l1.forward(x)
        h = self.bn1.forward(h)
        h = self.drop1.forward(h)

        h = self.l2.forward(h)
        h = self.bn2.forward(h)
        h = self.drop2.forward(h)

        h = self.l3.forward(h)
        h = self.bn3.forward(h)
        h = self.drop3.forward(h)

        self._h = h  # cache for backprop

        # Heads
        eco    = self.eco_head.forward(h)          # (N, 1)
        ethics = self.eth_head.forward(h)          # (N, 1)
        carbon = softmax(self.carbon_head.forward(h))  # (N, 4)
        tags   = self.tag_head.forward(h)          # (N, 5)

        return eco, ethics, carbon, tags

    def backward(self, eco_pred, eth_pred, carbon_pred, tag_pred,
                 eco_tgt,  eth_tgt,  carbon_tgt,  tag_tgt):

        # ── Head gradients ────────────────────────────────────
        d_eco    = mse_grad(eco_pred, eco_tgt)
        d_ethics = mse_grad(eth_pred, eth_tgt)

        # Softmax + cross-entropy gradient
        d_carbon = carbon_pred.copy()
        d_carbon[np.arange(len(carbon_tgt)), carbon_tgt] -= 1
        d_carbon /= len(carbon_tgt)

        d_tags = bce_grad(tag_pred, tag_tgt)

        # Backprop through heads
        d_h  = self.eco_head.backward(d_eco)
        d_h += self.eth_head.backward(d_ethics)
        d_h += self.carbon_head.backward(d_carbon)
        d_h += self.tag_head.backward(d_tags)

        # ── Backbone backprop ─────────────────────────────────
        d_h = self.drop3.backward(d_h)
        d_h = self.bn3.backward(d_h)
        d_h = self.l3.backward(d_h)

        d_h = self.drop2.backward(d_h)
        d_h = self.bn2.backward(d_h)
        d_h = self.l2.backward(d_h)

        d_h = self.drop1.backward(d_h)
        d_h = self.bn1.backward(d_h)
        d_h = self.l1.backward(d_h)

    def step(self, lr=0.001):
        for layer in self.all_layers:
            layer.adam_update(lr=lr)

    def save(self, path):
        """Serialize all layer weights to a .pkl file."""
        data = {}
        layer_names = [
            'l1','bn1','l2','bn2','l3','bn3',
            'eco_head','eth_head','carbon_head','tag_head'
        ]
        for name in layer_names:
            layer = getattr(self, name)
            layer_data = {}
            for attr in vars(layer):
                val = getattr(layer, attr)
                if isinstance(val, np.ndarray):
                    layer_data[attr] = val
            data[name] = layer_data
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"[EcoMindNet] Model saved → {path}")

    def load(self, path):
        """Load weights from a .pkl file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        layer_names = [
            'l1','bn1','l2','bn2','l3','bn3',
            'eco_head','eth_head','carbon_head','tag_head'
        ]
        for name in layer_names:
            layer = getattr(self, name)
            for attr, val in data[name].items():
                setattr(layer, attr, val)
        print(f"[EcoMindNet] Weights loaded ← {path}")
        return self
