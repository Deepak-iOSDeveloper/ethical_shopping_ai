"""
Train the EcoMind RNN Intent Classifier.
Run: python ecomind_rnn/train_rnn.py
"""

import numpy as np
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecomind_rnn.rnn_model import (
    IntentClassifier, Tokenizer, AdamOptimizer
)
from ecomind_rnn.training_data import TRAINING_DATA, INTENT_CLASSES

SAVE_DIR = os.path.join(os.path.dirname(__file__), 'saved_model')
os.makedirs(SAVE_DIR, exist_ok=True)


def augment_data(texts, labels, factor=6):
    """Augment training data by word dropout and shuffling."""
    aug_texts, aug_labels = list(texts), list(labels)
    for text, label in zip(texts, labels):
        words = text.split()
        for _ in range(factor - 1):
            if len(words) <= 2:
                aug_texts.append(text)
            else:
                # Random word dropout (20%)
                kept = [w for w in words if np.random.rand() > 0.2]
                if not kept:
                    kept = words
                aug_texts.append(' '.join(kept))
            aug_labels.append(label)
    return aug_texts, aug_labels


def train():
    print("\n" + "="*55)
    print("  EcoMind RNN Intent Classifier — Training")
    print("="*55)

    # ── Prepare data ──────────────────────────────────────
    texts  = [d[0] for d in TRAINING_DATA]
    labels = [d[1] for d in TRAINING_DATA]

    print(f"\nOriginal samples: {len(texts)}")
    print(f"Intent classes  : {len(INTENT_CLASSES)}")

    # Class distribution
    from collections import Counter
    dist = Counter(labels)
    for idx, name in enumerate(INTENT_CLASSES):
        print(f"  [{idx}] {name:20s}: {dist[idx]:3d} samples")

    # Augment
    texts, labels = augment_data(texts, labels, factor=8)
    print(f"\nAfter augmentation: {len(texts)} samples")

    # Shuffle
    idx = np.random.permutation(len(texts))
    texts  = [texts[i]  for i in idx]
    labels = [labels[i] for i in idx]

    # Train/val split 85/15
    split  = int(len(texts) * 0.85)
    tr_txt, vl_txt = texts[:split], texts[split:]
    tr_lbl, vl_lbl = labels[:split], labels[split:]

    # ── Build tokenizer ───────────────────────────────────
    tok = Tokenizer(max_vocab=2000, max_len=20)
    tok.build_vocab(texts)
    print(f"Vocabulary size : {tok.vocab_size}")

    X_train = tok.encode_batch(tr_txt)
    y_train = np.array(tr_lbl, dtype=np.int32)
    X_val   = tok.encode_batch(vl_txt)
    y_val   = np.array(vl_lbl, dtype=np.int32)

    # ── Build model ───────────────────────────────────────
    model = IntentClassifier(
        vocab_size  = tok.vocab_size,
        embed_dim   = 32,
        hidden_dim  = 64,
        num_classes = len(INTENT_CLASSES),
        max_len     = 20,
    )
    opt = AdamOptimizer(lr=0.002)

    # ── Training loop ─────────────────────────────────────
    EPOCHS     = 120
    BATCH_SIZE = 32
    best_val_acc = 0.0
    patience     = 15
    wait         = 0

    print(f"\nTraining for {EPOCHS} epochs, batch={BATCH_SIZE}...\n")

    for epoch in range(1, EPOCHS + 1):
        # Shuffle train
        perm   = np.random.permutation(len(X_train))
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]

        tr_loss, tr_acc = 0.0, 0.0
        n_batches = 0

        for i in range(0, len(X_shuf), BATCH_SIZE):
            xb = X_shuf[i:i+BATCH_SIZE]
            yb = y_shuf[i:i+BATCH_SIZE]

            probs = model.forward(xb, training=True)
            loss  = model.cross_entropy_loss(probs, yb)
            acc   = model.accuracy(probs, yb)
            model.backward(yb)
            opt.step(model.all_params())

            tr_loss += loss
            tr_acc  += acc
            n_batches += 1

        tr_loss /= n_batches
        tr_acc  /= n_batches

        # Validation
        v_probs  = model.forward(X_val, training=False)
        v_loss   = model.cross_entropy_loss(v_probs, y_val)
        v_acc    = model.accuracy(v_probs, y_val)

        if epoch % 10 == 0 or epoch <= 5:
            print(f"Epoch {epoch:3d}/{EPOCHS} | "
                  f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.2%} | "
                  f"Val Loss: {v_loss:.4f} Acc: {v_acc:.2%}")

        # Early stopping + best model
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            wait = 0
            model.save(os.path.join(SAVE_DIR, 'rnn_weights.json'))
        else:
            wait += 1
            if wait >= patience:
                print(f"\nEarly stopping at epoch {epoch}. Best val acc: {best_val_acc:.2%}")
                break

    print(f"\n✅ Best validation accuracy: {best_val_acc:.2%}")

    # ── Save tokenizer ────────────────────────────────────
    tok.save(os.path.join(SAVE_DIR, 'tokenizer.json'))

    # Save model config
    config = {
        'vocab_size': tok.vocab_size,
        'embed_dim':  32,
        'hidden_dim': 64,
        'num_classes': len(INTENT_CLASSES),
        'max_len':    20,
        'intent_classes': INTENT_CLASSES,
        'best_val_acc': round(best_val_acc, 4),
    }
    with open(os.path.join(SAVE_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✅ Model saved to {SAVE_DIR}/")
    print("\nIntent classes:")
    for i, name in enumerate(INTENT_CLASSES):
        print(f"  {i}: {name}")

    # ── Per-class accuracy ────────────────────────────────
    model.load(os.path.join(SAVE_DIR, 'rnn_weights.json'))
    probs_all = model.forward(X_val, training=False)
    preds     = probs_all.argmax(axis=1)
    print("\nPer-class accuracy on validation set:")
    for idx, name in enumerate(INTENT_CLASSES):
        mask     = y_val == idx
        if mask.sum() == 0:
            continue
        cls_acc  = (preds[mask] == idx).mean()
        print(f"  {name:22s}: {cls_acc:.0%}  ({mask.sum()} samples)")

    return model, tok


if __name__ == '__main__':
    np.random.seed(42)
    train()
