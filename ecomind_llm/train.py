"""
EcoMind LLM — Training Script
==============================
Trains the EcoMindNet neural network on the product dataset.

Run:
    python train.py

Outputs:
    ecomind_llm/saved_model/ecomind_net.pkl      ← trained weights
    ecomind_llm/saved_model/tfidf_vectorizer.pkl ← fitted vectorizer
    ecomind_llm/saved_model/training_log.txt     ← loss history
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecomind_llm.neural_net   import EcoMindNet
from ecomind_llm.preprocessor import (
    build_input_text, TFIDFVectorizer,
    carbon_to_class, encode_tags,
    normalise_score, CARBON_LABELS, TAG_NAMES
)

# Import the product data directly from the AI model
from ai_model.model import PRODUCTS_DATA, COLUMNS


# ════════════════════════════════════════════════════════════
# 1.  LOAD DATASET
# ════════════════════════════════════════════════════════════
def load_dataset():
    import pandas as pd
    df = pd.DataFrame(PRODUCTS_DATA, columns=COLUMNS)
    print(f"[Train] Loaded {len(df)} products")
    return df


# ════════════════════════════════════════════════════════════
# 2.  DATA AUGMENTATION
#     Our dataset has 203 products. We augment to 1000+ samples
#     by creating variations with slight perturbations.
# ════════════════════════════════════════════════════════════
def augment_dataset(df, multiplier=6):
    """
    Augment by:
    - Shuffling word order in description
    - Dropping random words
    - Adding noise to numeric scores (small perturbation)
    - Combining materials/cert words differently
    """
    rows = df.to_dict('records')
    augmented = list(rows)  # start with originals

    np.random.seed(42)
    for _ in range(multiplier - 1):
        for row in rows:
            new_row = row.copy()

            # Text augmentation: drop ~20% of words randomly
            desc_words = str(row['description']).split()
            keep_mask  = np.random.rand(len(desc_words)) > 0.2
            new_desc   = ' '.join([w for w, k in zip(desc_words, keep_mask) if k])
            if new_desc:
                new_row['description'] = new_desc

            # Numeric noise: ±0.15 on scores (keep within 0-10)
            new_row['eco_score']    = float(np.clip(row['eco_score']    + np.random.randn()*0.15, 0, 10))
            new_row['ethics_score'] = float(np.clip(row['ethics_score'] + np.random.randn()*0.15, 0, 10))
            new_row['carbon_footprint'] = float(np.clip(row['carbon_footprint'] + abs(np.random.randn()*0.1), 0, 20))

            augmented.append(new_row)

    import pandas as pd
    aug_df = pd.DataFrame(augmented)
    print(f"[Train] After augmentation: {len(aug_df)} samples")
    return aug_df


# ════════════════════════════════════════════════════════════
# 3.  PREPARE FEATURES AND TARGETS
# ════════════════════════════════════════════════════════════
def prepare_data(df, vectorizer=None, fit=True):
    # Build input text
    texts = [build_input_text(row) for _, row in df.iterrows()]

    # Vectorise
    if fit:
        vectorizer = TFIDFVectorizer(max_features=512, ngram_max=2)
        X = vectorizer.fit_transform(texts)
    else:
        X = vectorizer.transform(texts)

    # Targets
    eco_targets    = np.array([[normalise_score(r['eco_score'])]    for _, r in df.iterrows()], dtype=np.float32)
    ethics_targets = np.array([[normalise_score(r['ethics_score'])] for _, r in df.iterrows()], dtype=np.float32)
    carbon_targets = np.array([carbon_to_class(r['carbon_footprint']) for _, r in df.iterrows()], dtype=np.int32)
    tag_targets    = np.array([
        encode_tags(r['sustainability_cert'], r['materials'], r['description'])
        for _, r in df.iterrows()
    ], dtype=np.float32)

    print(f"[Train] X shape: {X.shape}")
    print(f"[Train] Eco targets range: {eco_targets.min():.2f} – {eco_targets.max():.2f}")
    print(f"[Train] Carbon classes: {dict(zip(*np.unique(carbon_targets, return_counts=True)))}")
    print(f"[Train] Tag distribution: {tag_targets.mean(axis=0).round(2)}")

    return X, eco_targets, ethics_targets, carbon_targets, tag_targets, vectorizer


# ════════════════════════════════════════════════════════════
# 4.  MINI-BATCH GENERATOR
# ════════════════════════════════════════════════════════════
def batch_iter(X, eco, eth, carbon, tags, batch_size=32, shuffle=True):
    N = len(X)
    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, N, batch_size):
        end  = min(start + batch_size, N)
        b    = idx[start:end]
        yield X[b], eco[b], eth[b], carbon[b], tags[b]


# ════════════════════════════════════════════════════════════
# 5.  TRAINING LOOP
# ════════════════════════════════════════════════════════════
def train(epochs=120, lr=0.001, batch_size=32):
    print("\n" + "="*60)
    print("   EcoMind Neural Network — Training")
    print("="*60)

    # ── Load & augment ──────────────────────────────────────
    df     = load_dataset()
    aug_df = augment_dataset(df, multiplier=6)

    # ── Train/Val split ─────────────────────────────────────
    n_val   = max(20, int(0.15 * len(aug_df)))
    val_df  = aug_df.sample(n=n_val,  random_state=42)
    train_df= aug_df.drop(val_df.index)
    print(f"[Train] Train: {len(train_df)} | Val: {len(val_df)}")

    # ── Prepare features ────────────────────────────────────
    X_tr, eco_tr, eth_tr, car_tr, tag_tr, vectorizer = prepare_data(train_df, fit=True)
    X_val, eco_val, eth_val, car_val, tag_val, _     = prepare_data(val_df, vectorizer=vectorizer, fit=False)

    # ── Build model ──────────────────────────────────────────
    model = EcoMindNet(input_dim=X_tr.shape[1], hidden1=256, hidden2=128, hidden3=64)

    # ── LR schedule ──────────────────────────────────────────
    def get_lr(epoch):
        if epoch < 20:   return lr
        elif epoch < 60: return lr * 0.5
        elif epoch < 90: return lr * 0.2
        else:            return lr * 0.05

    best_val_loss = float('inf')
    best_weights_path = None
    log = []

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'Eco MAE':>9} {'Eth MAE':>9} {'Carbon Acc':>11} {'LR':>8}")
    print("-" * 75)

    for epoch in range(1, epochs + 1):
        model._set_training_mode(True)
        cur_lr = get_lr(epoch)
        train_loss = 0.0
        n_batches  = 0

        for Xb, eco_b, eth_b, car_b, tag_b in batch_iter(X_tr, eco_tr, eth_tr, car_tr, tag_tr, batch_size):
            # Forward
            eco_p, eth_p, car_p, tag_p = model.forward(Xb)

            # Individual losses
            l_eco    = np.mean((eco_p - eco_b)**2)
            l_eth    = np.mean((eth_p - eth_b)**2)
            l_carbon = -np.mean(np.log(car_p[np.arange(len(car_b)), car_b] + 1e-7))
            eps = 1e-7
            tp  = np.clip(tag_p, eps, 1-eps)
            l_tags   = -np.mean(tag_b * np.log(tp) + (1-tag_b) * np.log(1-tp))

            total_loss = 0.4*l_eco + 0.4*l_eth + 0.1*l_carbon + 0.1*l_tags
            train_loss += total_loss
            n_batches  += 1

            # Backward
            model.backward(eco_p, eth_p, car_p, tag_p,
                           eco_b, eth_b, car_b, tag_b)
            model.step(lr=cur_lr)

        train_loss /= n_batches

        # ── Validation ───────────────────────────────────────
        if epoch % 5 == 0 or epoch == 1:
            model._set_training_mode(False)
            eco_vp, eth_vp, car_vp, tag_vp = model.forward(X_val)

            val_eco_mae = np.mean(np.abs(eco_vp - eco_val)) * 10  # back to 0-10 scale
            val_eth_mae = np.mean(np.abs(eth_vp - eth_val)) * 10
            car_pred_cls= np.argmax(car_vp, axis=1)
            val_carbon_acc = np.mean(car_pred_cls == car_val) * 100

            l_eco_v = np.mean((eco_vp - eco_val)**2)
            l_eth_v = np.mean((eth_vp - eth_val)**2)
            l_car_v = -np.mean(np.log(car_vp[np.arange(len(car_val)), car_val]+1e-7))
            eps = 1e-7; tvp = np.clip(tag_vp, eps, 1-eps)
            l_tag_v = -np.mean(tag_val*np.log(tvp)+(1-tag_val)*np.log(1-tvp))
            val_loss = 0.4*l_eco_v + 0.4*l_eth_v + 0.1*l_car_v + 0.1*l_tag_v

            print(f"{epoch:>6} {train_loss:>12.5f} {val_loss:>10.5f} "
                  f"{val_eco_mae:>9.3f} {val_eth_mae:>9.3f} "
                  f"{val_carbon_acc:>10.1f}% {cur_lr:>8.5f}")

            log.append({
                'epoch': epoch, 'train_loss': round(float(train_loss), 5),
                'val_loss': round(float(val_loss), 5),
                'eco_mae': round(float(val_eco_mae), 3),
                'eth_mae': round(float(val_eth_mae), 3),
                'carbon_acc': round(float(val_carbon_acc), 1),
            })

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs("saved_model", exist_ok=True)
                model.save("ecomind_llm/saved_model/ecomind_net.pkl")
                best_weights_path = "saved_model/ecomind_net.pkl"

    # ── Save vectorizer & log ────────────────────────────────
    vectorizer.save("ecomind_llm/saved_model/tfidf_vectorizer.pkl")
    with open("saved_model/training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print("\n" + "="*60)
    print(f"  Training complete! Best val loss: {best_val_loss:.5f}")
    print(f"  Model saved → ecomind_llm/saved_model/")
    print("="*60)

    return model, vectorizer, log


# ════════════════════════════════════════════════════════════
# 6.  QUICK EVALUATION
# ════════════════════════════════════════════════════════════
def evaluate_on_examples(model, vectorizer):
    from ecomind_llm.predictor import EcoMindPredictor
    predictor = EcoMindPredictor()
    predictor.model      = model
    predictor.vectorizer = vectorizer
    predictor.ready      = True

    print("\n── Sample Predictions ──────────────────────────────────")
    test_cases = [
        {
            "name": "Organic Cotton Tee",
            "brand": "Patagonia",
            "category": "Clothing",
            "materials": "organic_cotton",
            "sustainability_cert": "GOTS",
            "description": "Classic organic cotton t-shirt, GOTS certified, fair trade factory."
        },
        {
            "name": "Bamboo Toothbrush",
            "brand": "Eco Brand",
            "category": "Personal Care",
            "materials": "bamboo",
            "sustainability_cert": "B-Corp",
            "description": "100% biodegradable bamboo toothbrush with natural bristles."
        },
        {
            "name": "Recycled Plastic Bottle",
            "brand": "GreenBottle",
            "category": "Kitchen",
            "materials": "recycled_plastic",
            "sustainability_cert": "Non-GMO",
            "description": "Water bottle made from 100% recycled ocean plastic."
        },
        {
            "name": "Some Random Cheap Shirt",
            "brand": "FastFashion Co",
            "category": "Clothing",
            "materials": "polyester",
            "sustainability_cert": "None",
            "description": "Cheap polyester shirt made in unverified factory."
        },
    ]

    for tc in test_cases:
        result = predictor.predict(tc)
        print(f"\nProduct : {tc['name']}")
        print(f"  Eco Score    : {result['eco_score']}/10")
        print(f"  Ethics Score : {result['ethics_score']}/10")
        print(f"  Carbon Level : {result['carbon_level']} ({result['carbon_label']})")
        print(f"  Tags         : {result['tags']}")
        print(f"  Confidence   : {result['confidence']}%")
        print(f"  Overall      : {result['overall_score']}/10")
    print()


if __name__ == "__main__":
    model, vectorizer, log = train(epochs=120, lr=0.001, batch_size=32)
    evaluate_on_examples(model, vectorizer)