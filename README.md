# 🌿 EcoMind — AI-Driven Ethical Shopping Assistant

### 🚀 Live at → [https://www.ecomindshop.online/](https://www.ecomindshop.online/)

A complete AI-powered web application that recommends eco-friendly and ethical products using a custom-built neural network and machine learning pipeline — **zero external APIs, 100% own code.**

---

## 🧠 AI Model Architecture

| Component | Library | Purpose |
|-----------|---------|---------|
| Data layer | **Pandas** | Product DataFrame, filtering, feature engineering |
| Numerics | **NumPy** | Neural network engine — matrix ops, gradients, backprop |
| ML Model | **Scikit-learn** | RandomForestClassifier, MinMaxScaler, LabelEncoder, cosine_similarity |
| Neural Network | **EcoMindNet (NumPy only)** | Custom multi-output deep neural network built from scratch |
| Optimiser | **Adam (custom)** | Implemented from scratch — adaptive lr, β1=0.9, β2=0.999 |
| Regularisation | **Dropout + BatchNorm** | Custom layers — prevents overfitting, stabilises training |
| Text Features | **TF-IDF (custom)** | 512-dimensional bigram vectoriser, no external NLP library |
| Web Framework | **Django** | Views, Templates, REST API endpoints |

---

## 🔬 EcoMindNet — Custom Neural Network

Built entirely from scratch using **NumPy only**. No PyTorch, no TensorFlow, no Keras.

### Architecture
```
Input (512 TF-IDF features)
    ↓
Dense(256) → BatchNorm → LeakyReLU → Dropout(0.30)
    ↓
Dense(128) → BatchNorm → LeakyReLU → Dropout(0.20)
    ↓
Dense(64)  → BatchNorm → LeakyReLU → Dropout(0.10)
    ↓
┌──────────────────┬──────────────────┬────────────────────┬──────────────────────┐
│  Eco Score Head  │ Ethics Score Head │ Carbon Level Head  │  Ethical Tags Head   │
│  Linear → (0-10) │  Linear → (0-10) │ Softmax → 4 classes│ Sigmoid → 5 labels   │
│  MSE loss        │  MSE loss        │ CrossEntropy loss  │ Binary CE loss       │
└──────────────────┴──────────────────┴────────────────────┴──────────────────────┘
```

### Training Results
| Metric | Value |
|--------|-------|
| Carbon classification accuracy | **95.1%** |
| Eco score MAE | **0.29** (out of 10) |
| Ethics score MAE | **0.30** (out of 10) |
| Training samples (after augmentation) | **1,218** |
| Original products | **203** |
| Training epochs | **120** |

### What was manually implemented
- Forward propagation through all layers
- Backpropagation via chain rule
- Adam optimiser (β1, β2, bias correction)
- Batch Normalisation (gamma, beta, running stats)
- Dropout (inverted dropout with train/inference modes)
- He weight initialisation
- Step-decay learning rate scheduler (0.001 → 0.0005 → 0.0002 → 0.00005)
- TF-IDF vectoriser with bigram support
- Data augmentation (word dropout, Gaussian noise, description shuffling)

---

## 🧮 Scoring Formulas

```python
# Composite Score — product database ranking
composite_score = (eco_score × 0.40) + (ethics_score × 0.40) + ((10 - carbon_footprint) × 0.20)

# Overall Score — chatbot card display
overall_score = (eco_score × 0.45) + (ethics_score × 0.45)
              + max(3 - carbon_level, 0) × 0.50   # carbon class bonus
              + len(ethical_tags) × 0.10            # per-tag bonus
```

---

## 🚀 Quick Setup (5 minutes)

### 1. Clone / Extract the project
```bash
cd ethical_shopping_ai
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Django migrations
```bash
python manage.py migrate
```

### 5. Start the server
```bash
python manage.py runserver
```

### 6. Open in browser
```
http://127.0.0.1:8000/
```

### Optional — Retrain EcoMindNet from scratch
```bash
python ecomind_llm/train.py
```

---

## 📁 Project Structure

```
ethical_shopping_ai/
│
├── ai_model/
│   ├── __init__.py
│   └── model.py                  ← Core ML (Pandas + Scikit-learn + RandomForest)
│
├── ecomind_llm/                  ← Custom neural network
│   ├── __init__.py
│   ├── neural_net.py             ← EcoMindNet — built from scratch with NumPy
│   ├── preprocessor.py           ← TF-IDF vectoriser + label encoders
│   ├── train.py                  ← Training pipeline with data augmentation
│   ├── predictor.py              ← Inference engine (singleton pattern)
│   └── saved_model/
│       ├── ecomind_net.pkl       ← Trained weights
│       ├── tfidf_vectorizer.pkl  ← Fitted vectoriser
│       └── training_log.json     ← Training history
│
├── ethical_shopping/             ← Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── shop_assistant/               ← Django app
│   ├── views.py                  ← Full pipeline: NLP → filter → EcoMindNet → HTML cards
│   ├── chatbot.py                ← NLP intent parser + Intent Guard (1000+ signals)
│   ├── models.py                 ← SearchLog, FavoriteProduct
│   ├── urls.py                   ← URL routing
│   └── templates/
│       └── shop_assistant/
│           ├── base.html         ← Shared layout
│           ├── home.html         ← Dashboard + search
│           ├── recommend.html    ← AI results page
│           ├── product_detail.html
│           └── about.html        ← Technical deep-dive page
│
├── manage.py
└── requirements.txt
```

---

## 🔌 REST API Endpoints

```
GET  /api/recommend/?category=Clothing&budget=500&keyword=organic
GET  /api/stats/
GET  /api/categories/
POST /api/recommend/     ← JSON body with same params
POST /api/chat/          ← EcoMind AI chatbot (natural language)
POST /api/predict/       ← EcoMindNet direct prediction for any product
```

### Example — Chatbot
```bash
POST /api/chat/
{"message": "show me organic food under ₹500"}
```

### Example — Predict
```bash
POST /api/predict/
{"name": "Hemp T-Shirt", "description": "organic fair trade certified", "category": "Clothing"}
```

### Example — Recommend response
```json
{
  "status": "success",
  "count": 4,
  "products": [
    {
      "name": "Organic Cotton Tee",
      "brand": "Patagonia",
      "eco_score": 9.2,
      "ethics_score": 9.5,
      "composite_score": 9.28,
      "ethical_probability": 96.5,
      "price_inr": 3780,
      "sustainability_cert": "GOTS",
      "carbon_footprint": 0.8
    }
  ]
}
```

---

## ⚙️ Full AI Pipeline

Every chatbot query runs through this 8-step pipeline — entirely on your own server:

```
User message
    │
    ▼
[1] Intent Guard        — 1000+ signals block off-topic / vulgar / political queries
    │
    ▼
[2] NLP Intent Parser   — extract budget (₹), category, ethical filters via regex
    │
    ▼
[3] Pandas Filter       — filter 203-product DataFrame by category + budget + keyword
    │
    ▼
[4] Ethical Re-ranking  — score each result against requested ethical filters
    │
    ▼
[5] TF-IDF Vectorise    — convert product text → 512-dim feature vector
    │
    ▼
[6] EcoMindNet Predict  — predict eco_score, ethics_score, carbon_level, ethical_tags
    │
    ▼
[7] Overall Scoring     — compute composite score + confidence badge
    │
    ▼
[8] HTML Card Builder   — render score bars, tags, ₹ price, CO₂ level, AI badge
```

---

## 🛡️ Intent Guard

Multi-layer content classifier protecting the chatbot from off-topic and harmful queries.

| Layer | Example triggers | Response |
|-------|-----------------|----------|
| Vulgar / abusive | "fuck", "idiot" | 🚫 Firm but polite refusal |
| Political | "BJP", "election", "Modi" | 🌍 Neutral redirect |
| Gender / discrimination | "feminist", "sexist" | 🤝 Respectful redirect |
| Off-topic | "weather", "cricket", "bitcoin", "recipe" | 🌿 "Sorry, couldn't find products for that" |

1000+ off-topic signals across 20 categories with smart shopping-context override to prevent false positives.

---

## 📊 Dataset

**203 real-world ethical products** across **12 categories** from 30+ brands including:
Patagonia, Allbirds, Veja, Eileen Fisher, Nudie Jeans, Girlfriend Collective,
Manduka, Lush, Equal Exchange, BioLite, and more.

**Price range:** ₹250 – ₹2,32,000

**Categories:** Clothing, Footwear, Sportswear, Personal Care, Kitchen, Home, Accessories, Outdoors, Food, Sports, Baby, Electronics

**Certifications:** GOTS, B-Corp, Fair Trade, OEKO-TEX, COSMOS, FSC, USDA Organic, Bluesign, Rainforest Alliance

---

## 🎓 Academic Note

This project demonstrates:

- **Custom Deep Learning** — Multi-output neural network built from scratch, no ML framework
- **Data Science** — Pandas DataFrames, feature engineering, TF-IDF, normalisation
- **Machine Learning** — RandomForest classifier, cosine similarity, content-based filtering
- **NLP** — Intent detection, keyword extraction, 1000-signal content safety filtering
- **Web Development** — Django MVC, REST APIs, template rendering
- **Software Engineering** — Modular design, singleton pattern, clean separation of concerns

Built for university AI/ML course project at **Lovely Professional University (LPU)**.  
Built with love by **Deepak Kumar Behera**
