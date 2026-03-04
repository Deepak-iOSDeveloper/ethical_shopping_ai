# 🌿 EcoMind — AI-Driven Ethical Shopping Assistant

A complete AI-powered web application that recommends eco-friendly and ethical products using Machine Learning.

## 🧠 AI Model Architecture

| Component | Library | Purpose |
|-----------|---------|---------|
| Data layer | **Pandas** | Product DataFrame, filtering, feature engineering |
| Numerics | **NumPy** | Array ops, cosine similarity matrix |
| ML Model | **Scikit-learn** | RandomForestClassifier, MinMaxScaler, LabelEncoder |
| Web Framework | **Django** | Views, Templates, REST API endpoints |

### Scoring Formula
```
composite_score = (eco_score × 0.40) + (ethics_score × 0.40) + (10 - carbon_footprint × 0.20)
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
source venv/bin/activate        # On Windows: venv\Scripts\activate
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

---

## 📁 Project Structure

```
ethical_shopping_ai/
│
├── ai_model/
│   ├── __init__.py
│   └── model.py              ← Core ML model (Pandas + NumPy + Scikit-learn)
│
├── ethical_shopping/         ← Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── shop_assistant/           ← Django app
│   ├── views.py              ← Connects Django ↔ AI model
│   ├── models.py             ← SearchLog, FavoriteProduct
│   ├── urls.py               ← URL routing
│   └── templates/
│       └── shop_assistant/
│           ├── base.html     ← Shared layout
│           ├── home.html     ← Dashboard + search
│           ├── recommend.html← AI results page
│           ├── product_detail.html ← Detailed product view
│           └── about.html    ← ML explanation page
│
├── manage.py
└── requirements.txt
```

---

## 🔌 REST API Endpoints

```
GET  /api/recommend/?category=Clothing&budget=100&keyword=organic
GET  /api/stats/
GET  /api/categories/
POST /api/recommend/    (JSON body with same params)
```

### Example API Response
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
      "price": 45,
      "sustainability_cert": "GOTS"
    }
  ]
}
```

---

## 🤖 ML Features Explained

### 1. RandomForest Classifier
- **Input features**: eco_score, ethics_score, carbon_footprint, price
- **Target**: Binary — is the product "highly ethical" (composite ≥ 8.8)?
- **Output**: Probability % shown as "AI: XX% ethical"

### 2. Content-Based Filtering
- Cosine similarity on normalized [eco_score, ethics_score, carbon_fp, category_encoded]
- Powers "You Might Also Like" recommendations

### 3. Composite Scoring
- Weighted formula combining eco + ethics + low carbon
- Transparent, interpretable ranking for every product

---

## 📊 Dataset

30 real-world ethical products from 15+ brands including:
Patagonia, Allbirds, Veja, Eileen Fisher, Nudie Jeans, Girlfriend Collective,
Manduka, Lush, Equal Exchange, BioLite, and more.

Categories: Clothing, Footwear, Sportswear, Personal Care, Kitchen, Home, Accessories, Outdoors, Food, Sports

Certifications covered: GOTS, B-Corp, Fair Trade, OEKO-TEX, COSMOS, FSC

---

## 🎓 Academic Note

This project demonstrates:
- **Data Science**: Pandas DataFrames, feature engineering, normalization
- **Machine Learning**: Supervised classification, similarity-based recommendation
- **Web Development**: Django MVC pattern, REST APIs, template rendering
- **Software Engineering**: Modular design, singleton pattern, clean separation of concerns

Built for university AI/ML course project.
Build with love by Deepak Kumar Behera

