"""
AI-Driven Ethical Shopping Assistant - Core ML Model
Uses: numpy, pandas, scikit-learn
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# ─────────────────────────────────────────────
# DATASET — 400+ products across all daily-life categories
# prices in USD (approx: $1 = ₹83)
# columns: name, brand, category, eco_score, ethics_score,
#          price_usd, sustainability_cert, materials, carbon_footprint, description
# ─────────────────────────────────────────────
def _load_ai_scored_products():
    """Load all products with AI-predicted scores from EcoMindNet."""
    import json as _j, os as _o
    json_path = _o.path.join(_o.path.dirname(__file__), 'ai_scored_products.json')
    if _o.path.exists(json_path):
        with open(json_path) as f:
            data = _j.load(f)
        print(f"[EcoMind] Loaded {len(data)} AI-scored products ✅")
        return [tuple(row) for row in data]
    try:
        from ai_model.products_extended import EXTENDED_PRODUCTS as _E
    except ImportError:
        try:
            from products_extended import EXTENDED_PRODUCTS as _E
        except ImportError:
            _E = []
    print("[EcoMind] Using original scores (run score_products.py to enable AI scoring)")
    return list(_E)

PRODUCTS_DATA = _load_ai_scored_products()


COLUMNS = ["name", "brand", "category", "eco_score", "ethics_score", "price",
           "sustainability_cert", "materials", "carbon_footprint", "description"]

BRAND_STORY = {
    "Patagonia":            "Donates 1% of sales to environmental causes. B-Corp certified. Fighting climate change since 1973.",
    "Allbirds":             "Carbon-neutral company. Natural materials only. Mission to prove fashion can be sustainable.",
    "Veja":                 "Transparent supply chain. Amazon rubber, fair-trade cotton. No advertising — invests it in ethics.",
    "Eileen Fisher":        "Take-back program for old clothes. Women-owned, B-Corp certified. Lifelong garments.",
    "Nudie Jeans":          "Free repairs for life. Organic cotton. Circular fashion pioneer from Sweden.",
    "Girlfriend Collective":"Sizes XXS–6XL. Made from recycled bottles. Inclusive and sustainable.",
    "Manduka":              "Lifetime guarantee on mats. PVC-free. Yoga gear that outlasts trends.",
    "Thought Clothing":     "Bamboo and hemp specialists. Mindful manufacturing. UK ethical fashion leader.",
    "Voltaic Systems":      "Solar-powered gear for adventures. Military-grade sustainability.",
    "Baggu":                "Minimal waste, maximum reuse. Organic cotton and recycled nylon.",
    "Lush":                 "100% vegetarian. Fighting animal testing globally. Package-free products.",
    "BioLite":              "Access to energy for all. Every purchase funds clean energy for off-grid communities.",
    "Equal Exchange":       "Worker-owned co-op. 100% fair trade. Direct relationships with farmers.",
    "Pela":                 "First compostable phone case. Ocean plastic warrior. Circular economy pioneer.",
    "Organic India":        "Farmer-owned company. Regenerative organic agriculture in India since 1997.",
    "Dr. Bronner's":        "Family-owned since 1948. All profits fund social and environmental causes.",
    "Ethique":              "World's first plastic-free beauty brand. Saves 50M+ plastic bottles yearly.",
    "Tentree":              "Plants 10 trees per purchase. B-Corp certified. 1 billion trees by 2030.",
    "Avocado Green":        "Certified organic and vegan mattress brand. Carbon negative company.",
    "Fairphone":            "World's most ethical smartphone — modular, repairable, conflict-free minerals.",
    "Prana":                "Fair-trade yoga apparel. Organic cotton pioneer since 1992.",
    "Outerknown":           "Founded by pro surfer Kelly Slater. Bluesign certified. Radical transparency.",
    "Naadam":               "Direct from Mongolian herders. No middlemen. Preserving nomadic culture.",
    "FabIndia":             "Connects 55,000 Indian craft artisans to global markets since 1960.",
    "Khadi Gramodyog":      "Gandhian vision of village self-reliance. Hand-spun, hand-woven fabric.",
    "GoCoop":               "India's largest handloom cooperative. Directly empowers 5,000 weavers.",
    "Hape":                 "World's largest maker of wooden toys. FSC certified, water-based paints.",
    "Brilliant Earth":      "Beyond conflict-free. Lab-grown diamonds, recycled precious metals.",
    "Vahdam":               "India's leading tea brand. Directly sources from farmers, carbon-positive company.",
    "Anokhi":               "Champions Rajasthani block-printing traditions since 1970. Natural dyes only.",
    "24 Mantra":            "India's largest organic food brand. Works with 20,000 smallholder farmers.",
    "Forest Essentials":    "Luxury Ayurvedic beauty inspired by ancient Indian science. Slow beauty.",
    "Burt's Bees":          "Natural personal care since 1984. Committed to 100% natural ingredients.",
    "Weleda":               "Biodynamic farming and anthroposophic medicine since 1921. No synthetics.",
    "Back Market":          "World's leading refurbished tech marketplace. Every device is tested, certified.",
    "Hevea":                "Natural rubber baby products from certified organic rubber plantations.",
    "Green Toys":           "All toys made from 100% recycled US milk jugs. No BPA, phthalates, or PVC.",
}


class EthicalShoppingAI:
    """
    Core AI model for recommending ethical & eco-friendly products.
    Uses content-based filtering + RandomForest ML scoring.
    """

    def __init__(self):
        self.df = None
        self.scaler = MinMaxScaler()
        self.label_enc = LabelEncoder()
        self.rf_classifier = None
        self.feature_matrix = None
        self._build_dataset()
        self._train_model()

    def _build_dataset(self):
        self.df = pd.DataFrame(PRODUCTS_DATA, columns=COLUMNS)

        # ── Load user-saved products from DB (persists across restarts) ──
        try:
            import django
            import os as _os
            _os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ethical_shopping.settings')
            if django.apps.apps.ready:
                from shop_assistant.models import UserSavedProduct
                carbon_map = {'ultra_low': 0.3, 'low': 1.0, 'moderate': 2.5, 'high': 5.0}
                saved = UserSavedProduct.objects.filter(
                    eco_score__isnull=False, ethics_score__isnull=False
                )
                extra_rows = []
                for sp in saved:
                    extra_rows.append((
                        sp.name,
                        sp.brand or 'Unknown',
                        sp.category or 'General',
                        float(sp.eco_score or 5.0),
                        float(sp.ethics_score or 5.0),
                        float(sp.price or 10.0),
                        sp.cert or '',
                        sp.materials or '',
                        carbon_map.get(sp.carbon_level or 'moderate', 2.5),
                        sp.description or '',
                    ))
                if extra_rows:
                    extra_df = pd.DataFrame(extra_rows, columns=COLUMNS)
                    self.df = pd.concat([self.df, extra_df], ignore_index=True)
                    print(f"[EcoMind] ✅ Loaded {len(extra_rows)} user-saved products from DB")
        except Exception as e:
            pass  # DB not ready yet on first run — that's fine

        # Composite ethical score (weighted average)
        self.df["composite_score"] = (
            self.df["eco_score"] * 0.4 +
            self.df["ethics_score"] * 0.4 +
            (10 - self.df["carbon_footprint"].clip(upper=10)) * 0.2
        ).round(2)

        num_cols = ["eco_score", "ethics_score", "price", "carbon_footprint", "composite_score"]
        self.df[num_cols] = self.df[num_cols].astype(float)

        self.df["category_encoded"] = self.label_enc.fit_transform(self.df["category"])

        self.df["price_tier"] = pd.cut(
            self.df["price"],
            bins=[0, 15, 50, 150, 500, 9_999_999],
            labels=["budget", "affordable", "mid", "premium", "luxury"]
        )

        feature_cols = ["eco_score", "ethics_score", "carbon_footprint", "category_encoded"]
        feature_data = self.df[feature_cols].values
        self.feature_matrix = self.scaler.fit_transform(feature_data)

    def _train_model(self):
        X = self.df[["eco_score", "ethics_score", "carbon_footprint", "price"]].values
        y = (self.df["composite_score"] >= 8.8).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_classifier.fit(X_train, y_train)
        accuracy = self.rf_classifier.score(X_test, y_test)
        print(f"[AI Model] RandomForest trained on {len(self.df)} products. Accuracy: {accuracy:.2%}")

    def predict_ethical_rating(self, eco_score, ethics_score, carbon_fp, price):
        # Percentile-based scoring — more meaningful than RandomForest on a skewed dataset
        # (RF returned ~100% for everything because 68% of products already score >= 8.8)
        eco_pct    = float((self.df['eco_score']        <= eco_score).mean())
        ethics_pct = float((self.df['ethics_score']     <= ethics_score).mean())
        carbon_pct = float((self.df['carbon_footprint'] >= carbon_fp).mean())  # lower = better
        percentile = (eco_pct * 0.35 + ethics_pct * 0.35 + carbon_pct * 0.30)
        # Scale: median product ~50%, best ~95%, worst ~15%
        prob = 15.0 + (percentile * 80.0)
        return round(max(5.0, min(98.0, prob)), 1)

    def recommend(self, category=None, budget=None, keyword=None, top_n=50):
        df = self.df.copy()

        if category and category != "All":
            df = df[df["category"].str.lower() == category.lower()]
        if budget:
            df = df[df["price"] <= float(budget)]
        if keyword:
            kw = keyword.lower()
            mask = (
                df["name"].str.lower().str.contains(kw) |
                df["brand"].str.lower().str.contains(kw) |
                df["materials"].str.lower().str.contains(kw) |
                df["description"].str.lower().str.contains(kw) |
                df["sustainability_cert"].str.lower().str.contains(kw) |
                df["category"].str.lower().str.contains(kw)
            )
            df = df[mask]

        if df.empty:
            return df

        df = df.sort_values("composite_score", ascending=False)
        return df.head(top_n).reset_index(drop=True)

    def get_similar_products(self, product_name, top_n=3):
        try:
            # Use positional index (iloc position) not DataFrame index label
            mask = self.df["name"] == product_name
            if not mask.any():
                return pd.DataFrame()
            pos = mask.values.argmax()  # positional index in df

            # Safety check — feature_matrix must match df length
            if pos >= len(self.feature_matrix):
                return pd.DataFrame()

            sim_scores  = cosine_similarity([self.feature_matrix[pos]], self.feature_matrix)[0]
            sim_indices = np.argsort(sim_scores)[::-1]
            # Skip self, take top_n
            sim_indices = [i for i in sim_indices if i != pos][:top_n]
            return self.df.iloc[sim_indices][["name", "brand", "category", "composite_score", "price"]].reset_index(drop=True)
        except Exception as e:
            print(f"[EcoMind] get_similar_products error: {e}")
            return pd.DataFrame()

    def get_stats(self):
        return {
            "total_products": len(self.df),
            "total_brands": self.df["brand"].nunique(),
            "avg_eco_score": round(self.df["eco_score"].mean(), 2),
            "avg_ethics_score": round(self.df["ethics_score"].mean(), 2),
            "categories": sorted(self.df["category"].unique().tolist()),
            "top_brand": self.df.groupby("brand")["composite_score"].mean().idxmax(),
        }

    def get_brand_story(self, brand):
        return BRAND_STORY.get(brand, "A brand committed to ethical and sustainable practices.")

    def get_all_categories(self):
        return ["All"] + sorted(self.df["category"].unique().tolist())


_model_instance = None

def get_model():
    global _model_instance
    if _model_instance is None:
        _model_instance = EthicalShoppingAI()
        print(f"[EcoMind] Model loaded: {len(_model_instance.df)} products across {_model_instance.df['category'].nunique()} categories")
    return _model_instance

def reset_model():
    """Call this to force a fresh reload of the model."""
    global _model_instance
    _model_instance = None


def add_user_product(product_dict):
    """
    Add a user-saved product into the live DataFrame so it
    immediately participates in recommendations and similarity.
    Call reset_model() after to rebuild the ML model with it.

    product_dict keys: name, brand, category, description, materials,
                       cert, eco_score, ethics_score, carbon_footprint, price
    """
    model = get_model()

    carbon_map = {'ultra_low': 0.3, 'low': 1.0, 'moderate': 2.5, 'high': 5.0}

    new_row = {
        'name':               product_dict.get('name', 'Unknown Product'),
        'brand':              product_dict.get('brand', 'Unknown Brand'),
        'category':           product_dict.get('category', 'General'),
        'eco_score':          float(product_dict.get('eco_score', 5.0)),
        'ethics_score':       float(product_dict.get('ethics_score', 5.0)),
        'price':              float(product_dict.get('price', 10.0)),
        'sustainability_cert': product_dict.get('cert', ''),
        'materials':          product_dict.get('materials', ''),
        'carbon_footprint':   carbon_map.get(product_dict.get('carbon_level', 'moderate'), 2.5),
        'description':        product_dict.get('description', ''),
    }

    # Compute composite score
    new_row['composite_score'] = round(
        new_row['eco_score'] * 0.40 +
        new_row['ethics_score'] * 0.40 +
        (10 - new_row['carbon_footprint']) * 0.20, 2
    )

    # Append to live DataFrame
    new_df = pd.DataFrame([new_row])
    model.df = pd.concat([model.df, new_df], ignore_index=True)

    # Rebuild features + retrain RandomForest with the new data
    # Re-encode categories (new category might have appeared)
    try:
        model.df['category_encoded'] = model.label_enc.transform(model.df['category'])
    except ValueError:
        model.df['category_encoded'] = model.label_enc.fit_transform(model.df['category'])

    model.df['composite_score'] = (
        model.df['eco_score'] * 0.4 +
        model.df['ethics_score'] * 0.4 +
        (10 - model.df['carbon_footprint'].clip(upper=10)) * 0.2
    ).round(2)

    feature_cols  = ['eco_score', 'ethics_score', 'carbon_footprint', 'category_encoded']
    model.feature_matrix = model.scaler.fit_transform(model.df[feature_cols].values)
    model._train_model()

    print(f"[EcoMind] ✅ User product added: '{new_row['name']}' — total products: {len(model.df)}")
    return len(model.df)
