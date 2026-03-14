"""
Views for Ethical Shopping Assistant
100% own models - No external API needed.
Pipeline: User message → NLP parser → Product filter → EcoMindNet scoring → Display
"""
import sys
import os
import json
from urllib.parse import unquote
from django.shortcuts import render
from django.http import JsonResponse, Http404
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from ai_model.model import get_model


def home(request):
    model = get_model()
    stats = model.get_stats()
    categories = model.get_all_categories()
    featured_df = model.recommend(top_n=6)
    featured = featured_df.to_dict('records') if not featured_df.empty else []
    return render(request, 'shop_assistant/home.html', {
        'stats': stats, 'categories': categories, 'featured': featured, 'page': 'home'
    })


def recommend(request):
    model = get_model()
    categories = model.get_all_categories()
    category = request.GET.get('category', 'All')
    budget   = request.GET.get('budget', None)
    keyword  = request.GET.get('keyword', None)
    results_df = model.recommend(
        category=category if category != 'All' else None,
        budget=float(budget) if budget else None,
        keyword=keyword, top_n=100
    )
    results = []
    for _, row in results_df.iterrows():
        product = row.to_dict()
        product['ethical_probability'] = model.predict_ethical_rating(
            row['eco_score'], row['ethics_score'], row['carbon_footprint'], row['price'])
        product['brand_story'] = model.get_brand_story(row['brand'])
        results.append(product)
    return render(request, 'shop_assistant/recommend.html', {
        'results': results, 'categories': categories, 'selected_category': category,
        'budget': budget, 'keyword': keyword, 'result_count': len(results), 'page': 'recommend',
    })


def product_detail(request, product_name):
    model = get_model()
    product_name = unquote(unquote(product_name))
    df = model.df[model.df['name'] == product_name]
    if df.empty:
        raise Http404("Product not found.")
    product = df.iloc[0].to_dict()
    product['ethical_probability'] = model.predict_ethical_rating(
        product['eco_score'], product['ethics_score'],
        product['carbon_footprint'], product['price'])
    product['brand_story'] = model.get_brand_story(product['brand'])
    similar_df = model.get_similar_products(product_name)
    similar = similar_df.to_dict('records') if not similar_df.empty else []
    return render(request, 'shop_assistant/product_detail.html', {
        'product': product, 'similar': similar, 'page': 'detail'
    })


@csrf_exempt
@require_http_methods(["GET", "POST"])
def api_recommend(request):
    model = get_model()
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
    else:
        data = request.GET
    category = data.get('category')
    budget   = data.get('budget')
    keyword  = data.get('keyword')
    top_n    = int(data.get('top_n', 6))
    results_df = model.recommend(
        category=category, budget=float(budget) if budget else None,
        keyword=keyword, top_n=top_n)
    products = []
    for _, row in results_df.iterrows():
        products.append({
            'name': row['name'], 'brand': row['brand'], 'category': row['category'],
            'eco_score': row['eco_score'], 'ethics_score': row['ethics_score'],
            'composite_score': row['composite_score'], 'price': row['price'],
            'sustainability_cert': row['sustainability_cert'], 'materials': row['materials'],
            'carbon_footprint': row['carbon_footprint'], 'description': row['description'],
            'ethical_probability': model.predict_ethical_rating(
                row['eco_score'], row['ethics_score'], row['carbon_footprint'], row['price']),
        })
    return JsonResponse({'status': 'success', 'count': len(products), 'products': products,
        'filters': {'category': category, 'budget': budget, 'keyword': keyword}})


def api_stats(request):
    return JsonResponse({'status': 'success', 'stats': get_model().get_stats()})


def api_categories(request):
    return JsonResponse({'status': 'success', 'categories': get_model().get_all_categories()})


def about(request):
    return render(request, 'shop_assistant/about.html', {'page': 'about'})


# ═══════════════════════════════════════════════════════════
# ECOMIND LLM — Our own trained neural network
# ═══════════════════════════════════════════════════════════

@csrf_exempt
@require_http_methods(["POST"])
def api_predict_ethical(request):
    """
    POST /api/predict/
    Uses OWN trained EcoMindNet to predict ethical scores.
    """
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    try:
        sys.path.insert(0, BASE_DIR)
        from ecomind_llm.predictor import get_predictor
        predictor = get_predictor()
        result = predictor.predict(data)
        result['status'] = 'ok'
        result['engine'] = 'ecomind_llm'
        return JsonResponse(result)
    except FileNotFoundError as e:
        return JsonResponse({"error": str(e)}, status=500)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


# ═══════════════════════════════════════════════════════════
# CHATBOT — 100% own pipeline, zero external API
#
# Flow:
#   1. chatbot.py  → parses user message (budget, category, filters)
#   2. model.py    → fetches matching products from database
#   3. EcoMindNet  → predicts ethical scores for each product
#   4. Build HTML  → rich product cards returned to user
# ═══════════════════════════════════════════════════════════

import importlib.util, pathlib
_cb_path = pathlib.Path(__file__).parent / "chatbot.py"
_cb_spec = importlib.util.spec_from_file_location("chatbot", _cb_path)
_cb_mod  = importlib.util.module_from_spec(_cb_spec)
_cb_spec.loader.exec_module(_cb_mod)

# Import individual functions from chatbot.py
_extract_budget_inr   = _cb_mod._extract_budget_inr
_extract_categories   = _cb_mod._extract_categories
_extract_ethical_filters = _cb_mod._extract_ethical_filters
_make_ethical_tags    = _cb_mod._make_ethical_tags
_build_product_link   = _cb_mod._build_product_link
_greeting_response    = _cb_mod._greeting_response
_help_response        = _cb_mod._help_response
_no_results_response  = _cb_mod._no_results_response


CARD_STYLES = (
    "<style>"
    ".eco-llm-card{background:#141f17;border:1px solid rgba(74,222,128,0.15);border-radius:14px;"
    "padding:1rem 1.1rem;margin-bottom:0.75rem;animation:fadeUp 0.4s ease both;}"
    ".elc-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.5rem;}"
    ".elc-name{font-family:'Playfair Display',serif;font-size:0.95rem;font-weight:700;"
    "color:#e8f5ee;text-decoration:none;}"
    ".elc-name:hover{color:#4ade80;}"
    ".elc-brand{font-size:0.75rem;color:#8aaa94;margin-top:2px;}"
    ".elc-score{font-family:'DM Mono',monospace;font-size:1.3rem;font-weight:600;color:#4ade80;line-height:1;}"
    ".elc-desc{font-size:0.78rem;color:#8aaa94;margin-bottom:0.6rem;line-height:1.5;}"
    ".elc-tags{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:0.65rem;}"
    ".elc-tag{padding:2px 8px;border-radius:5px;font-size:0.68rem;font-weight:700;letter-spacing:0.04em;}"
    ".elc-tag.cert{background:rgba(74,222,128,0.15);color:#4ade80;}"
    ".elc-tag.carbon{background:rgba(167,243,193,0.1);color:#86efac;}"
    ".elc-tag.eco{background:rgba(74,222,128,0.08);color:#a7f3c1;}"
    ".elc-tag.ethics{background:rgba(251,191,36,0.1);color:#fbbf24;}"
    ".elc-footer{display:flex;justify-content:space-between;align-items:center;"
    "padding-top:0.55rem;border-top:1px solid rgba(74,222,128,0.08);gap:0.5rem;}"
    ".elc-price{font-weight:700;color:#e8f5ee;font-size:0.9rem;}"
    ".elc-match{font-size:0.7rem;color:#4a6856;font-style:italic;flex:1;text-align:right;}"
    ".elc-summary{font-size:0.82rem;color:#4ade80;margin-top:0.75rem;padding-top:0.5rem;"
    "border-top:1px solid rgba(74,222,128,0.1);}"
    ".elc-llm-badge{display:inline-block;background:rgba(74,222,128,0.1);"
    "border:1px solid rgba(74,222,128,0.3);color:#4ade80;padding:3px 10px;"
    "border-radius:5px;font-size:0.68rem;font-weight:700;margin-bottom:0.75rem;}"
    ".elc-bar-wrap{background:rgba(255,255,255,0.05);border-radius:99px;height:4px;overflow:hidden;}"
    ".elc-bar{height:100%;border-radius:99px;background:linear-gradient(90deg,#2d6a4f,#4ade80);}"
    "</style>"
)


def _get_carbon_info(carbon):
    """Return color and label for carbon footprint value."""
    if carbon <= 0.5:
        return "#4ade80", "Ultra Low"
    elif carbon <= 1.5:
        return "#86efac", "Low"
    elif carbon <= 3.0:
        return "#fbbf24", "Moderate"
    else:
        return "#f87171", "High"


def _build_product_card(row, rank, llm_result=None):
    """
    Build a rich HTML card for one product.
    llm_result: dict from EcoMindNet prediction (optional, enhances scores)
    """
    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    medal  = medals[rank] if rank < len(medals) else str(rank + 1)

    # Use EcoMindNet predicted scores if available, else use dataset scores
    if llm_result:
        eco_score    = llm_result.get("eco_score",    row["eco_score"])
        ethics_score = llm_result.get("ethics_score", row["ethics_score"])
        carbon_label = llm_result.get("carbon_label", "")
        llm_tags     = llm_result.get("tags", [])
        confidence   = llm_result.get("confidence", 0)
        overall      = llm_result.get("overall_score", row["composite_score"])
    else:
        eco_score    = row["eco_score"]
        ethics_score = row["ethics_score"]
        carbon_label = ""
        llm_tags     = []
        confidence   = 0
        overall      = row["composite_score"]

    carbon       = row["carbon_footprint"]
    price_inr    = int(row["price"] * 83)
    carbon_color, carbon_text = _get_carbon_info(carbon)
    link         = _build_product_link(row["name"], row["brand"])

    # Build tags — from dataset + EcoMindNet predictions
    dataset_tags = _make_ethical_tags(row.to_dict())
    extra_tags   = ""
    for t in llm_tags:
        label = t.replace("_", " ").title()
        extra_tags += (
            '<span style="background:rgba(74,222,128,0.12);color:#a7f3c1;'
            'padding:2px 8px;border-radius:5px;font-size:0.68rem;font-weight:700;">'
            + label + " 🧠</span> "
        )

    tags_html = " ".join(dataset_tags) + " " + extra_tags

    # Score bar widths
    eco_w  = int(eco_score * 10)
    eth_w  = int(ethics_score * 10)
    ovr_w  = int(float(overall) * 10)

    # LLM confidence badge
    conf_badge = ""
    if confidence > 0:
        conf_badge = (
            '<span style="font-size:0.65rem;color:#4ade80;'
            'background:rgba(74,222,128,0.08);padding:1px 6px;border-radius:4px;">'
            "🧠 " + str(confidence) + "% confident</span>"
        )

    delay = rank * 80

    return (
        '<div class="eco-llm-card" style="animation-delay:' + str(delay) + 'ms">'

        # ── Header ──────────────────────────────────────────
        '<div class="elc-header">'
        '<div style="flex:1">'
        '<div style="display:flex;align-items:center;gap:0.4rem;margin-bottom:2px;">'
        '<span>' + medal + '</span>'
        '<a href="' + link + '" target="_blank" class="elc-name">' + str(row["name"]) + ' ↗</a>'
        '</div>'
        '<div class="elc-brand">by ' + str(row["brand"]) + ' · ' + str(row["category"]) + '</div>'
        '</div>'
        '<div style="text-align:right;flex-shrink:0;">'
        '<div class="elc-score">' + str(overall) + '</div>'
        '<div style="font-size:0.62rem;color:#4a6856;">AI Score</div>'
        '</div>'
        '</div>'

        # ── Description ──────────────────────────────────────
        '<p class="elc-desc">' + str(row["description"]) + '</p>'

        # ── Tags ─────────────────────────────────────────────
        '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:0.65rem;">'
        + tags_html +
        '</div>'

        # ── Score bars ────────────────────────────────────────
        '<div style="display:grid;grid-template-columns:56px 1fr 34px;gap:3px 6px;'
        'align-items:center;font-size:0.68rem;margin-bottom:0.65rem;">'

        '<span style="color:#4a6856;">Eco</span>'
        '<div class="elc-bar-wrap"><div class="elc-bar" style="width:' + str(eco_w) + '%"></div></div>'
        '<span style="color:#4ade80;font-family:DM Mono,monospace;">' + str(eco_score) + '</span>'

        '<span style="color:#4a6856;">Ethics</span>'
        '<div class="elc-bar-wrap"><div class="elc-bar" style="width:' + str(eth_w) + '%"></div></div>'
        '<span style="color:#4ade80;font-family:DM Mono,monospace;">' + str(ethics_score) + '</span>'

        '<span style="color:#4a6856;">Overall</span>'
        '<div class="elc-bar-wrap"><div class="elc-bar" style="width:' + str(ovr_w) + '%;'
        'background:linear-gradient(90deg,#16a34a,#86efac)"></div></div>'
        '<span style="color:#86efac;font-family:DM Mono,monospace;">' + str(overall) + '</span>'

        '</div>'

        # ── Footer ────────────────────────────────────────────
        '<div class="elc-footer">'
        '<div style="display:flex;align-items:center;gap:0.4rem;">'
        '<span style="color:' + carbon_color + ';font-size:0.72rem;">💨 ' + str(carbon) + 'kg CO₂</span>'
        '<span style="background:rgba(74,222,128,0.08);color:' + carbon_color + ';'
        'padding:1px 6px;border-radius:4px;font-size:0.65rem;font-weight:600;">' + carbon_text + '</span>'
        '</div>'
        '<div style="display:flex;align-items:center;gap:0.5rem;">'
        + conf_badge +
        '<span class="elc-price">₹' + str(price_inr) + '</span>'
        '</div>'
        '</div>'

        '</div>'
    )


def _score_row_against_filters(row, ethical_filters):
    """Re-rank products by how many ethical filters they match."""
    score   = 0
    cert    = str(row.get("sustainability_cert", "")).lower()
    mats    = str(row.get("materials", "")).lower()
    desc    = str(row.get("description", "")).lower()
    combined = cert + " " + mats + " " + desc

    filter_map = {
        "organic":     ["organic", "usda", "cosmos", "gots"],
        "fair_trade":  ["fair trade", "fairtrade"],
        "b_corp":      ["b-corp", "b corp"],
        "vegan":       ["vegan", "peta", "plant"],
        "recycled":    ["recycled", "upcycled"],
        "natural":     ["natural", "plant", "botanical", "herb", "organic"],
        "zero_waste":  ["zero", "package", "compostable", "refill"],
        "sustainable": ["sustainable", "eco", "certified"],
        "certified":   ["gots", "oeko", "bluesign", "fsc", "certified",
                        "fair trade", "b-corp", "usda", "cosmos"],
    }

    for f in ethical_filters:
        for kw in filter_map.get(f, []):
            if kw in combined:
                score += 1
                break
        if f == "low_carbon" and float(row.get("carbon_footprint", 99)) <= 1.5:
            score += 2

    return score


@csrf_exempt
@require_http_methods(["POST"])
def api_chat(request):
    """
    100% own pipeline chatbot — no external API.

    Step 1: Parse user message  → budget, category, ethical filters
    Step 2: Filter products     → from 203-product database
    Step 3: EcoMindNet scores   → predict ethical scores for results
    Step 4: Build HTML cards    → return to user
    """
    try:
        data    = json.loads(request.body)
        message = data.get("message", "").strip()
    except (json.JSONDecodeError, KeyError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if not message:
        return JsonResponse({"error": "Empty message"}, status=400)

    text_lower = message.lower()

    # ── Greeting ──────────────────────────────────────────────
    greet_words = ["hello", "hi", "hey", "good morning", "good evening", "howdy"]
    if any(text_lower.startswith(g) for g in greet_words) and len(message.split()) <= 4:
        return JsonResponse({"html": _greeting_response(), "status": "ok", "engine": "ecomind"})

    # ── Help ──────────────────────────────────────────────────
    help_words = ["help", "what can you do", "how does", "capabilities"]
    if any(h in text_lower for h in help_words):
        return JsonResponse({"html": _help_response(), "status": "ok", "engine": "ecomind"})

    # ── Intent Guard — filters all non-shopping queries ───────

    # 1. Vulgar / abusive language
    vulgar_words = [
        "fuck", "shit", "bitch", "asshole", "bastard", "damn", "crap",
        "dick", "pussy", "cock", "ass", "idiot", "stupid", "dumb", "moron",
        "hell", "sex", "porn", "nude", "naked", "sexy", "horny", "nsfw",
    ]
    if any(v in text_lower for v in vulgar_words):
        html = (
            '<div style="background:#1a1a2e;border:1px solid rgba(248,113,113,0.25);'
            'border-radius:14px;padding:1.2rem 1.3rem;">'
            '<div style="font-size:1.1rem;margin-bottom:0.5rem;">🚫</div>'
            '<p style="color:#f87171;font-weight:700;margin-bottom:0.4rem;font-size:0.9rem;">'
            'That kind of language isn\'t something I can engage with.</p>'
            '<p style="color:#8aaa94;font-size:0.8rem;line-height:1.6;">'
            'I\'m EcoMind — your ethical shopping assistant. '
            'I\'m here to help you find sustainable, eco-friendly products '
            'that are good for you and the planet. 🌿<br><br>'
            'Try asking something like:<br>'
            '<em style="color:#4ade80;">"Show me organic food under ₹500"</em> or '
            '<em style="color:#4ade80;">"Fair trade clothing with low carbon footprint"</em>'
            '</p>'
            '</div>'
        )
        return JsonResponse({"html": CARD_STYLES + html, "status": "ok", "engine": "ecomind"})

    # 2. Political questions
    political_words = [
        "politics", "political", "election", "vote", "voting", "government",
        "minister", "prime minister", "president", "mp", "mla", "party",
        "bjp", "congress", "aap", "modi", "rahul", "kejriwal", "trump",
        "biden", "democrat", "republican", "labour", "tory", "parliament",
        "constitution", "war", "military", "army", "protest", "revolution",
        "abortion", "gun", "religion", "hindu", "muslim", "christian",
        "temple", "mosque", "church", "caste", "reservation",
    ]
    if any(p in text_lower for p in political_words):
        html = (
            '<div style="background:#141f17;border:1px solid rgba(251,191,36,0.2);'
            'border-radius:14px;padding:1.2rem 1.3rem;">'
            '<div style="font-size:1.1rem;margin-bottom:0.5rem;">🌍</div>'
            '<p style="color:#fbbf24;font-weight:700;margin-bottom:0.4rem;font-size:0.9rem;">'
            'That\'s outside my area of expertise.</p>'
            '<p style="color:#8aaa94;font-size:0.8rem;line-height:1.6;">'
            'I respect that political and social topics are important — '
            'but I\'m EcoMind, an ethical shopping assistant, and I\'m not the right '
            'source for political opinions or discussions.<br><br>'
            'What I <em>can</em> help with is finding products that align with your '
            'values — ethically sourced, eco-certified, low carbon footprint. 🌿<br><br>'
            'Try: <em style="color:#4ade80;">"Show me certified fair trade products"</em>'
            '</p>'
            '</div>'
        )
        return JsonResponse({"html": CARD_STYLES + html, "status": "ok", "engine": "ecomind"})

    # 3. Gender / discrimination based questions
    gender_words = [
        "gender", "feminist", "feminism", "sexist", "sexism", "misogyn",
        "patriarchy", "lgbtq", "gay", "lesbian", "transgender", "queer",
        "racist", "racism", "sexist", "discrimination", "inequality",
    ]
    if any(g in text_lower for g in gender_words):
        html = (
            '<div style="background:#141f17;border:1px solid rgba(167,243,193,0.15);'
            'border-radius:14px;padding:1.2rem 1.3rem;">'
            '<div style="font-size:1.1rem;margin-bottom:0.5rem;">🤝</div>'
            '<p style="color:#86efac;font-weight:700;margin-bottom:0.4rem;font-size:0.9rem;">'
            'I treat every person with equal respect.</p>'
            '<p style="color:#8aaa94;font-size:0.8rem;line-height:1.6;">'
            'Social equity topics deserve thoughtful, expert conversation — '
            'which is beyond what I\'m built for as a shopping assistant.<br><br>'
            'I do believe in ethical business practices, including fair wages '
            'and equal treatment of workers in supply chains — and that\'s '
            'reflected in the products I recommend. 🌿<br><br>'
            'Try: <em style="color:#4ade80;">"Show me fair trade certified products"</em>'
            '</p>'
            '</div>'
        )
        return JsonResponse({"html": CARD_STYLES + html, "status": "ok", "engine": "ecomind"})

    # 4. Off-topic — non-shopping queries
    off_topic_signals = [
        # tech/general knowledge
        "weather", "temperature", "news", "sports", "cricket", "football",
        "movie", "song", "music", "recipe", "cook", "joke", "poem", "story",
        "capital of", "who is", "what is the", "how to make", "explain",
        "tell me about", "define", "meaning of", "translate", "calculate",
        "math", "solve", "equation", "history of", "write a", "code",
        "program", "python", "java", "javascript", "chatgpt", "ai model",
        # vehicles / non-products
        "car", "bike", "petrol", "diesel", "motorcycle", "vehicle",
        "flight", "hotel", "travel", "trip", "tour",
        # health / medical
        "doctor", "medicine", "hospital", "disease", "symptom", "cure",
        "covid", "vaccine", "tablet", "prescription",
        # finance
        "stock", "crypto", "bitcoin", "invest", "share market", "trading",
        "loan", "emi", "bank", "insurance",
    ]
    if any(o in text_lower for o in off_topic_signals):
        # Make sure it's truly off-topic and not accidentally matching a product word
        shopping_signals = [
            "product", "buy", "shop", "organic", "eco", "sustainable",
            "ethical", "carbon", "certified", "natural", "vegan", "fair trade",
            "clothing", "food", "kitchen", "personal care", "footwear",
            "under", "below", "rupees", "price", "budget",
        ]
        is_shopping = any(s in text_lower for s in shopping_signals)
        if not is_shopping:
            html = (
                '<div style="background:#141f17;border:1px solid rgba(74,222,128,0.12);'
                'border-radius:14px;padding:1.2rem 1.3rem;">'
                '<div style="font-size:1.1rem;margin-bottom:0.5rem;">🌿</div>'
                '<p style="color:#e8f5ee;font-weight:700;margin-bottom:0.4rem;font-size:0.9rem;">'
                'Sorry, I couldn\'t find products related to that.</p>'
                '<p style="color:#8aaa94;font-size:0.8rem;line-height:1.6;">'
                'I\'m EcoMind — I specialise exclusively in ethical and eco-friendly '
                'product recommendations. I\'m not able to help with general questions '
                'outside of sustainable shopping.<br><br>'
                'Here\'s what I\'m great at:</p>'
                '<ul style="color:#8aaa94;font-size:0.78rem;line-height:1.8;'
                'padding-left:1.2rem;margin:0.4rem 0 0.75rem;">'
                '<li>Finding <strong style="color:#4ade80;">organic</strong> and '
                '<strong style="color:#4ade80;">certified</strong> products</li>'
                '<li>Filtering by <strong style="color:#4ade80;">budget in ₹</strong></li>'
                '<li>Recommending <strong style="color:#4ade80;">low carbon footprint</strong> items</li>'
                '<li>Showing <strong style="color:#4ade80;">fair trade</strong> and '
                '<strong style="color:#4ade80;">vegan</strong> options</li>'
                '</ul>'
                '<p style="color:#4a6856;font-size:0.75rem;">Try: '
                '<em style="color:#4ade80;">"Organic food under ₹500"</em> · '
                '<em style="color:#4ade80;">"Fair trade clothing"</em> · '
                '<em style="color:#4ade80;">"Zero waste kitchen products"</em>'
                '</p>'
                '</div>'
            )
            return JsonResponse({"html": CARD_STYLES + html, "status": "ok", "engine": "ecomind"})

    # ── Step 1: Parse intent ──────────────────────────────────
    budget_usd   = _extract_budget_inr(message)
    categories   = _extract_categories(message)
    eth_filters  = _extract_ethical_filters(message)

    # ── Step 2: Filter products from database ─────────────────
    ai_model = get_model()
    df = ai_model.df.copy()

    # Category filter
    if categories:
        df = df[df["category"].isin(categories)]

    # Budget filter — gracefully relax if too tight
    budget_relaxed = False
    if budget_usd:
        strict = df[df["price"] <= budget_usd]
        if strict.empty:
            budget_relaxed = True   # show cheapest available
        else:
            df = strict

    # Keyword filter
    import re
    stop = {"i","want","need","looking","find","show","get","give","me","some","a","an",
            "the","and","or","with","for","that","which","should","is","are","be","have",
            "has","very","quite","really","less","more","most","best","good","please",
            "below","under","within","budget","price","cost","rupees","rs","inr","₹"}
    words   = re.findall(r'[a-z]+', text_lower)
    kw_list = [w for w in words if w not in stop and len(w) > 3]
    keyword = " ".join(kw_list[:3]) if kw_list else None

    if keyword and not df.empty:
        kw = keyword.lower()
        kw_mask = (
            df["name"].str.lower().str.contains(kw, na=False) |
            df["brand"].str.lower().str.contains(kw, na=False) |
            df["materials"].str.lower().str.contains(kw, na=False) |
            df["description"].str.lower().str.contains(kw, na=False) |
            df["sustainability_cert"].str.lower().str.contains(kw, na=False) |
            df["category"].str.lower().str.contains(kw, na=False)
        )
        if kw_mask.any():
            df = df[kw_mask]

    # Ethical filter re-ranking
    if eth_filters and not df.empty:
        df = df.copy()
        df["_eth_score"] = df.apply(
            lambda r: _score_row_against_filters(r.to_dict(), eth_filters), axis=1
        )
        # Hard filter for low carbon
        if "low_carbon" in eth_filters:
            lc = df[df["carbon_footprint"] <= 2.0]
            if not lc.empty:
                df = lc
        # Hard filter for organic/certified
        if "organic" in eth_filters or "certified" in eth_filters:
            org_certs = ["organic", "gots", "cosmos", "usda", "certified", "non-gmo"]
            org_mask  = df["sustainability_cert"].str.lower().apply(
                lambda c: any(kw in c for kw in org_certs)
            )
            if org_mask.any():
                df = df[org_mask]

        df = df.sort_values(["_eth_score", "composite_score"], ascending=[False, False])
    else:
        df = df.sort_values("composite_score", ascending=False)

    # Top 5 results
    results = df.head(5)

    if results.empty:
        return JsonResponse({
            "html": _no_results_response(message, budget_usd, categories),
            "status": "ok", "engine": "ecomind"
        })

    # ── Step 3: EcoMindNet — predict ethical scores ───────────
    llm_predictions = {}
    llm_ready = os.path.exists(
        os.path.join(BASE_DIR, "ecomind_llm", "saved_model", "ecomind_net.pkl")
    )
    if llm_ready:
        try:
            from ecomind_llm.predictor import get_predictor
            predictor = get_predictor()
            product_list = [row.to_dict() for _, row in results.iterrows()]
            batch = predictor.predict_batch(product_list)
            llm_predictions = {i: batch[i] for i in range(len(batch))}
        except Exception:
            llm_predictions = {}

    # ── Step 4: Build HTML response ───────────────────────────
    budget_inr = int(budget_usd * 83) if budget_usd else None
    cat_str    = ", ".join(categories) if categories else "all categories"
    filter_str = ", ".join(eth_filters).replace("_", " ") if eth_filters else ""

    # Summary header
    if budget_relaxed:
        min_price = int(results["price"].min() * 83)
        summary = (
            '<p style="font-size:0.8rem;color:#fbbf24;margin-bottom:0.5rem;">'
            '⚠️ No products under ₹' + str(budget_inr) + '. '
            'Showing closest matches from ₹' + str(min_price) + '.</p>'
            '<p style="font-size:0.78rem;color:#8aaa94;margin-bottom:0.75rem;">'
            'Found <strong style="color:#4ade80">' + str(len(results)) + '</strong>'
            ' products in <em>' + cat_str + '</em>'
        )
    else:
        summary = (
            '<p style="font-size:0.78rem;color:#8aaa94;margin-bottom:0.75rem;">'
            'Found <strong style="color:#4ade80">' + str(len(results)) + '</strong>'
            ' ethical products in <em>' + cat_str + '</em>'
            + (' under ₹' + str(budget_inr) if budget_inr else '')
        )
    if filter_str:
        summary += ' · <em>' + filter_str + '</em>'
    summary += '</p>'

    # LLM badge
    llm_badge = ""
    if llm_ready and llm_predictions:
        llm_badge = (
            '<span class="elc-llm-badge">'
            '🧠 Scores predicted by EcoMind Neural Network'
            '</span><br>'
        )

    # Build all product cards
    cards = ""
    for i, (_, row) in enumerate(results.iterrows()):
        llm_result = llm_predictions.get(i, None)
        cards += _build_product_card(row, i, llm_result)

    # Suggestion chips
    suggestions = [
        "Organic food under ₹500",
        "Fair trade clothing",
        "Low carbon kitchen products",
        "Vegan personal care",
    ]
    chips = "".join([
        '<button onclick="setInput(this.textContent)" '
        'style="background:rgba(74,222,128,0.07);border:1px solid rgba(74,222,128,0.18);'
        'color:#8aaa94;padding:3px 10px;border-radius:99px;font-size:0.7rem;cursor:pointer;'
        'font-family:inherit;transition:all 0.2s;" '
        'onmouseover="this.style.color=\'#4ade80\'" '
        'onmouseout="this.style.color=\'#8aaa94\'">'
        + s + '</button>'
        for s in suggestions
    ])
    suggest_html = (
        '<div style="margin-top:0.75rem;padding-top:0.75rem;'
        'border-top:1px solid rgba(74,222,128,0.08);">'
        '<div style="font-size:0.68rem;color:#4a6856;margin-bottom:0.4rem;">💡 Try asking:</div>'
        '<div style="display:flex;flex-wrap:wrap;gap:0.4rem;">' + chips + '</div>'
        '</div>'
    )

    full_html = CARD_STYLES + llm_badge + summary + cards + suggest_html

    return JsonResponse({
        "html": full_html,
        "status": "ok",
        "engine": "ecomind_llm" if llm_ready else "ecomind_rule_based"
    })


# ═══════════════════════════════════════════════════════════
# PRODUCT SCANNER — Scan any product via EcoMindNet
# ═══════════════════════════════════════════════════════════

def scan(request):
    """Render the product scanner page."""
    return render(request, 'shop_assistant/scan.html', {'page': 'scan'})


@csrf_exempt
@require_http_methods(["POST"])
def api_scan(request):
    """
    POST /api/scan/
    Accepts product details from user, runs through EcoMindNet,
    returns eco_score, ethics_score, carbon_level, tags, overall score.
    """
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    name        = data.get("name", "").strip()
    brand       = data.get("brand", "").strip()
    category    = data.get("category", "General").strip()
    description = data.get("description", "").strip()
    materials   = data.get("materials", "").strip()
    cert        = data.get("cert", "").strip()
    price       = data.get("price", "")

    if not name:
        return JsonResponse({"error": "Product name is required."}, status=400)

    # Build enriched description for EcoMindNet
    enriched = f"{name} {brand} {category} {description} {materials} {cert}".strip()

    try:
        sys.path.insert(0, BASE_DIR)
        from ecomind_llm.predictor import get_predictor
        predictor = get_predictor()

        payload = {
            "name": name,
            "brand": brand,
            "category": category,
            "description": enriched,
            "materials": materials,
            "sustainability_cert": cert,
        }
        result = predictor.predict(payload)

        eco     = round(float(result.get("eco_score", 5.0)), 1)
        ethics  = round(float(result.get("ethics_score", 5.0)), 1)
        carbon  = result.get("carbon_level", "moderate")
        tags    = result.get("tags", [])
        conf    = round(float(result.get("confidence", 50.0)), 1)  # predictor already returns % value

        # Carbon numeric mapping
        carbon_map = {"ultra_low": 0, "low": 1, "moderate": 2, "high": 3}
        carbon_val = carbon_map.get(carbon, 2)

        # Overall score
        overall = round(
            (eco * 0.45) + (ethics * 0.45)
            + max(3 - carbon_val, 0) * 0.5
            + len(tags) * 0.1,
            1
        )
        overall = min(overall, 10.0)

        # Verdict
        if overall >= 8.5:
            verdict = "Excellent"
            verdict_color = "#4ade80"
            verdict_icon = "🌟"
            verdict_msg = "This product meets the highest ethical and environmental standards."
        elif overall >= 7.0:
            verdict = "Good"
            verdict_color = "#86efac"
            verdict_icon = "✅"
            verdict_msg = "A solid ethical choice with good eco credentials."
        elif overall >= 5.5:
            verdict = "Average"
            verdict_color = "#fbbf24"
            verdict_icon = "⚠️"
            verdict_msg = "Some ethical aspects are present but there is room to improve."
        elif overall >= 4.0:
            verdict = "Below Average"
            verdict_color = "#f97316"
            verdict_icon = "📉"
            verdict_msg = "This product has limited ethical or eco-friendly credentials."
        else:
            verdict = "Poor"
            verdict_color = "#f87171"
            verdict_icon = "❌"
            verdict_msg = "This product scores low on ethical and environmental metrics."

        # Carbon label
        carbon_labels = {
            "ultra_low": {"label": "Ultra Low", "color": "#4ade80", "kg": "< 0.5 kg CO₂"},
            "low":       {"label": "Low",        "color": "#86efac", "kg": "0.5 – 1.5 kg CO₂"},
            "moderate":  {"label": "Moderate",   "color": "#fbbf24", "kg": "1.5 – 4 kg CO₂"},
            "high":      {"label": "High",        "color": "#f87171", "kg": "> 4 kg CO₂"},
        }
        carbon_info = carbon_labels.get(carbon, carbon_labels["moderate"])

        return JsonResponse({
            "status": "ok",
            "engine": "ecomind_llm",
            "product": {
                "name": name,
                "brand": brand,
                "category": category,
            },
            "scores": {
                "eco_score": eco,
                "ethics_score": ethics,
                "overall_score": overall,
                "confidence": conf,
            },
            "carbon": {
                "level": carbon,
                "label": carbon_info["label"],
                "color": carbon_info["color"],
                "estimate": carbon_info["kg"],
            },
            "tags": tags,
            "verdict": {
                "label": verdict,
                "color": verdict_color,
                "icon": verdict_icon,
                "message": verdict_msg,
            }
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


# ═══════════════════════════════════════════════════════════
# BARCODE LOOKUP — Fetches product info from Open Food Facts
# ═══════════════════════════════════════════════════════════

import urllib.request
import urllib.error

@csrf_exempt
@require_http_methods(["POST"])
def api_barcode_lookup(request):
    """
    POST /api/barcode-lookup/
    Looks up barcode on Open Food Facts / Open Beauty Facts / Open Products Facts
    Returns product name, brand, category, ingredients, certifications.
    """
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    barcode = str(data.get("barcode", "")).strip()
    if not barcode:
        return JsonResponse({"error": "Barcode is required."}, status=400)

    # Try multiple Open Facts databases in order
    databases = [
        ("food",    f"https://world.openfoodfacts.org/api/v2/product/{barcode}.json"),
        ("beauty",  f"https://world.openbeautyfacts.org/api/v2/product/{barcode}.json"),
        ("product", f"https://world.openproductsfacts.org/api/v2/product/{barcode}.json"),
    ]

    product_data = None
    source_db    = None

    for db_name, url in databases:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "EcoMindAI/1.0 (deepak@ecomind.app)"}
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                result = json.loads(resp.read().decode())
                if result.get("status") == 1 and result.get("product"):
                    product_data = result["product"]
                    source_db = db_name
                    break
        except Exception:
            continue

    if not product_data:
        return JsonResponse({
            "found": False,
            "message": "Product not found in our databases. Please fill in the details manually."
        })

    p = product_data

    # Extract fields
    name        = p.get("product_name") or p.get("product_name_en") or ""
    brand       = p.get("brands", "").split(",")[0].strip()
    ingredients = p.get("ingredients_text") or p.get("ingredients_text_en") or ""
    labels      = p.get("labels", "") or p.get("labels_tags", "")
    categories  = p.get("categories", "") or p.get("categories_en", "")
    quantity    = p.get("quantity", "")
    countries   = p.get("countries", "")
    image_url   = p.get("image_front_small_url") or p.get("image_url") or ""

    # Map category to our category list
    cat_lower = categories.lower() if categories else ""
    if any(w in cat_lower for w in ["clothing","apparel","fashion","textile","wear"]):
        category = "Clothing"
    elif any(w in cat_lower for w in ["shoe","footwear","boot","sandal"]):
        category = "Footwear"
    elif any(w in cat_lower for w in ["food","beverage","drink","snack","cereal","dairy","bread","fruit","vegetable","meat","fish"]):
        category = "Food"
    elif any(w in cat_lower for w in ["beauty","cosmetic","shampoo","conditioner","cream","lotion","serum","makeup","skincare","haircare","soap","perfume","deodorant"]):
        category = "Personal Care"
    elif any(w in cat_lower for w in ["kitchen","cookware","utensil","container"]):
        category = "Kitchen"
    elif any(w in cat_lower for w in ["home","household","furniture","linen","candle"]):
        category = "Home"
    elif any(w in cat_lower for w in ["sport","gym","yoga","fitness","athletic"]):
        category = "Sportswear"
    elif any(w in cat_lower for w in ["electronic","gadget","tech","device","phone","laptop","charger"]):
        category = "Electronics"
    elif any(w in cat_lower for w in ["baby","infant","child","toddler","diaper","nappy"]):
        category = "Baby"
    elif source_db == "beauty":
        category = "Personal Care"
    elif source_db == "food":
        category = "Food"
    else:
        category = "General"

    # Parse certifications from labels
    cert_map = {
        "organic":     "Organic",
        "fair-trade":  "Fair Trade",
        "fairtrade":   "Fair Trade",
        "b-corp":      "B-Corp",
        "vegan":       "Vegan",
        "vegetarian":  "Vegetarian",
        "gots":        "GOTS",
        "oeko-tex":    "OEKO-TEX",
        "cosmos":      "COSMOS",
        "rainforest":  "Rainforest Alliance",
        "halal":       "Halal",
        "kosher":      "Kosher",
        "cruelty-free":"Cruelty Free",
        "gluten-free": "Gluten Free",
        "non-gmo":     "Non-GMO",
        "fsc":         "FSC",
        "recyclable":  "Recyclable",
        "recycled":    "Recycled",
    }

    labels_str = labels if isinstance(labels, str) else " ".join(labels)
    labels_lower = labels_str.lower()
    found_certs = [v for k, v in cert_map.items() if k in labels_lower]
    cert_string = ", ".join(dict.fromkeys(found_certs))  # deduplicate

    # Build description from available data
    description_parts = []
    if ingredients:
        description_parts.append(f"Ingredients: {ingredients[:300]}")
    if labels_str:
        description_parts.append(f"Labels: {labels_str[:200]}")
    if quantity:
        description_parts.append(f"Quantity: {quantity}")
    description = ". ".join(description_parts)

    return JsonResponse({
        "found":       True,
        "source":      source_db,
        "barcode":     barcode,
        "product": {
            "name":        name,
            "brand":       brand,
            "category":    category,
            "materials":   ingredients[:300] if ingredients else "",
            "cert":        cert_string,
            "description": description[:500],
            "image_url":   image_url,
            "quantity":    quantity,
            "countries":   countries,
            "labels_raw":  labels_str[:300],
        }
    })


# ═══════════════════════════════════════════════════════════
# AUTH — Email OTP Login (no password)
# ═══════════════════════════════════════════════════════════

from django.core.mail import send_mail
from django.template.loader import render_to_string

def _get_logged_in_user(request):
    """Return EcoUser if logged in via session, else None."""
    from shop_assistant.models import EcoUser
    uid = request.session.get('eco_user_id')
    if uid:
        try:
            return EcoUser.objects.get(id=uid)
        except EcoUser.DoesNotExist:
            pass
    return None


def login_page(request):
    return render(request, 'shop_assistant/login.html', {'page': 'login'})


@csrf_exempt
@require_http_methods(["POST"])
def api_send_otp(request):
    """POST /api/auth/send-otp/  { email }"""
    try:
        from shop_assistant.models import OTPCode
        data  = json.loads(request.body)
        email = data.get('email', '').strip().lower()
        if not email or '@' not in email:
            return JsonResponse({'error': 'Enter a valid email address.'}, status=400)

        otp = OTPCode.generate(email)

        # Always print to terminal — works even without email config
        print(f'\n{"="*45}')
        print(f'[EcoMind OTP]  Email : {email}')
        print(f'[EcoMind OTP]  Code  : {otp.code}')
        print(f'{"="*45}\n')

        # Send email — read credentials directly from env
        import os as _os
        email_user = _os.environ.get('EMAIL_HOST_USER', '').strip()
        email_pass = _os.environ.get('EMAIL_HOST_PASSWORD', '').strip()

        print(f'[EcoMind OTP] EMAIL_HOST_USER set: {bool(email_user)}')
        print(f'[EcoMind OTP] EMAIL_HOST_PASSWORD set: {bool(email_pass)}')

        email_sent = False
        email_error = ''

        if email_user and email_pass:
            try:
                from django.core.mail import get_connection, EmailMessage
                # Build connection manually to ensure fresh credentials
                connection = get_connection(
                    backend='django.core.mail.backends.smtp.EmailBackend',
                    host='smtp.gmail.com',
                    port=587,
                    username=email_user,
                    password=email_pass,
                    use_tls=True,
                    fail_silently=False,
                )
                msg_obj = EmailMessage(
                    subject='Your EcoMind Login Code',
                    body=(
                        f'Your EcoMind AI login code is:\n\n'
                        f'  {otp.code}\n\n'
                        f'This code expires in 10 minutes.\n'
                        f'If you did not request this, ignore this email.'
                    ),
                    from_email=email_user,
                    to=[email],
                    connection=connection,
                )
                msg_obj.send(fail_silently=False)
                email_sent = True
                print(f'[EcoMind OTP] ✅ Email sent successfully to {email}')
            except Exception as mail_err:
                email_error = str(mail_err)
                print(f'[EcoMind OTP] ❌ Email send failed: {mail_err}')
        else:
            email_error = 'EMAIL_HOST_USER or EMAIL_HOST_PASSWORD not set in environment'
            print(f'[EcoMind OTP] ⚠️  {email_error}')

        msg = f'OTP sent to {email}' if email_sent else f'OTP generated (check server terminal)'
        return JsonResponse({
            'status': 'sent',
            'message': msg,
            'dev_mode': not email_sent,
            'email_error': email_error if not email_sent else '',
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return JsonResponse({'error': f'Server error: {str(e)} — did you run migrations?'}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def api_verify_otp(request):
    """POST /api/auth/verify-otp/  { email, code }"""
    try:
        from shop_assistant.models import EcoUser, OTPCode
        try:
            data  = json.loads(request.body)
            email = data.get('email', '').strip().lower()
            code  = data.get('code', '').strip()
        except Exception:
            return JsonResponse({'error': 'Invalid request'}, status=400)

        try:
            otp = OTPCode.objects.filter(email=email, code=code, used=False).latest('created_at')
        except OTPCode.DoesNotExist:
            return JsonResponse({'error': 'Invalid or expired code. Try again.'}, status=400)

        if not otp.is_valid():
            return JsonResponse({'error': 'Code expired. Please request a new one.'}, status=400)

        otp.used = True
        otp.save()

        # Get or create user
        from django.utils import timezone
        user, created = EcoUser.objects.get_or_create(email=email)
        user.last_login = timezone.now()
        user.save()

        # Store in session
        request.session['eco_user_id'] = user.id
        request.session['eco_user_email'] = user.email

        return JsonResponse({
            'status': 'ok',
            'user': {'email': user.email, 'name': user.name, 'id': user.id},
            'created': created,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return JsonResponse({'error': f'Server error: {str(e)}. Run: python manage.py migrate'}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def api_logout(request):
    request.session.flush()
    return JsonResponse({'status': 'ok'})


def api_auth_status(request):
    """GET /api/auth/status/  — check if logged in"""
    user = _get_logged_in_user(request)
    if user:
        return JsonResponse({'logged_in': True, 'email': user.email, 'name': user.name})
    return JsonResponse({'logged_in': False})


# ═══════════════════════════════════════════════════════════
# BARCODE PROXY — browser calls this, Django calls Open Food Facts
# Avoids all browser CORS/fetch issues
# ═══════════════════════════════════════════════════════════
def api_barcode_lookup(request):
    """
    GET /api/barcode/?code=8901030857973
    Django proxies the request to multiple product databases.
    Returns unified product JSON or {'found': False}.
    """
    import urllib.request
    import urllib.error

    barcode = request.GET.get('code', '').strip()
    if not barcode:
        return JsonResponse({'found': False, 'error': 'No barcode provided'})

    endpoints = [
        f'https://world.openfoodfacts.org/api/v0/product/{barcode}.json',
        f'https://in.openfoodfacts.org/api/v0/product/{barcode}.json',
        f'https://world.openbeautyfacts.org/api/v0/product/{barcode}.json',
        f'https://world.openproductsfacts.org/api/v0/product/{barcode}.json',
    ]

    headers = {
        'User-Agent': 'EcoMindAI/1.0 (university project; contact: ecomind@lpu.in)',
        'Accept': 'application/json',
    }

    for url in endpoints:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as resp:
                import json as _json
                data = _json.loads(resp.read().decode('utf-8'))
                p = data.get('product', {})
                if not p:
                    continue

                name = (p.get('product_name') or p.get('product_name_en') or
                        p.get('product_name_hi') or p.get('generic_name') or
                        p.get('abbreviated_product_name') or '').strip()
                if not name:
                    continue

                brand       = (p.get('brands') or '').split(',')[0].strip()
                ingredients = (p.get('ingredients_text') or p.get('ingredients_text_en') or '')[:400]
                labels      = ', '.join(
                    l.replace('en:', '').replace('fr:', '').replace('-', ' ')
                    for l in (p.get('labels_tags') or [])
                )
                packaging   = ', '.join(
                    x.replace('en:', '').replace('fr:', '').replace('-', ' ')
                    for x in (p.get('packaging_tags') or [])
                )
                quantity    = p.get('quantity') or ''
                countries   = ', '.join(
                    c.replace('en:', '') for c in (p.get('countries_tags') or [])
                )

                cats = ' '.join(p.get('categories_tags') or []) + ' ' + (p.get('categories') or '')
                cats = cats.lower()
                category = 'Food'
                if any(w in cats for w in ['shirt','cloth','apparel','jacket','dress','wear','jean']):
                    category = 'Clothing'
                elif any(w in cats for w in ['shoe','boot','footwear','sandal','sneaker']):
                    category = 'Footwear'
                elif any(w in cats for w in ['cosmetic','beauty','shampoo','lotion','cream','soap','hair','skin','perfume']):
                    category = 'Personal Care'
                elif any(w in cats for w in ['baby','infant','diaper']):
                    category = 'Baby'
                elif any(w in cats for w in ['kitchen','cleaning','detergent','household']):
                    category = 'Kitchen'

                desc_parts = []
                if ingredients: desc_parts.append(f'Ingredients: {ingredients}')
                if packaging:   desc_parts.append(f'Packaging: {packaging}')
                if quantity:    desc_parts.append(f'Quantity: {quantity}')
                if countries:   desc_parts.append(f'Made in: {countries}')

                return JsonResponse({
                    'found':       True,
                    'source':      url.split('/')[2],
                    'name':        name,
                    'brand':       brand,
                    'category':    category,
                    'materials':   ingredients,
                    'cert':        labels,
                    'description': '. '.join(desc_parts),
                    'quantity':    quantity,
                })
        except Exception as e:
            continue

    # UPCitemdb fallback (no auth needed for trial)
    try:
        upc_url = f'https://api.upcitemdb.com/prod/trial/lookup?upc={barcode}'
        req = urllib.request.Request(upc_url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            import json as _json
            data = _json.loads(resp.read().decode('utf-8'))
            if data.get('code') == 'OK' and data.get('items'):
                item = data['items'][0]
                name = (item.get('title') or '').strip()
                if name:
                    return JsonResponse({
                        'found':       True,
                        'source':      'upcitemdb.com',
                        'name':        name,
                        'brand':       item.get('brand') or '',
                        'category':    'General',
                        'materials':   item.get('ingredients') or '',
                        'cert':        '',
                        'description': (item.get('description') or '')[:300],
                        'quantity':    '',
                    })
    except Exception:
        pass

    return JsonResponse({'found': False, 'barcode': barcode})


# ═══════════════════════════════════════════════════════════
# SAVE PRODUCT — User saves a scanned product to their account
# ═══════════════════════════════════════════════════════════

@csrf_exempt
@require_http_methods(["POST"])
def api_save_product(request):
    """
    POST /api/save-product/
    Saves scanned product to UserSavedProduct AND adds it to
    the live ML model DataFrame so it affects future recommendations.
    """
    user = _get_logged_in_user(request)
    if not user:
        return JsonResponse({'error': 'Login required to save products.'}, status=401)

    try:
        data = json.loads(request.body)
    except Exception:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    from shop_assistant.models import UserSavedProduct
    from ai_model.model import add_user_product

    # Save to database
    product = UserSavedProduct.objects.create(
        user         = user,
        name         = data.get('name', '')[:200],
        brand        = data.get('brand', '')[:100],
        category     = data.get('category', 'General')[:100],
        description  = data.get('description', ''),
        materials    = data.get('materials', '')[:500],
        cert         = data.get('cert', '')[:300],
        price        = float(data['price']) if data.get('price') else None,
        barcode      = data.get('barcode', '')[:50],
        eco_score    = float(data['eco_score'])    if data.get('eco_score')    else None,
        ethics_score = float(data['ethics_score']) if data.get('ethics_score') else None,
        carbon_level = data.get('carbon_level', ''),
        overall_score= float(data['overall_score'])if data.get('overall_score')else None,
        tags         = ','.join(data.get('tags', [])),
        source       = data.get('source', 'manual'),
    )

    # Add to live ML model
    total = add_user_product({
        'name':         product.name,
        'brand':        product.brand,
        'category':     product.category,
        'description':  product.description,
        'materials':    product.materials,
        'cert':         product.cert,
        'price':        product.price or 10.0,
        'eco_score':    product.eco_score or 5.0,
        'ethics_score': product.ethics_score or 5.0,
        'carbon_level': product.carbon_level or 'moderate',
    })

    return JsonResponse({
        'status':    'saved',
        'message':   f'"{product.name}" saved to your account!',
        'product_id': product.id,
        'total_products': total,
    })


@require_http_methods(["GET"])
def api_my_products(request):
    """GET /api/my-products/  — list saved products for logged-in user"""
    user = _get_logged_in_user(request)
    if not user:
        return JsonResponse({'error': 'Login required'}, status=401)

    from shop_assistant.models import UserSavedProduct
    products = UserSavedProduct.objects.filter(user=user).values(
        'id','name','brand','category','eco_score','ethics_score',
        'overall_score','carbon_level','tags','saved_at','source'
    )
    return JsonResponse({'status': 'ok', 'products': list(products)})
