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