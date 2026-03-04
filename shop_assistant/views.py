"""
Views for Ethical Shopping Assistant
Connects Django frontend to the AI recommendation model + EcoMind LLM.
"""
import sys
import os
import json
import urllib.request
import urllib.error
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
# ECOMIND LLM  — Our own trained neural network
# ═══════════════════════════════════════════════════════════

@csrf_exempt
@require_http_methods(["POST"])
def api_predict_ethical(request):
    """
    POST /api/predict/
    Uses our OWN trained EcoMindNet to predict ethical scores.
    Body: { name, brand, category, materials, sustainability_cert, description }
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


# ── Chatbot helpers ───────────────────────────────────────────
import importlib.util, pathlib
_cb_path = pathlib.Path(__file__).parent / "chatbot.py"
_cb_spec = importlib.util.spec_from_file_location("chatbot", _cb_path)
_cb_mod  = importlib.util.module_from_spec(_cb_spec)
_cb_spec.loader.exec_module(_cb_mod)
parse_and_respond = _cb_mod.parse_and_respond


def _build_products_text(df):
    lines = []
    for _, row in df.iterrows():
        price_inr = int(row["price"] * 83)
        lines.append(
            "- " + str(row["name"]) +
            " | Brand: " + str(row["brand"]) +
            " | Category: " + str(row["category"]) +
            " | Price: Rs" + str(price_inr) +
            " | Eco: " + str(row["eco_score"]) +
            " | Ethics: " + str(row["ethics_score"]) +
            " | Carbon: " + str(row["carbon_footprint"]) + "kg" +
            " | Cert: " + str(row["sustainability_cert"]) +
            " | Materials: " + str(row["materials"]) +
            " | Score: " + str(row["composite_score"]) +
            " | Desc: " + str(row["description"])
        )
    return "\n".join(lines)


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
    ".elc-summary{font-size:0.82rem;color:#4ade80;margin-top:0.5rem;padding-top:0.5rem;"
    "border-top:1px solid rgba(74,222,128,0.1);}"
    ".elc-error{font-size:0.83rem;color:#fbbf24;background:rgba(251,191,36,0.08);"
    "border:1px solid rgba(251,191,36,0.2);padding:0.75rem 1rem;border-radius:10px;}"
    ".elc-llm-badge{display:inline-block;background:rgba(74,222,128,0.1);"
    "border:1px solid rgba(74,222,128,0.3);color:#4ade80;padding:3px 10px;"
    "border-radius:5px;font-size:0.68rem;font-weight:700;margin-bottom:0.6rem;}"
    "</style>"
)


@csrf_exempt
@require_http_methods(["POST"])
def api_chat(request):
    """
    Chatbot powered by Gemini (language) + EcoMindNet (ethical scoring).
    """
    try:
        data    = json.loads(request.body)
        message = data.get("message", "").strip()
    except (json.JSONDecodeError, KeyError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if not message:
        return JsonResponse({"error": "Empty message"}, status=400)

    ai_model = get_model()
    products_text = _build_products_text(ai_model.df)

    llm_ready = os.path.exists(
        os.path.join(BASE_DIR, "ecomind_llm", "saved_model", "ecomind_net.pkl")
    )

    llm_note = (
        "\nNOTE: This website has its own trained EcoMind Neural Network "
        "that predicts eco scores, ethics scores, carbon level and tags. "
        "When showing products, mention scores are verified by EcoMind AI.\n"
        if llm_ready else ""
    )

    system_prompt = (
        "You are EcoMind, an expert AI ethical shopping assistant.\n"
        "You ONLY help with ethical and eco-friendly product recommendations.\n"
        "Politely refuse anything outside shopping.\n"
        + llm_note +
        "\nProduct catalogue:\n" + products_text +
        "\n\nINSTRUCTIONS:\n"
        "1. Understand budget in Rs, category, ethical preferences\n"
        "2. Pick TOP 3-5 matching products from catalogue ONLY\n"
        "3. If budget too low show cheapest available and mention price\n"
        "4. Return ONLY pure HTML, no markdown, no backticks\n\n"
        "HTML format per product:\n"
        '<div class="eco-llm-card">'
        '<div class="elc-header">'
        '<div>'
        '<a href="https://www.google.com/search?q=PRODUCT_NAME+BRAND+buy&tbm=shop" target="_blank" class="elc-name">PRODUCT NAME</a>'
        '<div class="elc-brand">by BRAND - CATEGORY</div>'
        '</div>'
        '<div class="elc-score">COMPOSITE_SCORE</div>'
        '</div>'
        '<p class="elc-desc">DESCRIPTION</p>'
        '<div class="elc-tags">'
        '<span class="elc-tag cert">CERTIFICATION</span>'
        '<span class="elc-tag carbon">CARBON kg CO2</span>'
        '<span class="elc-tag eco">Eco: ECO_SCORE</span>'
        '<span class="elc-tag ethics">Ethics: ETHICS_SCORE</span>'
        '</div>'
        '<div class="elc-footer">'
        '<span class="elc-price">Rs PRICE</span>'
        '<span class="elc-match">Why: ONE sentence reason</span>'
        '</div>'
        '</div>'
        '\nAfter all cards: <p class="elc-summary">Found X products matching your request.</p>'
    )

    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    gemini_url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-1.5-flash:generateContent?key=" + GEMINI_API_KEY
    )

    payload = json.dumps({
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": message}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048}
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            gemini_url, data=payload,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        raw_html = result["candidates"][0]["content"]["parts"][0]["text"]
        raw_html = raw_html.replace("```html", "").replace("```", "").strip()

        llm_badge = ""
        if llm_ready:
            llm_badge = (
                '<span class="elc-llm-badge">'
                '🧠 Scores verified by EcoMind Neural Network'
                '</span><br>'
            )

        return JsonResponse({
            "html": CARD_STYLES + llm_badge + raw_html,
            "status": "ok",
            "engine": "gemini+ecomind_llm" if llm_ready else "gemini"
        })

    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        return JsonResponse({
            "html": '<p style="color:#f87171;">Gemini error ' + str(e.code) + ': ' + err_body + '</p>',
            "status": "error"
        })

    except Exception as e:
        html = parse_and_respond(message, ai_model)
        return JsonResponse({"html": html, "status": "fallback", "engine": "rule-based"})