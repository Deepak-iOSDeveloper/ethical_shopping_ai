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
        # Leaders
        "Trump", "Modi", "Putin", "Xi Jinping", "Macron", "Starmer", "Netanyahu",
        "Zelenskyy", "Lula", "Milei", "Kim Jong Un", "Ayatollah", "Pezeshkian",
        "kim jung un", "kim jong un", "narendra", "rahul", "gandhi", "jawahar", "sunia", "samaj", "badi", "party",

        # Parties & Orgs
        "BJP", "Congress Party", "Tories", "Labour", "CCP", "NATO", "UNSC", "G7",
        "BRICS", "European Union", "Hezbollah", "Hamas", "Taliban", "RSS",

        # Issues & Conflicts
        "Gaza", "West Bank", "Taiwan Strait", "Ukraine War", "Red Sea Crisis",
        "Sanctions", "Tariffs", "Gerrymandering", "Sovereignty", "Secession",
        "Immigration", "Refugee", "Climate Accord", "Cyberwarfare", "Propaganda",

        # Countries (High Political Context)
        "Taiwan", "Iran", "Israel", "Palestine", "Ukraine", "Russia", "North Korea",
        "Venezuela", "Syria", "Sudan", "Myanmar", "China", "India", "USA",
    # A-E
    "abrogation", "absolutism", "abstention", "accountability", "activism", "adjudication",
    "administration", "adversary", "advocacy", "agenda", "alderman", "allegiance", "alliance",
    "amnesty", "anarchy", "annexation", "anti-clericalism", "apartheid", "appeasement",
    "apportionment", "appropriation", "aristocracy", "armistice", "asylum", "austerity",
    "authoritarianism", "autocracy", "autonomy", "backbencher", "ballot", "balkanization",
    "belligerent", "bicameral", "bilateralism", "bill", "bipartisan", "bloc", "border",
    "brinkmanship", "bureaucracy", "by-election", "cabinet", "campaign", "capitalism",
    "caucus", "censure", "centralism", "chancellor", "charter", "checks and balances",
    "citizenship", "civil liberties", "civil rights", "civil society", "cloture",
    "coalition", "collectivism", "colonialism", "communism", "confederation",
    "constituency", "constitution", "consulate", "containment", "convention", "corruption",
    "cosmopolitanism", "coup d'état", "decentralization", "decolonization", "decree",
    "deficit", "delegation", "demagogue", "democracy", "democratization", "deportation",
    "deregulation", "despotism", "detente", "deterrence", "devolution", "dictatorship",
    "diplomacy", "disarmament", "disenfranchisement", "dissident", "dissolution",
    "doctrine", "dogma", "domestic policy", "dovish", "electorate", "embargo", "emigration",
    "enfranchisement", "envoy", "espionage", "establishment", "ethics", "ethnicity",
    "executive", "exile", "exit poll", "expansionism", "extradition", "extremism",

    # F-N
    "faction", "fascism", "federalism", "filibuster", "fiscal", "foreign policy",
    "franchise", "fundamentalism", "geopolitics", "gerrymandering", "globalism",
    "governance", "grassroots", "guerrilla", "hard power", "hegemony", "hierarchy",
    "ideology", "impeachment", "imperialism", "inauguration", "incumbent", "indemnity",
    "independence", "indoctrination", "initiative", "insurgency", "integration",
    "interdependence", "interim", "interventionism", "isolationism", "jingoism",
    "judicial review", "judiciary", "junta", "jurisdiction", "knightly", "laissez-faire",
    "lawmaker", "left-wing", "legislation", "legislature", "legitimacy", "liberalism",
    "libertarianism", "lobbyist", "local government", "majoritarianism", "mandate",
    "manifesto", "marginal seat", "martial law", "marxism", "mayor", "militarism",
    "minister", "minority", "monarchy", "multilateralism", "municipality", "nationalism",
    "nationalization", "nativism", "naturalization", "negotiation", "neutrality",
    "nihilism", "nomination", "non-alignment", "non-intervention", "normalization",

    # O-Z
    "oligarchy", "opposition", "ordinance", "pacifism", "parliament", "partisan",
    "patriotism", "patronage", "petition", "platform", "plebiscite", "pluralism",
    "plutocracy", "policy", "politburo", "populism", "portfolio", "prerogative",
    "presidency", "primary", "privatization", "proclamation", "progressive",
    "propaganda", "proportional", "prorogation", "protectorate", "protocol", "proxy",
    "psephology", "public sector", "purges", "quorum", "radicalization", "ratification",
    "reactionary", "realpolitik", "redistricting", "referendum", "reform", "refugee",
    "regime", "regulation", "republic", "reservation", "resolution", "revisionism",
    "revolution", "rhetoric", "right-wing", "sanctions", "secession", "secularism",
    "sedition", "segregation", "senate", "separation of powers", "sharia", "socialism",
    "sovereignty", "speaker", "sphere of influence", "statism", "statute", "suffrage",
    "summit", "supranational", "surveillance", "theocracy", "totalitarianism", "treaty",
    "tribalism", "tyranny", "unilateralism", "unitary state", "uprising", "urbanization",
    "veto", "voter turnout", "welfare state", "whip", "writ", "xenophobia", "zionism"
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
        "marry", "woman", "man", "married", "when", "how",
        "Pakistan", "pakistan", "north korea", "america", "israel", "nato", "china", "south korea", "russia", "ukrain"
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
        # ── Weather & Nature ──────────────────────────────────────
        "weather", "temperature", "forecast", "rain", "sunny", "cloudy", "humidity",
        "wind", "storm", "tornado", "hurricane", "earthquake", "flood", "drought",
        "snowfall", "hailstorm", "thunder", "lightning", "climate change", "global warming",
        "monsoon", "season", "winter", "summer", "spring", "autumn", "celsius", "fahrenheit",

        # ── News & Current Events ─────────────────────────────────
        "news", "headline", "breaking news", "latest news", "today news", "current events",
        "newspaper", "journalist", "reporter", "media", "press", "broadcast", "editorial",
        "bbc", "cnn", "ndtv", "times of india", "hindustan times", "the hindu", "aaj tak",

        # ── Sports ────────────────────────────────────────────────
        "cricket", "football", "soccer", "basketball", "tennis", "badminton", "hockey",
        "volleyball", "baseball", "rugby", "golf", "swimming", "athletics", "marathon",
        "ipl", "world cup", "premier league", "la liga", "bundesliga", "nba", "nfl",
        "fifa", "icc", "bcci", "olympics", "commonwealth games", "asian games",
        "match", "tournament", "championship", "league", "stadium", "wicket", "goal",
        "score", "batsman", "bowler", "striker", "goalkeeper", "referee", "umpire",
        "virat kohli", "rohit sharma", "dhoni", "sachin", "messi", "ronaldo", "federer",
        "nadal", "djokovic", "neymar", "mbappé", "lebron", "serena williams",
        "medal", "trophy", "gold medal", "silver medal", "bronze medal",

        # ── Movies & Entertainment ────────────────────────────────
        "movie", "film", "cinema", "actor", "actress", "director", "bollywood",
        "hollywood", "netflix", "amazon prime", "hotstar", "disney plus", "hulu",
        "web series", "season", "episode", "trailer", "review", "box office",
        "oscar", "grammy", "filmfare", "bafta", "golden globe", "emmy",
        "srk", "shah rukh", "salman khan", "aamir khan", "deepika", "priyanka",
        "ranveer", "ranbir", "alia bhatt", "tom cruise", "brad pitt", "leonardo",
        "animated", "cartoon", "anime", "manga", "superhero", "marvel", "dc comics",
        "avengers", "spider-man", "batman", "superman", "ironman",

        # ── Music ─────────────────────────────────────────────────
        "song", "music", "album", "playlist", "lyrics", "singer", "rapper",
        "band", "concert", "spotify", "youtube music", "gaana", "saavn",
        "arijit singh", "shreya ghoshal", "ar rahman", "taylor swift", "ed sheeran",
        "drake", "eminem", "billie eilish", "weeknd", "beyonce", "rihanna",
        "pop", "rock", "jazz", "classical", "hip hop", "rap", "edm", "lo-fi",
        "guitar", "piano", "drums", "violin", "beats", "remix", "cover song",
        "music video", "music album", "chart topper",

        # ── Food Recipes ──────────────────────────────────────────
        "recipe", "cook", "cooking", "bake", "baking", "how to make", "ingredients",
        "dish", "cuisine", "meal", "breakfast recipe", "lunch recipe", "dinner recipe",
        "biryani recipe", "pasta recipe", "pizza recipe", "burger recipe", "cake recipe",
        "chef", "kitchen tips", "food blog", "calories", "nutrition facts",
        "protein content", "how many calories", "diet plan", "keto", "intermittent fasting",

        # ── Jokes & Entertainment ─────────────────────────────────
        "joke", "funny", "meme", "prank", "comedy", "laugh", "hilarious",
        "tell me a joke", "make me laugh", "roast", "sarcasm", "riddle",
        "knock knock", "pun", "one liner", "stand up comedy",

        # ── Creative Writing ──────────────────────────────────────
        "poem", "poetry", "story", "write a", "write me", "essay", "paragraph",
        "speech", "letter", "email draft", "cover letter", "application letter",
        "creative writing", "fiction", "novel", "short story", "haiku", "sonnet",
        "rhyme", "limerick", "caption", "quote", "motivational quote", "slogan",

        # ── General Knowledge ─────────────────────────────────────
        "capital of", "who is", "what is the", "tell me about", "explain",
        "define", "meaning of", "definition", "history of", "origin of",
        "biography", "born in", "founded by", "invented by", "discovered by",
        "who invented", "when was", "how old is", "how tall is", "how far",
        "distance between", "population of", "area of", "currency of",
        "national animal", "national bird", "national flower", "flag of",
        "president of", "prime minister of", "capital city",

        # ── Language & Translation ────────────────────────────────
        "translate", "translation", "in hindi", "in english", "in tamil",
        "in telugu", "in marathi", "in bengali", "in french", "in spanish",
        "in arabic", "in chinese", "in japanese", "synonym", "antonym",
        "grammar", "spell check", "correct my", "proofread",

        # ── Mathematics ───────────────────────────────────────────
        "math", "maths", "calculate", "calculation", "solve", "equation",
        "algebra", "geometry", "calculus", "trigonometry", "statistics",
        "percentage", "percentage of", "square root", "cube root", "factorial",
        "prime number", "fibonacci", "probability", "matrix", "derivative",
        "integral", "differential", "addition", "subtraction", "multiplication",
        "division", "fraction", "decimal", "binary", "hexadecimal",

        # ── Science ───────────────────────────────────────────────
        "physics", "chemistry", "biology", "science", "hypothesis", "theory",
        "experiment", "periodic table", "element", "atom", "molecule", "electron",
        "proton", "neutron", "nucleus", "dna", "rna", "cell", "photosynthesis",
        "gravity", "newton", "einstein", "quantum", "relativity", "black hole",
        "solar system", "planet", "galaxy", "universe", "big bang", "evolution",
        "darwin", "mendel", "genetics", "chromosome",

        # ── Coding & Technology ───────────────────────────────────
        "code", "coding", "program", "programming", "python", "java", "javascript",
        "html", "css", "react", "angular", "vue", "nodejs", "django", "flask",
        "sql", "database", "algorithm", "data structure", "api", "rest api",
        "machine learning", "deep learning", "neural network", "chatgpt", "ai model",
        "openai", "gemini", "llm", "gpt", "debug", "error fix", "stack overflow",
        "github", "git", "docker", "kubernetes", "cloud", "aws", "azure", "devops",
        "cybersecurity", "hacking", "linux", "terminal", "bash", "command line",
        "software", "hardware", "operating system", "windows", "macos", "ubuntu",
        "app development", "ios", "android", "flutter", "swift", "kotlin",

        # ── Vehicles ──────────────────────────────────────────────
        "car", "bike", "petrol", "diesel", "motorcycle", "vehicle", "automobile",
        "truck", "bus", "train", "metro", "auto rickshaw", "taxi", "uber",
        "ola", "rapido", "electric vehicle", "ev", "tesla", "tata nexon",
        "honda", "toyota", "bmw", "mercedes", "audi", "hyundai", "maruti",
        "mahindra", "bajaj", "royal enfield", "ktm", "yamaha", "suzuki",
        "engine", "horsepower", "mileage", "gear", "clutch", "brake",
        "tyre", "fuel", "cng", "hybrid car", "scooter", "cycle",

        # ── Travel & Tourism ──────────────────────────────────────
        "flight", "hotel", "travel", "trip", "tour", "vacation", "holiday",
        "visa", "passport", "booking", "airbnb", "makemytrip", "goibibo",
        "irctc", "train ticket", "bus ticket", "flight ticket", "itinerary",
        "tourist place", "destination", "resort", "hostel", "check in",
        "check out", "luggage", "backpack trip", "road trip", "cruise",
        "goa", "kerala", "rajasthan", "manali", "shimla", "ooty", "darjeeling",
        "paris", "london", "dubai", "singapore", "thailand", "bali", "maldives",
        "new york", "tokyo", "sydney", "rome", "barcelona",

        # ── Health & Medical ──────────────────────────────────────
        "doctor", "medicine", "hospital", "disease", "symptom", "cure",
        "covid", "vaccine", "tablet", "prescription", "diagnosis", "treatment",
        "surgery", "operation", "therapy", "physiotherapy", "blood test",
        "fever", "headache", "cold", "flu", "infection", "allergy",
        "diabetes", "cancer", "blood pressure", "heart attack", "stroke",
        "asthma", "thyroid", "cholesterol", "weight loss", "bmi",
        "calorie deficit", "gym workout", "exercise routine", "yoga pose",
        "meditation technique", "mental health tips", "anxiety treatment",
        "depression", "psychiatrist", "psychologist",

        # ── Finance & Banking ─────────────────────────────────────
        "stock", "crypto", "bitcoin", "invest", "share market", "trading",
        "loan", "emi", "bank", "insurance", "mutual fund", "sip", "fd",
        "rd", "ppf", "nps", "ipo", "sensex", "nifty", "nasdaq", "dow jones",
        "portfolio", "dividend", "compound interest", "simple interest",
        "tax", "income tax", "gst return", "itr filing", "pan card", "aadhaar",
        "credit card", "debit card", "upi", "paytm", "phonepe", "gpay",
        "neft", "rtgs", "imps", "account number", "ifsc", "swift code",
        "inflation", "recession", "gdp", "repo rate", "rbi", "sebi",
        "zerodha", "groww", "upstox", "angel broking", "hedge fund",

        # ── Education & Career ────────────────────────────────────
        "study", "exam", "test", "quiz", "syllabus", "notes", "lecture",
        "university", "college", "school", "jee", "neet", "upsc", "gate",
        "cat", "gmat", "gre", "ielts", "toefl", "sat", "scholarship",
        "admission", "fees", "degree", "diploma", "certificate", "internship",
        "job", "resume", "cv", "interview", "salary", "appraisal", "promotion",
        "linkedin", "naukri", "indeed", "glassdoor", "placement",

        # ── Relationships & Personal ──────────────────────────────
        "girlfriend", "boyfriend", "marriage", "wedding", "divorce", "breakup",
        "relationship advice", "dating", "tinder", "bumble", "propose",
        "love letter", "anniversary", "valentine", "friendship",
        "family problem", "parent issue", "sibling fight", "loneliness",
        "how to impress", "how to talk to", "how to make friends",

        # ── Astrology & Superstition ──────────────────────────────
        "horoscope", "zodiac", "astrology", "kundli", "rashifal", "numerology",
        "tarot", "palm reading", "vastu", "feng shui", "lucky number",
        "lucky colour", "gemstone", "superstition", "black magic",

        # ── Gaming ────────────────────────────────────────────────
        "game", "gaming", "pubg", "free fire", "bgmi", "fortnite", "minecraft",
        "gta", "call of duty", "valorant", "lol", "dota", "chess online",
        "playstation", "xbox", "nintendo", "steam", "esports", "twitch",
        "streamer", "youtuber", "content creator",

        # ── Social Media ──────────────────────────────────────────
        "instagram", "facebook", "twitter", "snapchat", "tiktok", "reddit",
        "whatsapp", "telegram", "discord", "pinterest", "tumblr", "quora",
        "follower", "like", "comment", "share", "viral", "trending", "hashtag",
        "influencer", "reel", "story", "post", "dm", "block", "unfollow",

        # ── Legal ─────────────────────────────────────────────────
        "lawyer", "advocate", "legal advice", "court", "judge", "police",
        "fir", "complaint", "bail", "arrest", "law", "constitution",
        "rights", "act", "section", "ipc", "high court", "supreme court",
        "property dispute", "divorce case", "cybercrime",

        # ── Random / Miscellaneous ────────────────────────────────
        "how are you", "what are you", "are you human", "are you robot",
        "who made you", "who created you", "where are you from",
        "what is your name", "how old are you", "do you have feelings",
        "can you think", "are you conscious", "pass time", "bored",
        "good night", "good afternoon", "thank you", "thanks", "bye",
        "goodbye", "see you", "take care", "ok cool", "nice", "wow",
        "amazing", "awesome", "interesting", "really", "seriously",
        # out of syllabus
        "girlfriend", "boyfriend", "mother", "father", "sister", "relationship", "gift", "nothing"
    ]
    if any(o in text_lower for o in off_topic_signals):
        # Make sure it's truly off-topic and not accidentally matching a product word
        shopping_signals = [
            "product", "buy", "shop", "organic", "eco", "sustainable",
            "ethical", "carbon", "certified", "natural", "vegan", "fair trade",
            "clothing", "food", "kitchen", "personal care", "footwear",
            "under", "below", "rupees", "price", "budget",
            # ── Original ──────────────────────────────────────────────
            "product", "buy", "shop", "organic", "eco", "sustainable",
            "ethical", "carbon", "certified", "natural", "vegan", "fair trade",
            "clothing", "food", "kitchen", "personal care", "footwear",
            "under", "below", "rupees", "price", "budget",

            # ── Ethical & Sustainability Terms ────────────────────────
            "ethical", "ethically", "eco-friendly", "environmentally friendly",
            "planet friendly", "earth friendly", "green product", "green brand",
            "conscious", "conscious shopping", "conscious consumer", "responsible",
            "responsible sourcing", "responsible brand", "sustainable brand",
            "sustainability", "sustainably made", "sustainably sourced",
            "low impact", "zero impact", "minimal impact", "planet positive",
            "carbon neutral", "carbon negative", "carbon offset", "carbon footprint",
            "carbon free", "low carbon", "ultra low carbon", "net zero",
            "climate friendly", "climate conscious", "eco score", "ethics score",

            # ── Certifications ────────────────────────────────────────
            "certified", "certification", "gots", "oeko tex", "bluesign",
            "fair trade certified", "fairtrade", "b corp", "b-corp",
            "usda organic", "cosmos organic", "ecocert", "fsc certified",
            "rainforest alliance", "cruelty free", "leaping bunny",
            "peta approved", "non gmo", "non-gmo verified", "soil association",
            "demeter", "ifoam", "eu organic", "india organic",

            # ── Eco Materials ─────────────────────────────────────────
            "organic cotton", "recycled cotton", "recycled polyester",
            "recycled plastic", "upcycled", "upcycled material", "reclaimed",
            "bamboo", "hemp", "linen", "jute", "cork", "seaweed", "lyocell",
            "tencel", "modal", "wool", "merino wool", "alpaca", "silk",
            "natural fiber", "natural material", "plant based material",
            "biodegradable", "compostable", "plastic free", "zero plastic",
            "ocean plastic", "recycled ocean plastic", "post consumer",
            "post consumer recycled", "pre consumer recycled",

            # ── Product Categories ────────────────────────────────────
            "clothing", "apparel", "fashion", "outfit", "wear", "shirt",
            "tshirt", "t-shirt", "jeans", "trouser", "dress", "skirt",
            "jacket", "hoodie", "sweater", "activewear", "sportswear",
            "swimwear", "underwear", "socks", "scarf", "hat", "cap",
            "food", "snack", "breakfast", "lunch", "dinner", "meal",
            "grocery", "pantry", "staple", "grain", "cereal", "muesli",
            "granola", "oats", "quinoa", "lentil", "pulse", "seed",
            "nut", "dried fruit", "tea", "coffee", "herbal tea",
            "supplement", "protein powder", "superfood",
            "personal care", "skincare", "haircare", "body care",
            "face wash", "shampoo", "conditioner", "moisturiser", "serum",
            "sunscreen", "deodorant", "toothpaste", "toothbrush", "soap",
            "body wash", "lotion", "lip balm", "face mask", "toner",
            "kitchen", "cookware", "utensil", "container", "storage",
            "reusable bag", "lunch box", "water bottle", "flask",
            "beeswax wrap", "silicone bag", "compost bin", "reusable straw",
            "cleaning", "detergent", "dish soap", "laundry", "surface cleaner",
            "footwear", "shoes", "sandals", "boots", "sneakers", "slippers",
            "baby", "baby product", "baby care", "toy", "diaper",
            "home decor", "candle", "diffuser", "bedding", "linen",
            "stationery", "notebook", "pen", "paper", "office supply",
            "pet", "pet food", "pet care", "dog", "cat",
            "electronics", "gadget", "charger", "solar", "solar panel",

            # ── Shopping Intent Words ─────────────────────────────────
            "buy", "purchase", "order", "get", "find", "show", "recommend",
            "suggest", "looking for", "need", "want", "search", "explore",
            "discover", "browse", "compare", "best", "top", "good",
            "affordable", "cheap", "expensive", "budget", "price", "cost",
            "under", "below", "within", "less than", "more than", "around",
            "rupees", "rs", "inr", "₹", "usd", "dollar",

            # ── Ethical Shopping Specific ─────────────────────────────
            "ethical shopping", "conscious shopping", "green shopping",
            "sustainable shopping", "eco shopping", "responsible shopping",
            "slow fashion", "fast fashion alternative", "ethical brand",
            "green brand", "eco brand", "sustainable brand", "conscious brand",
            "zero waste", "zero waste product", "package free", "packaging free",
            "minimal packaging", "plastic free packaging", "refillable",
            "refill", "reusable", "reuse", "reduce", "recycle",
            "circular economy", "circular product", "upcycle",
            "supply chain", "transparent brand", "ethical supply chain",
            "fair wage", "fair labour", "worker rights", "safe factory",

            # ── EcoMind Specific ──────────────────────────────────────
            "ecomind", "eco mind", "eco score", "ethics score", "composite score",
            "sustainability score", "ethical rating", "green rating",
            "carbon rating", "environmental rating", "ai recommendation",
            "recommend me", "what should i buy", "which product",
            "best eco product", "most ethical", "least carbon",
            "highest rated", "top rated ethical", "best certified",
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
