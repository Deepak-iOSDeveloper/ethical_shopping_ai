"""
EcoMind Chatbot NLP Engine
Parses natural language shopping queries → matches ethical products from the AI model.
"""
import re

# ── Constants ─────────────────────────────────────────────────
INR_TO_USD = 1 / 83.0

# ── Category keyword map ──────────────────────────────────────
CATEGORY_KEYWORDS = {
    "Food": [
        "food", "breakfast", "lunch", "dinner", "eat", "drink", "coffee", "tea",
        "snack", "meal", "grain", "spice", "oil", "honey", "chocolate", "rice",
        "protein", "beverage", "sugar", "flour", "bar", "chia", "quinoa", "moringa",
        "turmeric", "jaggery", "cacao", "oat", "almond", "lentil", "vinegar",
    ],
    "Personal Care": [
        "shampoo", "soap", "deodorant", "moisturizer", "skincare", "toothbrush",
        "toothpaste", "lotion", "serum", "perfume", "razor", "sunscreen",
        "conditioner", "body wash", "face wash", "lip balm", "hair oil",
        "personal care", "hygiene", "beauty", "grooming", "cream", "scrub",
        "argan", "neem", "ayurvedic", "rose water",
    ],
    "Clothing": [
        "shirt", "tee", "top", "dress", "pants", "jeans", "jacket", "coat",
        "sweater", "hoodie", "kurta", "saree", "clothes", "clothing", "wear",
        "outfit", "fashion", "blazer", "leggings", "socks", "pyjamas", "scarf",
        "shawl", "overalls", "jogger", "khadi", "linen", "cotton", "cashmere",
    ],
    "Footwear": [
        "shoes", "sneakers", "boots", "sandals", "footwear", "chappal",
        "flip flops", "loafers", "slippers", "espadrilles",
    ],
    "Sportswear": [
        "gym", "yoga", "sport", "workout", "activewear", "running", "athletic",
        "swim", "exercise", "fitness", "track", "compression",
    ],
    "Kitchen": [
        "kitchen", "cooking", "bottle", "mug", "straw", "cutlery", "lunch box",
        "food storage", "wrap", "napkin", "chopping", "dish", "jar", "produce",
        "trash bag", "skillet", "pan", "loofah", "water filter",
    ],
    "Home": [
        "home", "bedroom", "bedding", "pillow", "mattress", "blanket", "sheet",
        "towel", "curtain", "rug", "sofa", "furniture", "candle", "vase",
        "basket", "door mat", "shower curtain", "decor",
    ],
    "Sports": [
        "yoga mat", "camping", "hiking", "trekking", "outdoor", "surf", "climb",
        "hammock", "sleeping bag", "lantern", "wetsuit", "football", "resistance band",
    ],
    "Accessories": [
        "bag", "backpack", "wallet", "sunglasses", "tote", "belt", "hat",
        "umbrella", "laptop bag", "fanny pack", "jewellery", "jewelry", "pendant",
    ],
    "Electronics": [
        "phone", "charger", "solar charger", "laptop", "speaker", "keyboard",
        "lamp", "light", "energy", "electronics", "tech", "gadget", "e-bike",
    ],
    "Accessories": [
        "bag", "backpack", "wallet", "sunglasses", "tote", "belt", "hat",
        "umbrella", "laptop bag", "fanny pack", "jewellery", "jewelry", "pendant",
        "mobile cover", "phone case", "mobile case", "phone cover", "phone pouch",
        "mobile pouch", "phone stand", "phone holder", "cover", "case",
    ],
    "Baby & Kids": [
        "baby", "kids", "child", "infant", "toddler", "toy", "diaper", "onesie",
        "swaddle", "teether", "baby care", "newborn",
    ],
    "Baby": [
        "baby", "infant", "newborn", "toddler", "diaper", "onesie", "baby soap",
        "baby oil", "baby wipes", "teether", "baby food",
    ],
    "Health": [
        "health", "tablet", "capsule", "supplement", "vitamin", "ayurvedic",
        "triphala", "tulsi", "giloy", "ashwagandha", "immunity", "sanitizer",
        "tongue cleaner", "datun", "wellness", "herbal", "medicine",
    ],
    "Cleaning": [
        "cleaning", "cleaner", "detergent", "laundry", "dish wash", "dishwash",
        "toilet cleaner", "floor cleaner", "window cleaner", "soap nut", "soapnut",
        "washing powder", "mop", "broom", "scrubber", "cloth wipe",
    ],
    "Stationery": [
        "notebook", "pen", "pencil", "stationery", "diary", "journal",
        "notepad", "bulletin board", "cork board", "school", "writing", "office",
        "paper", "study", "seed paper", "recycled paper",
    ],
    "Pet Care": [
        "pet", "dog", "cat", "puppy", "kitten", "pet food", "pet shampoo",
        "collar", "leash", "pet bowl", "cat litter", "dog treat", "animal",
    ],
}

# ── Ethical keyword map ───────────────────────────────────────
ETHICAL_KEYWORDS = {
    "organic":     ["organic", "usda organic", "certified organic", "natural"],
    "fair_trade":  ["fair trade", "fairtrade", "fair-trade", "ethical labor", "no child labour"],
    "b_corp":      ["b-corp", "b corp", "bcorp", "certified b"],
    "vegan":       ["vegan", "cruelty free", "cruelty-free", "animal free", "plant based"],
    "recycled":    ["recycled", "upcycled", "recycle", "recycling", "second life"],
    "low_carbon":  ["low carbon", "carbon", "carbon footprint", "co2", "carbon neutral", "climate"],
    "natural":     ["natural", "plant based", "plant-based", "nature", "botanical", "herb"],
    "zero_waste":  ["zero waste", "zero-waste", "package free", "packaging free", "no plastic"],
    "sustainable": ["sustainable", "sustainability", "eco", "eco-friendly", "green", "earth"],
    "certified":   ["certified", "certification", "gots", "oeko-tex", "bluesign", "fsc"],
}

# ── Cert tag colors ───────────────────────────────────────────
CERT_COLORS = {
    "GOTS":              ("#4ade80", "#0b2e17"),
    "Fair Trade":        ("#fbbf24", "#2d1f00"),
    "B-Corp":            ("#60a5fa", "#0a1a2e"),
    "USDA Organic":      ("#86efac", "#0b2518"),
    "Certified Organic": ("#86efac", "#0b2518"),
    "OEKO-TEX":          ("#a78bfa", "#1a0a2e"),
    "COSMOS":            ("#34d399", "#0a2018"),
    "FSC":               ("#6ee7b7", "#072418"),
    "Non-GMO":           ("#fcd34d", "#271d00"),
    "Bluesign":          ("#38bdf8", "#0a1e2e"),
    "GOLS":              ("#4ade80", "#0b2e17"),
    "Vegan Society":     ("#86efac", "#0b2518"),
    "PETA Approved":     ("#f9a8d4", "#2d0a18"),
    "Khadi Mark":        ("#fde68a", "#271d00"),
    "Silk Mark":         ("#e9d5ff", "#1a0a2e"),
    "Energy Star":       ("#fcd34d", "#271d00"),
    "ZQ Merino":         ("#6ee7b7", "#072418"),
    "Woolmark":          ("#c4b5fd", "#1a0a2e"),
}


def _get_cert_style(cert):
    """Return inline style string for a certification badge."""
    bg, fg = CERT_COLORS.get(cert, ("#2d6a4f", "#a7f3c1"))
    return f"background:{bg};color:{fg};padding:2px 8px;border-radius:5px;font-size:0.7rem;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;"


def _extract_budget_inr(text):
    """Extract budget in INR from text, return USD float or None."""
    text = text.lower()

    # Patterns: ₹500, rs 500, rupees 500, 500 rupees, 500rs, inr 500
    patterns = [
        r'₹\s*([\d,]+)',
        r'rs\.?\s*([\d,]+)',
        r'inr\s*([\d,]+)',
        r'([\d,]+)\s*(?:rupees?|rs\.?|₹|inr)',
        r'under\s+([\d,]+)',
        r'below\s+([\d,]+)',
        r'less\s+than\s+([\d,]+)',
        r'within\s+([\d,]+)',
        r'budget\s+(?:of\s+|is\s+)?([\d,]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            amount_str = match.group(1).replace(',', '')
            try:
                inr = float(amount_str)
                return inr * INR_TO_USD
            except ValueError:
                continue
    return None


def _extract_categories(text):
    """Detect which product categories the user is asking about."""
    text_lower = text.lower()
    matched = []
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                matched.append(category)
                break
    return list(set(matched))


def _extract_ethical_filters(text):
    """Detect ethical requirements mentioned by the user."""
    text_lower = text.lower()
    filters = []
    for filter_name, keywords in ETHICAL_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                filters.append(filter_name)
                break
    return filters


def _extract_keyword(text):
    """Extract a generic keyword for model search."""
    # Pull meaningful content words from the query
    stop = {"i", "want", "need", "looking", "find", "show", "get", "give", "me",
            "some", "a", "an", "the", "and", "or", "with", "for", "that", "which",
            "should", "is", "are", "be", "have", "has", "very", "quite", "really",
            "less", "more", "most", "best", "good", "great", "nice", "please",
            "below", "under", "within", "budget", "price", "cost", "rupees", "rs",
            "inr", "usd", "dollar", "₹", "around", "approximately", "about",
            "suggest", "recommend", "product", "products", "item", "items"}
    words = re.findall(r'[a-z]+', text.lower())
    keywords = [w for w in words if w not in stop and len(w) > 3]
    return " ".join(keywords[:4]) if keywords else None


def _score_row_against_filters(row, ethical_filters):
    """Check how well a product row matches the ethical filter list."""
    score = 0
    cert = row.get("sustainability_cert", "").lower()
    materials = row.get("materials", "").lower()
    description = row.get("description", "").lower()
    combined = cert + " " + materials + " " + description

    filter_map = {
        "organic":     ["organic", "usda", "cosmos", "gots"],
        "fair_trade":  ["fair trade", "fairtrade"],
        "b_corp":      ["b-corp", "b corp"],
        "vegan":       ["vegan", "peta", "plant"],
        "recycled":    ["recycled", "upcycled"],
        "low_carbon":  [],   # handled by carbon_footprint value
        "natural":     ["natural", "plant", "botanical", "herb", "organic"],
        "zero_waste":  ["zero", "package", "compostable", "refill"],
        "sustainable": ["sustainable", "eco", "certified"],
        "certified":   ["gots", "oeko", "bluesign", "fsc", "certified", "fair trade",
                        "b-corp", "usda", "cosmos"],
    }

    for f in ethical_filters:
        keywords = filter_map.get(f, [])
        for kw in keywords:
            if kw in combined:
                score += 1
                break
        if f == "low_carbon" and row.get("carbon_footprint", 99) <= 1.0:
            score += 2

    return score


def _build_product_link(name, brand):
    """Build a Google Shopping search link for the product."""
    query = f"{name} {brand} buy".replace(" ", "+")
    return f"https://www.google.com/search?q={query}&tbm=shop"


def _format_tag(cert):
    """Return HTML for a certification tag."""
    style = _get_cert_style(cert)
    return f'<span style="{style}">{cert}</span>'


def _make_ethical_tags(row):
    """Generate all relevant ethical tags for a product."""
    tags = []
    cert = row.get("sustainability_cert", "")
    if cert:
        tags.append(_format_tag(cert))

    materials = row.get("materials", "").lower()
    eco = row.get("eco_score", 0)
    ethics = row.get("ethics_score", 0)
    carbon = row.get("carbon_footprint", 99)

    if "organic" in materials or "organic" in cert.lower():
        tags.append(_format_tag("Organic"))
    if carbon <= 0.5:
        tags.append('<span style="background:#dcfce7;color:#14532d;padding:2px 8px;border-radius:5px;font-size:0.7rem;font-weight:700;">Ultra Low CO₂</span>')
    elif carbon <= 1.5:
        tags.append('<span style="background:#d1fae5;color:#064e3b;padding:2px 8px;border-radius:5px;font-size:0.7rem;font-weight:700;">Low Carbon</span>')
    if eco >= 9.3:
        tags.append('<span style="background:#bbf7d0;color:#14532d;padding:2px 8px;border-radius:5px;font-size:0.7rem;font-weight:700;">🌿 Eco Champion</span>')
    if ethics >= 9.3:
        tags.append('<span style="background:#fef9c3;color:#713f12;padding:2px 8px;border-radius:5px;font-size:0.7rem;font-weight:700;">⚖️ Highly Ethical</span>')
    if "recycled" in materials or "upcycled" in materials:
        tags.append('<span style="background:#e0f2fe;color:#0c4a6e;padding:2px 8px;border-radius:5px;font-size:0.7rem;font-weight:700;">♻️ Recycled</span>')
    if "bamboo" in materials or "hemp" in materials or "cork" in materials:
        tags.append('<span style="background:#dcfce7;color:#14532d;padding:2px 8px;border-radius:5px;font-size:0.7rem;font-weight:700;">🌱 Natural Material</span>')
    return tags


def parse_and_respond(user_message, model):
    """
    Main chatbot function.
    Takes raw user message + loaded AI model, returns HTML response string.
    """
    text = user_message.strip()
    if len(text) < 3:
        return _greeting_response()

    text_lower = text.lower()

    # ── Greetings ─────────────────────────────────────────────
    greet_words = ["hello", "hi", "hey", "good morning", "good evening", "howdy", "hola"]
    if any(text_lower.startswith(g) for g in greet_words) and len(text.split()) <= 4:
        return _greeting_response()

    # ── Help intent ───────────────────────────────────────────
    help_words = ["help", "what can you do", "how does this work", "capabilities", "what do you know"]
    if any(h in text_lower for h in help_words):
        return _help_response()

    # ── Extract intent ────────────────────────────────────────
    budget_usd   = _extract_budget_inr(text)
    categories   = _extract_categories(text)
    eth_filters  = _extract_ethical_filters(text)
    keyword      = _extract_keyword(text)

    # ── Query the model ───────────────────────────────────────
    df = model.df.copy()

    # Filter by category if detected
    if categories:
        mask = df["category"].isin(categories)
        df = df[mask]

    # Filter by budget — gracefully relax if too tight
    budget_relaxed = False
    if budget_usd:
        strict = df[df["price"] <= budget_usd]
        if strict.empty:
            # Budget too tight: show cheapest available with a note
            budget_relaxed = True
        else:
            df = strict

    # Filter by keyword
    if keyword:
        kw = keyword.lower()
        kw_mask = (
            df["name"].str.lower().str.contains(kw, na=False) |
            df["brand"].str.lower().str.contains(kw, na=False) |
            df["materials"].str.lower().str.contains(kw, na=False) |
            df["description"].str.lower().str.contains(kw, na=False) |
            df["sustainability_cert"].str.lower().str.contains(kw, na=False) |
            df["category"].str.lower().str.contains(kw, na=False)
        )
        kw_df = df[kw_mask]
        if not kw_df.empty:
            df = kw_df

    # Apply ethical filters as soft re-ranking
    if eth_filters and not df.empty:
        df = df.copy()
        df["_eth_match"] = df.apply(
            lambda row: _score_row_against_filters(row.to_dict(), eth_filters), axis=1
        )

        # Hard filter: low_carbon requested → keep only carbon ≤ 2.0
        if "low_carbon" in eth_filters:
            low_c = df[df["carbon_footprint"] <= 2.0]
            if not low_c.empty:
                df = low_c

        # Hard filter: organic/certified → keep only cert-matched
        if "organic" in eth_filters or "certified" in eth_filters:
            org_certs = ["organic", "gots", "cosmos", "usda", "certified", "non-gmo", "fsc"]
            org_mask = df["sustainability_cert"].str.lower().apply(
                lambda c: any(kw in c for kw in org_certs)
            )
            if org_mask.any():
                df = df[org_mask]

        df = df.sort_values(["_eth_match", "composite_score"], ascending=[False, False])
    else:
        df = df.sort_values("composite_score", ascending=False)

    # Limit to top 5 for chat
    results = df.head(5)

    if results.empty:
        return _no_results_response(text, budget_usd, categories)

    # ── Predict scores & build HTML ───────────────────────────
    return _build_results_html(results, text, budget_usd, categories, eth_filters, model, budget_relaxed)


def _build_results_html(results, query, budget_usd, categories, eth_filters, model, budget_relaxed=False):
    """Render the product results as rich HTML for the chat bubble."""
    budget_inr = int(budget_usd * 83) if budget_usd else None

    # Summary line
    cat_str = ", ".join(categories) if categories else "all categories"
    budget_str = f" under ₹{budget_inr}" if budget_inr else ""
    filter_str = ", ".join(eth_filters).replace("_", " ") if eth_filters else ""

    if budget_relaxed:
        min_price_inr = int(results["price"].min() * 83)
        summary = f'<p style="font-size:0.82rem;color:#fbbf24;margin-bottom:0.5rem;">⚠️ No products found under ₹{budget_inr}. Showing closest matches starting from ₹{min_price_inr}.</p>'
        summary += f'<p style="font-size:0.82rem;color:#8aaa94;margin-bottom:1rem;">Found <strong style="color:#4ade80">{len(results)}</strong> products in <em>{cat_str}</em>'
    else:
        summary = f'<p style="font-size:0.82rem;color:#8aaa94;margin-bottom:1rem;">Found <strong style="color:#4ade80">{len(results)}</strong> ethical products in <em>{cat_str}</em>{budget_str}'
    if filter_str:
        summary += f' · filters: <em>{filter_str}</em>'
    summary += '</p>'

    cards_html = ""
    for i, (_, row) in enumerate(results.iterrows()):
        # Predict ethical probability
        prob = model.predict_ethical_rating(
            row["eco_score"], row["ethics_score"],
            row["carbon_footprint"], row["price"]
        )
        price_inr = int(row["price"] * 83)
        link = _build_product_link(row["name"], row["brand"])
        tags = _make_ethical_tags(row.to_dict())
        tags_html = " ".join(tags) if tags else ""

        # Score bar widths
        eco_w    = int(row["eco_score"] * 10)
        eth_w    = int(row["ethics_score"] * 10)
        comp_w   = int(row["composite_score"] * 10)

        # Carbon indicator
        carbon = row["carbon_footprint"]
        if carbon <= 0.5:
            carbon_color = "#4ade80"
            carbon_label = "Ultra Low"
        elif carbon <= 1.5:
            carbon_color = "#86efac"
            carbon_label = "Low"
        elif carbon <= 3.0:
            carbon_color = "#fbbf24"
            carbon_label = "Moderate"
        else:
            carbon_color = "#f87171"
            carbon_label = "High"

        # Rank medal
        medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
        medal = medals[i] if i < len(medals) else f"#{i+1}"

        delay = i * 80

        cards_html += f"""
<div style="background:#141f17;border:1px solid rgba(74,222,128,0.15);border-radius:14px;
            padding:1rem 1.1rem;margin-bottom:0.75rem;transition:all 0.3s;
            animation:fadeUp 0.4s {delay}ms ease both;">
  <!-- Header -->
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:0.5rem;margin-bottom:0.6rem;">
    <div style="flex:1;">
      <div style="display:flex;align-items:center;gap:0.4rem;margin-bottom:2px;">
        <span style="font-size:1rem;">{medal}</span>
        <a href="{link}" target="_blank" rel="noopener"
           style="font-family:'Playfair Display',serif;font-size:0.95rem;font-weight:700;
                  color:#e8f5ee;text-decoration:none;line-height:1.3;">
          {row["name"]}
          <span style="font-size:0.7rem;color:#4ade80;margin-left:4px;">↗</span>
        </a>
      </div>
      <div style="font-size:0.75rem;color:#8aaa94;">
        by <strong style="color:#a7f3c1;">{row["brand"]}</strong>
        &nbsp;·&nbsp; {row["category"]}
      </div>
    </div>
    <div style="text-align:right;flex-shrink:0;">
      <div style="font-family:'DM Mono',monospace;font-size:1.3rem;font-weight:600;color:#4ade80;line-height:1;">
        {row["composite_score"]}
      </div>
      <div style="font-size:0.65rem;color:#4a6856;">AI Score</div>
    </div>
  </div>

  <!-- Description -->
  <p style="font-size:0.78rem;color:#8aaa94;margin-bottom:0.65rem;line-height:1.5;">
    {row["description"]}
  </p>

  <!-- Tags -->
  <div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:0.65rem;">
    {tags_html}
  </div>

  <!-- Score bars -->
  <div style="display:grid;grid-template-columns:60px 1fr 36px;gap:3px 6px;
              align-items:center;font-size:0.7rem;margin-bottom:0.65rem;">
    <span style="color:#4a6856;">Eco</span>
    <div style="background:rgba(255,255,255,0.05);border-radius:99px;height:4px;overflow:hidden;">
      <div style="width:{eco_w}%;height:100%;background:linear-gradient(90deg,#2d6a4f,#4ade80);border-radius:99px;"></div>
    </div>
    <span style="color:#4ade80;font-family:'DM Mono',monospace;">{row["eco_score"]}</span>

    <span style="color:#4a6856;">Ethics</span>
    <div style="background:rgba(255,255,255,0.05);border-radius:99px;height:4px;overflow:hidden;">
      <div style="width:{eth_w}%;height:100%;background:linear-gradient(90deg,#2d6a4f,#4ade80);border-radius:99px;"></div>
    </div>
    <span style="color:#4ade80;font-family:'DM Mono',monospace;">{row["ethics_score"]}</span>

    <span style="color:#4a6856;">Overall</span>
    <div style="background:rgba(255,255,255,0.05);border-radius:99px;height:4px;overflow:hidden;">
      <div style="width:{comp_w}%;height:100%;background:linear-gradient(90deg,#16a34a,#86efac);border-radius:99px;"></div>
    </div>
    <span style="color:#86efac;font-family:'DM Mono',monospace;">{row["composite_score"]}</span>
  </div>

  <!-- Footer: carbon + price + AI prob -->
  <div style="display:flex;justify-content:space-between;align-items:center;
              padding-top:0.55rem;border-top:1px solid rgba(74,222,128,0.08);">
    <div style="display:flex;align-items:center;gap:0.4rem;font-size:0.72rem;">
      <span style="color:{carbon_color};">💨 {carbon}kg CO₂</span>
      <span style="background:rgba(74,222,128,0.08);color:{carbon_color};
                   padding:1px 6px;border-radius:4px;font-size:0.65rem;font-weight:600;">
        {carbon_label}
      </span>
    </div>
    <div style="display:flex;align-items:center;gap:0.6rem;">
      <span style="font-size:0.7rem;color:#8aaa94;">🤖 {prob}% ethical</span>
      <span style="font-weight:700;color:#e8f5ee;font-size:0.9rem;">₹{price_inr:,}</span>
    </div>
  </div>
</div>"""

    # Follow-up suggestions
    suggestion_html = _build_suggestions(categories, eth_filters)

    return f"""
<div>
  {summary}
  {cards_html}
  {suggestion_html}
</div>"""


def _build_suggestions(categories, eth_filters):
    """Build quick follow-up suggestion chips."""
    suggestions = []
    if categories:
        suggestions.append(f"Show me {categories[0]} under ₹500")
        suggestions.append(f"Best rated {categories[0]} products")
    else:
        suggestions.extend([
            "Show organic food under ₹300",
            "Best eco-friendly personal care",
        ])
    suggestions.append("Zero waste kitchen products")
    suggestions.append("Fair trade certified products")

    chips = "".join([
        f'<button onclick="setInput(this.textContent)" '
        f'style="background:rgba(74,222,128,0.08);border:1px solid rgba(74,222,128,0.2);'
        f'color:#a7f3c1;padding:4px 10px;border-radius:99px;font-size:0.72rem;'
        f'cursor:pointer;transition:all 0.2s;font-family:inherit;"'
        f' onmouseover="this.style.background=\'rgba(74,222,128,0.18)\'"'
        f' onmouseout="this.style.background=\'rgba(74,222,128,0.08)\'">'
        f'{s}</button>'
        for s in suggestions[:4]
    ])

    return f"""
<div style="margin-top:0.75rem;padding-top:0.75rem;border-top:1px solid rgba(74,222,128,0.1);">
  <div style="font-size:0.7rem;color:#4a6856;margin-bottom:0.4rem;">💡 Try asking:</div>
  <div style="display:flex;flex-wrap:wrap;gap:0.4rem;">{chips}</div>
</div>"""


def _greeting_response():
    return """
<div>
  <p style="margin-bottom:0.6rem;">👋 Hello! I'm <strong style="color:#4ade80;">EcoMind</strong> — your AI ethical shopping assistant.</p>
  <p style="color:#8aaa94;font-size:0.85rem;margin-bottom:0.8rem;">
    Tell me what you're looking for and I'll find ethical, eco-friendly products that match. I understand:
  </p>
  <ul style="color:#8aaa94;font-size:0.82rem;line-height:2;padding-left:1.2rem;">
    <li>💰 Budget in ₹ — <em>"under ₹500"</em></li>
    <li>🌿 Ethical filters — <em>"organic", "fair trade", "low carbon"</em></li>
    <li>📦 Categories — <em>"food", "clothing", "personal care"</em></li>
    <li>🔍 Keywords — <em>"bamboo", "recycled", "natural"</em></li>
  </ul>
  <p style="margin-top:0.8rem;font-size:0.82rem;color:#a7f3c1;">
    Try: <em>"I want breakfast below ₹500, organic and certified"</em>
  </p>
</div>"""


def _help_response():
    return """
<div>
  <p style="margin-bottom:0.7rem;font-weight:600;color:#4ade80;">🤖 What I can do:</p>
  <ul style="color:#8aaa94;font-size:0.83rem;line-height:2.1;padding-left:1.2rem;">
    <li>Find ethical products by <strong style="color:#e8f5ee;">budget in ₹</strong></li>
    <li>Filter by <strong style="color:#e8f5ee;">organic, fair trade, vegan, recycled, low carbon</strong></li>
    <li>Search across <strong style="color:#e8f5ee;">12 categories</strong> — Food, Clothing, Kitchen, Home and more</li>
    <li>Show <strong style="color:#e8f5ee;">AI-predicted ethical scores</strong> and carbon footprint</li>
    <li>Provide <strong style="color:#e8f5ee;">direct shopping links</strong> for every product</li>
    <li>Tag certifications: <strong style="color:#e8f5ee;">GOTS, B-Corp, Fair Trade, OEKO-TEX</strong> and more</li>
  </ul>
  <p style="margin-top:0.8rem;font-size:0.8rem;color:#4a6856;">
    Example: "Show me organic food under ₹300 with low carbon footprint"
  </p>
</div>"""


def _no_results_response(query, budget_usd, categories):
    budget_str = f"₹{int(budget_usd * 83)}" if budget_usd else "any budget"
    cat_str = ", ".join(categories) if categories else "any category"

    return f"""
<div>
  <p style="color:#f87171;margin-bottom:0.6rem;">😕 No products matched your query.</p>
  <p style="font-size:0.82rem;color:#8aaa94;margin-bottom:0.7rem;">
    I searched for <em>"{query}"</em> in {cat_str} under {budget_str}.
  </p>
  <p style="font-size:0.82rem;color:#a7f3c1;">Try:</p>
  <ul style="font-size:0.8rem;color:#8aaa94;line-height:1.9;padding-left:1.2rem;">
    <li>A higher budget — e.g. <em>"under ₹1000"</em></li>
    <li>Fewer filters — remove one condition</li>
    <li>A broader category — e.g. <em>"eco-friendly products"</em></li>
  </ul>
</div>"""
