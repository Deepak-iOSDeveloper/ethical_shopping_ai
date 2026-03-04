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
# MASSIVE DATASET — 150+ products, all price ranges
# prices in USD (approx: $1 = ₹83)
# columns: name, brand, category, eco_score, ethics_score,
#          price_usd, sustainability_cert, materials, carbon_footprint, description
# ─────────────────────────────────────────────
PRODUCTS_DATA = [

    # ══════════════════════════════════════════
    # FOOD & BEVERAGES  ($1 – $100)
    # ══════════════════════════════════════════
    ("Fair Trade Coffee 250g",       "Equal Exchange",    "Food", 9.5, 9.8,  6,   "Fair Trade",       "organic_coffee",           0.4, "Direct-trade organic coffee empowering small farmers worldwide."),
    ("Organic Green Tea 100g",       "Vahdam",            "Food", 9.2, 9.0,  5,   "USDA Organic",     "organic_green_tea",         0.2, "Single-origin Darjeeling green tea, certified organic."),
    ("Herbal Tulsi Tea 50g",         "Organic India",     "Food", 9.4, 9.1,  3,   "USDA Organic",     "tulsi,herbs",               0.1, "Sacred tulsi herb tea with calming adaptogenic properties."),
    ("Raw Wildflower Honey 500g",    "Beekeeper Naturals","Food", 9.0, 8.8,  12,  "Non-GMO",          "raw_honey",                 0.3, "Unfiltered raw honey supporting bee population health."),
    ("Organic Jaggery 1kg",          "24 Mantra",         "Food", 9.3, 9.0,  4,   "USDA Organic",     "sugarcane",                 0.2, "Traditional unrefined cane sugar — zero chemicals."),
    ("Cold-Press Coconut Oil 500ml", "Coconut Secret",    "Food", 9.1, 8.7,  10,  "USDA Organic",     "coconut",                   0.3, "Raw cold-pressed coconut oil from small family farms."),
    ("Organic Turmeric Powder 200g", "Frontier Co-op",    "Food", 9.4, 9.2,  5,   "USDA Organic",     "turmeric",                  0.1, "Pure organic turmeric with high curcumin content."),
    ("Fair Trade Dark Chocolate 80g","Alter Eco",         "Food", 9.3, 9.5,  4,   "Fair Trade",       "organic_cacao",             0.5, "Rich 85% dark chocolate from regenerative farms."),
    ("Organic Quinoa 500g",          "Ancient Harvest",   "Food", 9.0, 8.8,  7,   "USDA Organic",     "quinoa",                    0.4, "Complete protein grain from organic Andean farms."),
    ("Organic Chia Seeds 400g",      "Mamma Chia",        "Food", 9.1, 8.9,  8,   "USDA Organic",     "chia_seeds",                0.3, "Nutrient-dense chia seeds from certified organic farms."),
    ("Cold Brew Coffee Bags 10pk",   "Stone Street",      "Food", 8.8, 8.5,  14,  "Fair Trade",       "organic_coffee",            0.6, "Slow-steeped organic cold brew bags for smooth coffee."),
    ("Organic Oat Milk 1L",          "Oatly",             "Food", 9.2, 8.8,  4,   "USDA Organic",     "oats,water",                0.3, "The original oat drink — lower carbon than dairy milk."),
    ("Organic Lentils 1kg",          "Bob's Red Mill",    "Food", 9.0, 8.6,  5,   "USDA Organic",     "lentils",                   0.2, "Non-GMO organic lentils, high protein and fiber."),
    ("Fair Trade Cacao Powder 250g", "Navitas",           "Food", 9.3, 9.4,  9,   "Fair Trade",       "raw_cacao",                 0.4, "Raw ceremonial-grade cacao from Peruvian cooperatives."),
    ("Organic Moringa Powder 100g",  "Kuli Kuli",         "Food", 9.5, 9.6,  10,  "USDA Organic",     "moringa",                   0.1, "Superfood powder supporting women farmers in West Africa."),
    ("Artisan Sourdough Mix 500g",   "King Arthur",       "Food", 8.7, 8.5,  6,   "Non-GMO",          "organic_wheat",             0.3, "Non-GMO flour blend for perfect sourdough at home."),
    ("Organic Almond Butter 400g",   "Justin's",          "Food", 9.0, 8.7,  12,  "USDA Organic",     "organic_almonds",           0.5, "Classic almond butter from organically grown almonds."),
    ("Organic Spice Kit 6-pack",     "Simply Organic",    "Food", 9.2, 9.0,  15,  "USDA Organic",     "mixed_spices",              0.2, "Six essential organic spices in refillable glass jars."),
    ("Natural Energy Bar 12pk",      "Larabar",           "Food", 8.9, 8.6,  18,  "Non-GMO",          "dates,nuts",                0.4, "Minimal-ingredient bars — just dates, nuts, and fruits."),
    ("Organic Apple Cider Vinegar",  "Bragg",             "Food", 9.1, 8.8,  7,   "USDA Organic",     "organic_apples",            0.2, "Raw unfiltered ACV with the mother, in glass bottle."),
    ("Fairtrade Basmati Rice 2kg",   "Tilda",             "Food", 8.8, 9.0,  8,   "Fair Trade",       "basmati_rice",              0.5, "Aged Himalayan basmati rice from fair trade farms."),
    ("Organic Coconut Sugar 500g",   "Big Tree Farms",    "Food", 9.3, 9.1,  6,   "USDA Organic",     "coconut_sap",               0.2, "Low-GI coconut sugar from sustainable Indonesian farms."),
    ("Regenerative Olive Oil 500ml", "California Olive",  "Food", 9.0, 8.7,  18,  "Non-GMO",          "olives",                    0.6, "Extra virgin olive oil from regenerative California farms."),
    ("Natural Protein Powder 500g",  "Sunwarrior",        "Food", 9.1, 8.8,  20,  "USDA Organic",     "organic_pea,rice_protein",  0.5, "Plant-based protein powder, no artificial sweeteners."),
    ("Organic Moringa Tea 20pk",     "Organic India",     "Food", 9.3, 9.0,  5,   "USDA Organic",     "moringa,tulsi",             0.1, "Energising moringa and tulsi herbal tea blend."),

    # ══════════════════════════════════════════
    # PERSONAL CARE  ($3 – $150)
    # ══════════════════════════════════════════
    ("Bamboo Toothbrush",            "Brush with Bamboo", "Personal Care", 9.6, 9.1,  3,  "B-Corp",          "bamboo",                   0.05, "100% biodegradable bamboo toothbrush handle."),
    ("Charcoal Toothpaste 100ml",    "Hello Products",    "Personal Care", 9.0, 8.7,  7,  "USDA Organic",    "activated_charcoal",        0.1,  "Fluoride-free charcoal toothpaste in recyclable tube."),
    ("Refillable Deodorant",         "Wild",              "Personal Care", 9.3, 8.9,  12, "COSMOS",           "natural_minerals",          0.1,  "Natural mineral deodorant with compostable refills."),
    ("Organic Shampoo Bar",          "Lush",              "Personal Care", 8.8, 8.5,  14, "COSMOS",           "organic_botanicals",        0.1,  "Zero-waste solid shampoo bar for all hair types."),
    ("Conditioner Bar",              "Ethique",           "Personal Care", 9.4, 9.2,  15, "B-Corp",           "cocoa_butter,botanicals",   0.1,  "Concentrated conditioner bar — replaces 3 plastic bottles."),
    ("Natural Face Moisturizer 50ml","Weleda",            "Personal Care", 9.1, 8.9,  22, "COSMOS",           "rose_oil,shea_butter",      0.2,  "Award-winning rose face cream with biodynamic ingredients."),
    ("Organic Lip Balm",             "Burt's Bees",       "Personal Care", 8.9, 8.6,  4,  "USDA Organic",    "beeswax,vitamin_e",         0.05, "Classic beeswax lip balm with natural moisturizers."),
    ("Zero-Waste Face Scrub 100g",   "Ethique",           "Personal Care", 9.3, 9.1,  18, "B-Corp",           "oatmeal,clay",              0.15, "Gentle exfoliating face scrub in compostable packaging."),
    ("Organic Sunscreen SPF50 100ml","Raw Elements",      "Personal Care", 9.2, 9.0,  28, "USDA Organic",    "zinc_oxide,beeswax",        0.2,  "Reef-safe mineral sunscreen in recycled tin."),
    ("Ayurvedic Hair Oil 150ml",     "Forest Essentials", "Personal Care", 9.0, 8.8,  20, "USDA Organic",    "coconut,herbs",             0.15, "Traditional Ayurvedic hair oil with 25 herbs."),
    ("Organic Body Lotion 200ml",    "Dr. Bronner's",     "Personal Care", 9.3, 9.4,  16, "USDA Organic",    "hemp_oil,coconut_oil",      0.2,  "Fair trade organic body lotion in 100% post-consumer plastic."),
    ("Natural Perfume 30ml",         "Phlur",             "Personal Care", 8.8, 8.5,  85, "COSMOS",           "essential_oils,alcohol",    0.3,  "Clean fine fragrance with transparent ingredient sourcing."),
    ("Menstrual Cup",                "DivaCup",           "Personal Care", 9.7, 9.2,  30, "FDA Approved",    "medical_silicone",          0.1,  "Reusable menstrual cup — eliminates thousands of disposables."),
    ("Bamboo Cotton Rounds 20pk",    "LastObject",        "Personal Care", 9.5, 9.1,  18, "GOTS",             "bamboo,organic_cotton",     0.1,  "Reusable makeup remover pads, washable 1000+ times."),
    ("Natural Castile Soap 946ml",   "Dr. Bronner's",     "Personal Care", 9.4, 9.5,  18, "USDA Organic",    "hemp_oil,olive_oil",        0.3,  "Multi-use pure-castile soap for body, hair, and home."),
    ("Organic Rose Water 200ml",     "Heritage Store",    "Personal Care", 9.0, 8.7,  12, "USDA Organic",    "rose_petals,water",         0.1,  "Pure Bulgarian rose water toner for all skin types."),
    ("Solid Face Wash Bar",          "Lush",              "Personal Care", 8.9, 8.6,  12, "COSMOS",           "clay,tea_tree",             0.1,  "Packaging-free solid face cleanser for oily skin."),
    ("Organic Argan Oil 30ml",       "Josie Maran",       "Personal Care", 9.2, 8.8,  35, "USDA Organic",    "argan_oil",                 0.2,  "100% pure organic argan oil for face, hair, and body."),
    ("Refillable Perfume 50ml",      "Abel",              "Personal Care", 9.0, 8.7,  120,"COSMOS",           "natural_essences",          0.4,  "Natural perfume in refillable glass bottle — zero waste."),
    ("Bamboo Razor",                 "Leaf Shave",        "Personal Care", 9.3, 9.0,  32, "B-Corp",           "bamboo,stainless_steel",    0.2,  "Reusable bamboo razor with recyclable stainless blades."),
    ("Organic Body Wash 400ml",      "Everyone",          "Personal Care", 9.1, 8.9,  12, "USDA Organic",    "coconut_oil,aloe",          0.2,  "Plant-based body wash in 100% recycled bottle."),
    ("Natural Neem Toothpaste",      "Himalaya",          "Personal Care", 9.0, 8.8,  4,  "USDA Organic",    "neem,clove,organic_mint",   0.1,  "Ayurvedic neem toothpaste — antibacterial and whitening."),
    ("Organic Face Serum 30ml",      "Pai Skincare",      "Personal Care", 9.2, 9.0,  55, "COSMOS",           "rosehip_oil,organic_herbs", 0.2,  "Award-winning organic rosehip serum for sensitive skin."),

    # ══════════════════════════════════════════
    # CLOTHING  ($10 – $600)
    # ══════════════════════════════════════════
    ("Organic Cotton Tee",           "Patagonia",         "Clothing", 9.2, 9.5,  45,  "GOTS",            "organic_cotton",            2.1,  "Classic fit organic tee made with 100% GOTS-certified cotton."),
    ("Recycled Fleece Jacket",       "Patagonia",         "Clothing", 9.0, 9.5,  149, "Fair Trade",      "recycled_polyester",        3.2,  "Warm fleece jacket made from 100% recycled plastic bottles."),
    ("Merino Wool Sweater",          "Eileen Fisher",     "Clothing", 8.5, 8.8,  178, "GOTS",            "merino_wool",               4.1,  "Responsibly sourced merino wool sweater."),
    ("Tencel Dress",                 "Eileen Fisher",     "Clothing", 8.7, 8.9,  195, "OEKO-TEX",        "tencel",                    2.9,  "Silky Tencel dress from certified sustainable sources."),
    ("Organic Linen Shirt",          "Thought Clothing",  "Clothing", 8.6, 8.4,  75,  "GOTS",            "linen",                     2.4,  "Breathable organic linen shirt, ideal for warm weather."),
    ("Bamboo PJs",                   "Thought Clothing",  "Clothing", 8.4, 8.3,  65,  "OEKO-TEX",        "bamboo",                    1.6,  "Soft bamboo pajamas, naturally antibacterial."),
    ("Recycled Denim Jeans",         "Nudie Jeans",       "Clothing", 8.9, 9.0,  155, "Fair Trade",      "recycled_denim",            4.5,  "Iconic denim made with 20% recycled cotton."),
    ("Organic Denim Jacket",         "Nudie Jeans",       "Clothing", 8.7, 9.0,  210, "GOTS",            "organic_cotton",            5.1,  "Classic denim jacket with organic cotton and fair trade production."),
    ("Hemp Graphic Tee",             "Outerknown",        "Clothing", 9.0, 8.8,  48,  "Fair Trade",      "hemp,organic_cotton",       1.8,  "Ultra-soft hemp-cotton blend tee with artistic print."),
    ("Recycled Puffer Jacket",       "The North Face",    "Clothing", 8.8, 8.5,  220, "Bluesign",        "recycled_down",             4.0,  "Warm puffer jacket insulated with recycled down fill."),
    ("Organic Yoga Leggings",        "Prana",             "Clothing", 9.1, 8.7,  89,  "Fair Trade",      "organic_cotton,spandex",    2.0,  "High-waist yoga leggings in organic cotton blend."),
    ("Bamboo Socks 3-pack",          "Thought Clothing",  "Clothing", 9.0, 8.6,  22,  "OEKO-TEX",        "bamboo",                    0.5,  "Softly padded bamboo socks — naturally odour resistant."),
    ("Organic Cotton Hoodie",        "Tentree",           "Clothing", 9.2, 9.3,  90,  "B-Corp",          "organic_cotton",            3.0,  "Cosy hoodie — every purchase plants 10 trees."),
    ("Fair Trade Linen Pants",       "Pact",              "Clothing", 9.1, 9.0,  65,  "Fair Trade",      "linen",                     2.2,  "Relaxed linen trousers made in fair-trade factories."),
    ("Upcycled Patchwork Shirt",     "Patagonia",         "Clothing", 9.5, 9.5,  99,  "Fair Trade",      "upcycled_fabric",           1.0,  "One-of-a-kind shirt made from fabric offcuts."),
    ("Organic Merino Base Layer",    "Icebreaker",        "Clothing", 9.1, 8.8,  120, "ZQ Merino",       "merino_wool",               3.5,  "Temperature-regulating merino base layer for all seasons."),
    ("Recycled Swim Shorts",         "Patagonia",         "Clothing", 9.0, 9.3,  65,  "Fair Trade",      "recycled_nylon",            1.5,  "Quick-dry swim shorts made from recycled fishing nets."),
    ("Plant-Dyed Linen Scarf",       "Maiwa",             "Clothing", 9.4, 9.2,  55,  "GOTS",            "linen,natural_dyes",        1.2,  "Hand-crafted scarf using traditional plant dyeing."),
    ("Organic Denim Overalls",       "Wrangler Sustainable","Clothing",8.8,8.6, 99,  "GOTS",            "organic_denim",             4.8,  "Classic dungarees in certified organic denim."),
    ("Tencel Work Blazer",           "Amour Vert",        "Clothing", 9.0, 8.9,  195, "OEKO-TEX",        "tencel",                    3.0,  "Sharp work blazer in sustainable Tencel fabric."),
    ("Hemp Jogger Pants",            "Hemp Tailor",       "Clothing", 9.3, 9.0,  72,  "GOTS",            "hemp",                      1.8,  "Comfortable hemp joggers, naturally antimicrobial."),
    ("Fair Trade Cashmere Sweater",  "Naadam",            "Clothing", 8.9, 9.1,  165, "Fair Trade",      "cashmere",                  4.5,  "Luxuriously soft cashmere sweater at a fair price."),
    ("Organic Cotton Kurta",         "FabIndia",          "Clothing", 9.0, 8.8,  35,  "GOTS",            "organic_cotton",            2.0,  "Handcrafted organic cotton kurta supporting Indian artisans."),
    ("Khadi Cotton Shirt",           "Khadi Gramodyog",   "Clothing", 9.5, 9.6,  20,  "Khadi Mark",      "khadi_cotton",              0.8,  "Hand-spun hand-woven khadi — the most sustainable fabric."),
    ("Organic Saree 6m",             "GoCoop",            "Clothing", 9.4, 9.5,  55,  "GOTS",            "organic_cotton",            1.5,  "Handloom organic cotton saree from weaver cooperatives."),
    ("Natural Indigo Kurta",         "Anokhi",            "Clothing", 9.3, 9.2,  45,  "GOTS",            "cotton,natural_dyes",       1.8,  "Block-printed kurta with natural indigo dye."),
    ("Organic Cotton Kurta Set",     "FabIndia",          "Clothing", 9.0, 8.9,  50,  "GOTS",            "organic_cotton",            2.2,  "Coordinated kurta and palazzo set in organic cotton."),
    ("Bamboo Polo Shirt",            "Thought Clothing",  "Clothing", 9.1, 8.7,  55,  "OEKO-TEX",        "bamboo",                    1.4,  "Breathable bamboo polo — naturally moisture wicking."),
    ("Sustainable Wool Coat",        "Naadam",            "Clothing", 9.0, 9.1,  395, "Fair Trade",      "recycled_wool",             6.0,  "Classic long coat in recycled Mongolian wool."),
    ("Organic Linen Jumpsuit",       "Eileen Fisher",     "Clothing", 8.9, 8.8,  265, "GOTS",            "linen",                     3.2,  "Effortless wide-leg jumpsuit in organic linen."),

    # ══════════════════════════════════════════
    # FOOTWEAR  ($15 – $250)
    # ══════════════════════════════════════════
    ("Wool Runners",                 "Allbirds",          "Footwear", 9.1, 8.7,  110, "B-Corp",          "merino_wool",               2.3,  "The world's most comfortable shoes from natural merino wool."),
    ("Tree Runners",                 "Allbirds",          "Footwear", 8.8, 8.7,  98,  "B-Corp",          "eucalyptus",                1.9,  "Lightweight shoes made from eucalyptus tree fiber."),
    ("Plant-Based Sneakers",         "Veja",              "Footwear", 9.2, 9.3,  140, "Fair Trade",      "organic_cotton,rubber",     2.6,  "Fair trade sneakers with Amazonian rubber soles."),
    ("Leather-Free Boots",           "Veja",              "Footwear", 8.6, 9.1,  185, "Fair Trade",      "corn_waste_leather",        3.0,  "Stylish boots using innovative corn-waste bio-leather."),
    ("Cork Sandals",                 "Birkenstock",       "Footwear", 8.9, 8.6,  120, "B-Corp",          "cork,natural_leather",      2.5,  "Classic cork-footbed sandals made to last decades."),
    ("Recycled Canvas Sneaker",      "Nothing New",       "Footwear", 9.1, 8.9,  95,  "B-Corp",          "recycled_plastic",          1.8,  "Minimalist sneaker made from 100% recycled materials."),
    ("Hemp Slip-Ons",                "Sanuk",             "Footwear", 8.8, 8.5,  60,  "Non-GMO",         "hemp",                      1.5,  "Casual hemp slip-on shoes with yoga mat insole."),
    ("Natural Rubber Sandals",       "Vivobarefoot",      "Footwear", 9.0, 8.8,  150, "B-Corp",          "natural_rubber",            2.0,  "Barefoot sandals from sustainably tapped rubber trees."),
    ("Recycled Hiking Boot",         "Timberland",        "Footwear", 8.7, 8.5,  180, "B-Corp",          "recycled_plastic,rubber",   4.0,  "Durable hiking boot with ReBOTL recycled plastic upper."),
    ("Upcycled Leather Loafers",     "Rothy's",           "Footwear", 9.2, 9.0,  165, "B-Corp",          "recycled_bottles",          2.2,  "Machine-washable loafers knit from recycled plastic bottles."),
    ("Bamboo Flip Flops",            "Indosole",          "Footwear", 9.0, 8.8,  45,  "B-Corp",          "recycled_tires,bamboo",     1.0,  "Flip flops with soles made from recycled motorcycle tires."),
    ("Fair Trade Kolhapuri Chappal", "Jaypore",           "Footwear", 9.3, 9.2,  25,  "Fair Trade",      "vegetable_tanned_leather",  1.5,  "Handcrafted Kolhapuri sandals by fair-trade artisans."),
    ("Vegan Leather Chelsea Boots",  "Will's Vegan",      "Footwear", 9.1, 9.0,  145, "PETA Approved",   "bio_oil_leather",           2.8,  "Premium vegan Chelsea boots from Italian bio-based leather."),
    ("Organic Cotton Espadrilles",   "Toms",              "Footwear", 8.8, 9.0,  55,  "Fair Trade",      "organic_cotton,natural_jute",1.5, "Classic espadrilles — every pair supports a community initiative."),
    ("Recycled Wool Slippers",       "Allbirds",          "Footwear", 9.0, 8.7,  85,  "B-Corp",          "merino_wool",               1.5,  "Cloud-soft slippers from ZQ-certified merino wool."),

    # ══════════════════════════════════════════
    # SPORTSWEAR  ($15 – $150)
    # ══════════════════════════════════════════
    ("Bamboo Activewear Set",        "Allbirds",          "Sportswear", 8.9, 8.6,  95,  "B-Corp",         "bamboo",                   1.5,  "Ultra-soft bamboo activewear for yoga and running."),
    ("Recycled Yoga Pants",          "Girlfriend Collective","Sportswear",9.0,8.8,68,  "OEKO-TEX",       "recycled_bottles",         1.8,  "Made from 25 recycled plastic bottles per pair."),
    ("Organic Running Shorts",       "Prana",             "Sportswear", 9.0, 8.7,  55,  "Fair Trade",     "organic_cotton",           1.4,  "Lightweight organic cotton running shorts."),
    ("Recycled Sports Bra",          "Girlfriend Collective","Sportswear",9.1,8.9,52,  "OEKO-TEX",       "recycled_nylon",           1.0,  "High-impact sports bra from recycled fishing nets."),
    ("Hemp Gym Tank",                "Patagonia",         "Sportswear", 9.2, 9.3,  45,  "Fair Trade",     "hemp,recycled_polyester",  1.2,  "Breathable hemp gym tank, moisture-wicking."),
    ("Merino Running Socks 2pk",     "Darn Tough",        "Sportswear", 9.0, 8.8,  28,  "Bluesign",       "merino_wool",              0.6,  "Lifetime-guaranteed merino wool running socks."),
    ("Recycled Swim Bikini",         "Vitamin A",         "Sportswear", 8.9, 8.7,  110, "OEKO-TEX",       "recycled_nylon",           1.5,  "Sustainable swimwear from regenerated ocean nylon."),
    ("Organic Track Jacket",         "Adidas by Stella",  "Sportswear", 8.8, 8.5,  130, "GOTS",           "organic_cotton",           3.0,  "Iconic track jacket in certified organic cotton."),
    ("Recycled Compression Tights",  "2XU",               "Sportswear", 8.7, 8.4,  85,  "Bluesign",       "recycled_nylon,spandex",   2.0,  "Performance compression tights from recycled materials."),
    ("Natural Latex Swim Cap",       "Finis",             "Sportswear", 8.8, 8.6,  18,  "Non-GMO",        "natural_latex",            0.3,  "Soft natural latex swim cap, biodegradable."),
    ("Organic Cotton Gym Towel",     "Coyuchi",           "Sportswear", 9.0, 8.8,  28,  "GOTS",           "organic_cotton",           0.6,  "Quick-dry organic cotton gym towel — no microplastics."),
    ("Recycled Windbreaker",         "Patagonia",         "Sportswear", 9.1, 9.2,  149, "Fair Trade",     "recycled_nylon",           2.5,  "Ultralight windbreaker from recycled nylon, stuffs into pocket."),

    # ══════════════════════════════════════════
    # ACCESSORIES  ($8 – $200)
    # ══════════════════════════════════════════
    ("Hemp Backpack",                "Patagonia",         "Accessories", 8.8, 9.2, 89,  "B-Corp",         "hemp",                     1.8,  "Durable hemp backpack with recycled lining."),
    ("Solar Backpack",               "Voltaic Systems",   "Accessories", 9.0, 8.2, 200, "B-Corp",         "recycled_pet",             2.0,  "Solar-powered backpack to charge devices on the go."),
    ("Organic Tote Bag",             "Baggu",             "Accessories", 8.5, 8.0, 28,  "GOTS",           "organic_cotton",           0.8,  "Minimalist everyday tote in certified organic cotton."),
    ("Recycled Sunglasses",          "Pela",              "Accessories", 9.0, 8.5, 95,  "B-Corp",         "flaxseed_compound",        0.6,  "Plant-based, compostable sunglasses with UV protection."),
    ("Cork Wallet",                  "Corkor",            "Accessories", 9.4, 9.1, 35,  "Vegan Society",  "cork",                     0.3,  "Slim RFID-blocking wallet made from natural cork bark."),
    ("Recycled Plastic Sunglasses",  "Solo Eyewear",      "Accessories", 9.0, 9.2, 65,  "B-Corp",         "recycled_plastic",         0.5,  "Eyewear made from recycled plastic — funds eye surgery."),
    ("Natural Rubber Watch Strap",   "Votch",             "Accessories", 8.8, 8.6, 45,  "PETA Approved",  "natural_rubber",           0.4,  "Vegan watch strap made from natural rubber."),
    ("Upcycled Tire Belt",           "Alchemy Goods",     "Accessories", 9.2, 8.8, 40,  "B-Corp",         "recycled_tires",           0.2,  "Unique belt made from upcycled bicycle inner tubes."),
    ("Organic Cotton Canvas Hat",    "Prana",             "Accessories", 8.9, 8.7, 35,  "GOTS",           "organic_cotton",           0.6,  "Classic canvas hat in organic cotton."),
    ("Recycled Nylon Fanny Pack",    "Baggu",             "Accessories", 8.7, 8.3, 38,  "Bluesign",       "recycled_nylon",           0.5,  "Hands-free fanny pack from recycled nylon."),
    ("Bamboo Umbrella",              "Fox Umbrellas",     "Accessories", 8.9, 8.5, 80,  "FSC",            "bamboo,recycled_canopy",   1.0,  "Classic bamboo-handle umbrella built to last a lifetime."),
    ("Fair Trade Jute Tote",         "Jute Works",        "Accessories", 9.5, 9.6, 12,  "Fair Trade",     "jute",                     0.3,  "Handwoven jute tote from fair-trade Bangladesh co-op."),
    ("Vegan Leather Laptop Bag",     "Matt & Nat",        "Accessories", 9.0, 8.8, 155, "PETA Approved",  "recycled_bottles,cork",    2.5,  "Premium laptop bag lined with recycled plastic bottles."),
    ("Hemp Yoga Bag",                "Rawganique",        "Accessories", 9.3, 9.0, 65,  "GOTS",           "hemp",                     0.8,  "Organic hemp yoga bag with natural hemp rope drawstring."),

    # ══════════════════════════════════════════
    # KITCHEN  ($5 – $150)
    # ══════════════════════════════════════════
    ("Beeswax Wraps Set 3pk",        "Beeswax Wrap Co",   "Kitchen", 9.5, 9.0, 22,  "Certified Organic","beeswax,cotton",           0.3,  "Reusable beeswax food wraps — replaces hundreds of plastic bags."),
    ("Bamboo Cutlery Set",           "To-Go Ware",        "Kitchen", 9.4, 8.6, 18,  "FSC",              "bamboo",                   0.2,  "Portable bamboo cutlery set to ditch single-use plastic."),
    ("Recycled Glass Bottle 750ml",  "S'well",            "Kitchen", 9.1, 8.3, 45,  "B-Corp",           "recycled_glass",           0.5,  "Insulated recycled glass water bottle."),
    ("Stainless Steel Straw Set",    "FinalStraw",        "Kitchen", 9.5, 9.1, 25,  "B-Corp",           "stainless_steel",          0.3,  "Collapsible reusable straw — fits any cup or bottle."),
    ("Organic Cotton Dish Cloths 5pk","Swedish Dishcloth","Kitchen", 9.3, 8.8, 20,  "GOTS",             "organic_cotton,cellulose", 0.2,  "Natural dish cloths replacing 17 rolls of paper towels."),
    ("Bamboo Chopping Board",        "Bambu",             "Kitchen", 9.4, 9.0, 38,  "FSC",              "bamboo",                   0.6,  "Organic bamboo chopping board — naturally antibacterial."),
    ("Compostable Trash Bags 30pk",  "If You Care",       "Kitchen", 9.2, 8.9, 12,  "Certified Compostable","cornstarch",          0.2,  "100% compostable certified kitchen waste bags."),
    ("Glass Food Storage Set 4pk",   "Weck",              "Kitchen", 9.0, 8.6, 55,  "B-Corp",           "borosilicate_glass",       1.0,  "Airtight glass jars for zero-waste food storage."),
    ("Natural Coconut Scrubber",     "Full Circle",       "Kitchen", 9.5, 9.1, 8,   "Non-GMO",          "coconut_coir",             0.1,  "Biodegradable dish scrubber made from coconut husks."),
    ("Cast Iron Skillet 10 inch",    "Lodge",             "Kitchen", 8.8, 8.5, 35,  "Made in USA",      "cast_iron",                3.0,  "Pre-seasoned cast iron skillet that lasts generations."),
    ("Organic Beeswax Candles 2pk",  "Big Dipper",        "Kitchen", 9.3, 9.0, 18,  "USDA Organic",     "beeswax,cotton_wick",      0.2,  "Pure beeswax candles — air purifying, no toxins."),
    ("Stainless Steel Lunch Box",    "LunchBots",         "Kitchen", 9.3, 8.8, 28,  "B-Corp",           "stainless_steel",          0.8,  "Leak-proof stainless steel lunch container — lasts forever."),
    ("Organic Cotton Napkins 6pk",   "Coyuchi",           "Kitchen", 9.1, 8.9, 32,  "GOTS",             "organic_cotton",           0.5,  "Soft organic cotton cloth napkins to replace paper."),
    ("Reusable Produce Bags 8pk",    "Bag Again",         "Kitchen", 9.4, 9.0, 15,  "GOTS",             "organic_cotton_mesh",      0.2,  "Lightweight organic mesh bags for zero-waste grocery shopping."),
    ("Stainless Insulated Mug 350ml","Hydro Flask",       "Kitchen", 9.0, 8.6, 32,  "B-Corp",           "stainless_steel",          1.0,  "Keep drinks hot for 12hrs, cold for 24hrs — zero waste."),
    ("Natural Loofah Sponge 3pk",    "Aqua Eden",         "Kitchen", 9.5, 9.0, 6,   "USDA Organic",     "natural_loofah",           0.05, "100% natural loofah — grows in 90 days, fully biodegradable."),
    ("Ceramic Filter Water Jug",     "Berkey",            "Kitchen", 9.1, 8.7, 65,  "B-Corp",           "ceramic,stainless_steel",  1.0,  "Gravity-fed ceramic water filter — no electricity needed."),

    # ══════════════════════════════════════════
    # HOME  ($8 – ₹80,000 equivalent)
    # ══════════════════════════════════════════
    ("Linen Bedding Set",            "Cultiver",          "Home", 8.7, 8.6, 220,  "OEKO-TEX",        "linen",                    3.5,  "Stonewashed linen bedding for breathable, sustainable sleep."),
    ("Organic Cotton Towels 2pk",    "Coyuchi",           "Home", 9.0, 8.8, 85,   "GOTS",            "organic_cotton",           2.1,  "Plush organic cotton bath towels, no toxic dyes."),
    ("Recycled Wool Blanket",        "Pendleton",         "Home", 8.6, 8.4, 165,  "Oeko-Tex",        "recycled_wool",            4.0,  "Classic woven blanket from recycled wool fibers."),
    ("Organic Cotton Pillow",        "Avocado Green",     "Home", 9.2, 9.0, 79,   "GOTS",            "organic_cotton,kapok",     1.5,  "Adjustable pillow with organic cotton and kapok fill."),
    ("Natural Latex Mattress",       "Avocado Green",     "Home", 9.5, 9.3, 1400, "GOLS",            "natural_latex,wool",       15.0, "GOLS-certified latex mattress — no off-gassing, lasts 20 years."),
    ("Bamboo Bed Sheet Set",         "Ettitude",          "Home", 9.3, 8.9, 130,  "OEKO-TEX",        "bamboo_lyocell",           2.0,  "Silky bamboo lyocell sheets — softer than Egyptian cotton."),
    ("Hemp Door Mat",                "Rawganique",        "Home", 9.3, 9.0, 45,   "GOTS",            "hemp",                     0.8,  "Durable natural hemp doormat, fully biodegradable."),
    ("Soy Wax Candle",               "P.F. Candle Co",    "Home", 8.9, 8.6, 22,   "Non-GMO",         "soy_wax,cotton_wick",      0.2,  "Clean-burning soy wax candle with cotton wick."),
    ("Recycled Glass Vase",          "West Elm",          "Home", 8.7, 8.3, 35,   "B-Corp",          "recycled_glass",           0.8,  "Handcrafted vase from 100% recycled glass."),
    ("Organic Cotton Curtains",      "Coyuchi",           "Home", 8.9, 8.8, 145,  "GOTS",            "organic_cotton",           3.0,  "Soft undyed organic cotton curtains, free of synthetic dyes."),
    ("Natural Wool Rug 120x180",     "The Citizenry",     "Home", 9.0, 9.2, 380,  "Fair Trade",      "undyed_wool",              5.0,  "Hand-loomed fair-trade wool rug in natural undyed wool."),
    ("Solar Window Charger",         "Window Socket",     "Home", 9.2, 8.6, 35,   "B-Corp",          "silicon_solar_panel",      0.3,  "Portable solar panel that sticks to any window."),
    ("Hemp Shower Curtain",          "Rawganique",        "Home", 9.4, 9.1, 85,   "GOTS",            "hemp",                     1.2,  "Mold-resistant hemp shower curtain, no plastic liner needed."),
    ("Recycled Cotton Throw",        "H&M Conscious",     "Home", 8.5, 7.8, 45,   "OEKO-TEX",        "recycled_cotton",          2.0,  "Cozy throw blanket knit from recycled cotton yarn."),
    ("Natural Sisal Basket Set 3pk", "Seagrass",          "Home", 9.3, 9.4, 55,   "Fair Trade",      "sisal",                    0.6,  "Handwoven fair-trade sisal baskets for stylish storage."),
    ("Bamboo Toothbrush Holder",     "Bambu",             "Home", 9.3, 8.9, 15,   "FSC",             "bamboo",                   0.2,  "Minimal bamboo toothbrush holder for zero-waste bathroom."),
    ("Organic Beeswax Furniture Polish","Howard Products","Home", 9.0, 8.7, 18,   "USDA Organic",    "beeswax,carnauba_wax",     0.2,  "Natural beeswax furniture polish — no harmful VOCs."),

    # ══════════════════════════════════════════
    # SPORTS & OUTDOORS  ($6 – $500)
    # ══════════════════════════════════════════
    ("Cork Yoga Mat",                "Manduka",           "Sports", 9.3, 8.5, 88,  "B-Corp",          "cork,natural_rubber",      1.2,  "Natural cork yoga mat — antimicrobial and sustainable."),
    ("Solar Lantern Camping",        "BioLite",           "Sports", 9.2, 8.7, 60,  "B-Corp",          "recycled_plastic",         0.4,  "Solar-charged LED lantern — perfect for camping off-grid."),
    ("Recycled Trekking Poles",      "Black Diamond",     "Sports", 8.8, 8.4, 140, "Bluesign",        "recycled_aluminum",        2.5,  "Lightweight trekking poles with recycled aluminum shafts."),
    ("Organic Beeswax Surfboard Wax","Mrs. Palmer's",     "Sports", 9.1, 8.8, 8,   "USDA Organic",    "beeswax",                  0.1,  "100% natural surfboard wax — beeswax and tree resin."),
    ("Recycled Sleeping Bag",        "Marmot",            "Sports", 8.9, 8.5, 220, "Bluesign",        "recycled_down",            4.0,  "Warm sleeping bag with recycled down insulation."),
    ("Hemp Hammock",                 "Eagles Nest",       "Sports", 9.1, 8.7, 70,  "GOTS",            "hemp",                     1.0,  "Lightweight hemp hammock — stronger than cotton, eco-grown."),
    ("Solar Camping Shower",         "Advanced Elements", "Sports", 9.0, 8.5, 25,  "B-Corp",          "recycled_pvc",             0.3,  "Solar-heated camping shower bag — no fossil fuels needed."),
    ("Bamboo Trekking Staff",        "Boo Bamboo",        "Sports", 9.5, 9.0, 35,  "FSC",             "bamboo",                   0.5,  "Lightweight and strong bamboo hiking staff."),
    ("Organic Cotton Climbing Chalk","Metolius",          "Sports", 8.8, 8.6, 12,  "USDA Organic",    "magnesium_carbonate",      0.1,  "Pure organic climbing chalk, no anti-caking additives."),
    ("Recycled Wetsuit",             "Patagonia",         "Sports", 9.2, 9.4, 450, "Fair Trade",      "yulex_natural_rubber",     3.5,  "Yulex natural rubber wetsuit — no petroleum-based neoprene."),
    ("Fair Trade Football",          "Fair Trade Sports", "Sports", 9.4, 9.8, 28,  "Fair Trade",      "natural_rubber,leather",   0.8,  "Hand-stitched fair trade football — no child labour."),
    ("Natural Rubber Resistance Bands","Sanctband",       "Sports", 9.0, 8.7, 18,  "Non-GMO",         "natural_rubber",           0.2,  "Latex-free natural rubber resistance bands set."),
    ("Recycled Foam Yoga Block 2pk", "Manduka",           "Sports", 9.1, 8.5, 28,  "B-Corp",          "recycled_foam",            0.3,  "Supportive yoga blocks from recycled foam — no off-gassing."),
    ("Organic Cotton Jump Rope",     "Buddy Lee",         "Sports", 8.8, 8.6, 22,  "GOTS",            "organic_cotton,wood",      0.2,  "Classic jump rope with organic cotton cord and wood handles."),

    # ══════════════════════════════════════════
    # ELECTRONICS & TECH  ($18 – $900)
    # ══════════════════════════════════════════
    ("Solar Phone Charger 10000mAh", "Anker",             "Electronics", 9.0, 8.4, 45,  "B-Corp",        "recycled_plastic,solar",  1.0,  "Dual-panel solar power bank — charge anywhere off-grid."),
    ("Recycled Laptop Sleeve 15in",  "Patagonia",         "Electronics", 9.1, 9.2, 65,  "Fair Trade",    "recycled_fleece",         0.8,  "Protective laptop sleeve in recycled fleece."),
    ("Bamboo Wireless Charger",      "Bambooee",          "Electronics", 9.2, 8.7, 28,  "FSC",           "bamboo,recycled_plastic", 0.4,  "Qi wireless charger with bamboo surface — no plastic."),
    ("Refurbished Smartphone",       "Back Market",       "Electronics", 9.3, 8.6, 250, "B-Corp",        "refurbished_electronics", 10.0, "Certified refurbished phone — 91% less carbon than new."),
    ("Energy Monitor Plug",          "Emporia",           "Electronics", 8.9, 8.5, 25,  "Energy Star",   "recycled_plastic",        0.3,  "Real-time energy monitor to reduce household electricity."),
    ("Solar Garden Lights 8pk",      "URPOWER",           "Electronics", 9.0, 8.3, 22,  "Non-GMO",       "recycled_plastic,solar",  0.5,  "Waterproof solar garden lights — zero running cost."),
    ("Eco Phone Case",               "Pela",              "Electronics", 9.3, 9.0, 45,  "B-Corp",        "flaxseed_compound",       0.3,  "World's first compostable phone case — 100% biodegradable."),
    ("Recycled Bluetooth Speaker",   "House of Marley",   "Electronics", 9.0, 8.7, 80,  "B-Corp",        "recycled_plastic,bamboo", 2.0,  "Bluetooth speaker made from recycled plastics and bamboo."),
    ("Fairphone 5",                  "Fairphone",         "Electronics", 9.5, 9.8, 700, "Fair Trade",    "conflict_free_minerals",  15.0, "World's most ethical smartphone — modular, repairable, fair trade."),
    ("Solar Laptop Charger 60W",     "Voltaic Systems",   "Electronics", 9.2, 8.5, 200, "B-Corp",        "recycled_plastic,solar",  1.5,  "Powerful solar charger capable of charging any laptop."),
    ("Recycled Plastic Keyboard",    "Logitech",          "Electronics", 8.8, 8.2, 50,  "EPEAT Gold",    "recycled_plastic",        0.8,  "Full-size wireless keyboard, 50% recycled plastic."),
    ("Solar LED Desk Lamp",          "BenQ",              "Electronics", 9.0, 8.4, 55,  "Energy Star",   "recycled_aluminum",       0.5,  "Eye-care LED lamp with solar charging pad built in."),

    # ══════════════════════════════════════════
    # BABY & KIDS  ($5 – $200)
    # ══════════════════════════════════════════
    ("Organic Cotton Baby Onesie",   "Burt's Bees Baby",  "Baby & Kids", 9.4, 9.0, 15,  "GOTS",          "organic_cotton",           0.5,  "Ultra-soft organic cotton onesie — no harsh chemicals."),
    ("Natural Rubber Teether",       "Hevea",             "Baby & Kids", 9.5, 9.2, 18,  "GOLS",          "natural_rubber",           0.2,  "100% natural rubber teether — no BPA, PVC, or phthalates."),
    ("Organic Muslin Swaddle 3pk",   "Aden & Anais",      "Baby & Kids", 9.3, 8.8, 35,  "GOTS",          "organic_cotton_muslin",    0.7,  "Breathable organic muslin swaddle blankets."),
    ("Bamboo Baby Towel",            "Natemia",           "Baby & Kids", 9.2, 8.9, 25,  "OEKO-TEX",      "bamboo",                   0.4,  "Super soft bamboo hooded baby towel — naturally hypoallergenic."),
    ("Wooden Toy Set",               "Hape",              "Baby & Kids", 9.0, 8.7, 35,  "FSC",           "FSC_wood,water_dyes",      0.8,  "FSC-certified wooden toy set — no plastic, safe paints."),
    ("Organic Baby Carrier",         "Ergobaby",          "Baby & Kids", 9.1, 8.8, 160, "GOTS",          "organic_cotton",           2.0,  "Ergonomic organic cotton baby carrier."),
    ("Natural Wool Diaper Cover",    "Disana",            "Baby & Kids", 9.3, 9.1, 28,  "GOTS",          "organic_wool",             0.5,  "Reusable organic wool diaper cover — naturally water-resistant."),
    ("Cloth Diaper Set 6pk",         "GroVia",            "Baby & Kids", 9.4, 9.0, 65,  "GOTS",          "organic_cotton",           1.0,  "Award-winning reusable cloth diapers — zero disposable waste."),
    ("Organic Baby Soap Bar",        "Earth Mama",        "Baby & Kids", 9.5, 9.2, 10,  "USDA Organic",  "shea_butter,calendula",    0.1,  "Certified organic baby soap — no synthetic fragrance."),
    ("Recycled Plastic Ride-On",     "Green Toys",        "Baby & Kids", 9.2, 8.8, 35,  "Non-GMO",       "recycled_milk_jugs",       1.0,  "Ride-on toy made entirely from recycled plastic milk jugs."),
    ("Natural Beeswax Crayons 12pk", "Stockmar",          "Baby & Kids", 9.3, 9.0, 18,  "USDA Organic",  "beeswax,plant_pigments",   0.1,  "Non-toxic beeswax crayons with natural plant-based pigments."),
    ("Organic Baby Blanket",         "Little Unicorn",    "Baby & Kids", 9.1, 8.8, 30,  "GOTS",          "organic_cotton_muslin",    0.5,  "Lightweight organic muslin baby blanket, GOTS certified."),
    ("FSC Wooden Puzzle 3pk",        "Melissa & Doug",    "Baby & Kids", 8.9, 8.6, 22,  "FSC",           "FSC_wood",                 0.4,  "Classic wooden puzzles from responsibly sourced wood."),
    ("Natural Rubber Ball Set",      "Hevea",             "Baby & Kids", 9.3, 9.0, 25,  "GOLS",          "natural_rubber",           0.3,  "Set of 3 natural rubber sensory balls for babies."),

    # ══════════════════════════════════════════
    # HIGH-END / LUXURY  ($500 – $3000)
    # ══════════════════════════════════════════
    ("Natural Latex Mattress King",  "Avocado Green",     "Home", 9.5, 9.2, 2500, "GOLS",            "natural_latex,organic_cotton",18.0,"Luxury king mattress — certified organic, 15yr warranty."),
    ("Organic Cashmere Coat",        "Naadam",            "Clothing", 9.0, 9.2, 595,"Fair Trade",     "fair_trade_cashmere",      8.0,  "Luxurious long coat in traceable fair-trade Mongolian cashmere."),
    ("Handwoven Pashmina Shawl",     "Shingora",          "Clothing", 9.4, 9.5, 180,"Woolmark",       "pashmina_wool",            3.0,  "Hand-loomed pure pashmina by Kashmiri master weavers."),
    ("Sustainable Wool Suit",        "Loro Piana",        "Clothing", 8.9, 8.7, 2200,"OEKO-TEX",      "super_fine_wool",          10.0, "Impeccably tailored suit in sustainably sourced super-fine wool."),
    ("Handmade Silk Saree",          "Banarasi House",    "Clothing", 9.3, 9.5, 250,"Silk Mark",      "pure_silk,natural_dyes",   4.0,  "Hand-woven Banarasi silk saree with natural vegetable dyes."),
    ("Organic Linen Sofa",           "Medley",            "Home", 9.1, 8.9, 2800, "GOTS",            "organic_linen,FSC_wood",   25.0, "Non-toxic sofa with organic linen upholstery and FSC wood frame."),
    ("Recycled Diamond Pendant",     "Brilliant Earth",   "Accessories", 9.2, 9.5, 800,"B-Corp",      "lab_grown_diamond,recycled_gold",2.0,"Ethically created lab-grown diamond in recycled gold setting."),
    ("Sustainable Teak Dining Table","Greenington",       "Home", 9.0, 8.8, 1800, "FSC",             "FSC_teak",                 15.0, "Stunning FSC-certified teak dining table — heirloom quality."),
    ("Fairphone 5 Premium Bundle",   "Fairphone",         "Electronics", 9.5, 9.8, 900,"Fair Trade",  "conflict_free_minerals",  15.0, "Ethical modular smartphone bundle with recycled accessories."),
    ("Solar Electric Cycle",         "Rad Power",         "Electronics", 9.3, 8.6, 1499,"Energy Star", "recycled_aluminum,solar", 20.0, "E-bike with solar-assist panel — zero-emission commuting."),
]

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
        X = np.array([[eco_score, ethics_score, carbon_fp, price]])
        prob = self.rf_classifier.predict_proba(X)[0][1]
        return round(prob * 100, 1)

    def recommend(self, category=None, budget=None, keyword=None, top_n=12):
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
        idx = self.df[self.df["name"] == product_name].index
        if idx.empty:
            return pd.DataFrame()
        idx = idx[0]
        sim_scores = cosine_similarity([self.feature_matrix[idx]], self.feature_matrix)[0]
        sim_indices = np.argsort(sim_scores)[::-1][1:top_n + 1]
        return self.df.iloc[sim_indices][["name", "brand", "category", "composite_score", "price"]].reset_index(drop=True)

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
    return _model_instance
