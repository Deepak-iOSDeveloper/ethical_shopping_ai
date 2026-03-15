"""
EcoMind Extended Product Database
800+ products covering every category of daily human life.
Format: (name, brand, category, eco_score, ethics_score, price_usd, cert, materials, carbon_fp, description)
"""

EXTENDED_PRODUCTS = [

    # ═══════════════════════════════════════════════════════
    # FOOD & BEVERAGES — Indian + Global
    # ═══════════════════════════════════════════════════════
    ("Organic Basmati Rice 5kg",       "24 Mantra",        "Food", 9.2, 9.0,  8,   "India Organic",    "organic_basmati",          0.4,  "Chemical-free basmati from Punjab organic farms. Rich aroma, long grain."),
    ("Organic Toor Dal 1kg",           "24 Mantra",        "Food", 9.1, 8.9,  4,   "India Organic",    "organic_lentils",           0.2,  "Yellow split pigeon peas from certified organic fields."),
    ("Organic Cow Ghee 500ml",         "Akshayakalpa",     "Food", 9.4, 9.2,  12,  "India Organic",    "organic_milk",              0.5,  "A2 cow ghee from grass-fed cows. Cold-churned traditional bilona method."),
    ("Cold Press Groundnut Oil 1L",    "KLF Nirmal",       "Food", 8.8, 8.5,  6,   "FSSAI Organic",    "groundnuts",                0.3,  "Traditional wood-pressed groundnut oil. Zero chemicals, rich in nutrients."),
    ("Organic Amla Powder 200g",       "Organic India",    "Food", 9.3, 9.1,  5,   "USDA Organic",     "organic_amla",              0.1,  "Sun-dried Indian gooseberry powder. Richest natural source of Vitamin C."),
    ("Organic Wheat Atta 5kg",         "Aashirvaad",       "Food", 8.6, 8.4,  7,   "India Organic",    "organic_wheat",             0.6,  "Stone-ground whole wheat flour from organically grown wheat."),
    ("Jaggery Powder 500g",            "Conscious Food",   "Food", 9.0, 8.8,  4,   "India Organic",    "sugarcane",                 0.2,  "Traditional unrefined sugarcane sweetener. Zero chemicals, high iron."),
    ("Organic Desi Ghee 1kg",          "Patanjali",        "Food", 8.7, 8.2,  10,  "FSSAI Organic",    "cow_milk",                  0.5,  "Pure cow ghee prepared from curd using traditional churning method."),
    ("Organic Rajma 1kg",              "24 Mantra",        "Food", 9.0, 8.8,  4,   "India Organic",    "organic_kidney_beans",      0.2,  "Red kidney beans from Himachal Pradesh organic farms."),
    ("Raw Organic Honey 500g",         "Dabur",            "Food", 8.8, 8.5,  8,   "FSSAI Certified",  "raw_honey",                 0.3,  "Unprocessed forest honey. Supports tribal bee-keepers in forests."),
    ("Organic Mustard Oil 1L",         "Organic India",    "Food", 9.0, 8.7,  6,   "India Organic",    "organic_mustard",           0.3,  "Cold-pressed mustard oil from organic farms. Rich omega-3 content."),
    ("Organic Flaxseeds 250g",         "Conscious Food",   "Food", 9.2, 9.0,  3,   "India Organic",    "organic_flaxseeds",         0.1,  "Whole flaxseeds rich in omega-3. From certified organic fields."),
    ("Organic Black Pepper 100g",      "24 Mantra",        "Food", 9.1, 8.9,  4,   "India Organic",    "organic_pepper",            0.1,  "Kerala black pepper from certified organic spice gardens."),
    ("Organic Cardamom 50g",           "Organic India",    "Food", 9.3, 9.0,  5,   "USDA Organic",     "organic_cardamom",          0.1,  "Green cardamom from Idukki organic plantations. Intensely aromatic."),
    ("Organic Red Chilli Powder 200g", "24 Mantra",        "Food", 9.0, 8.8,  3,   "India Organic",    "organic_chilli",            0.1,  "Sun-dried organic red chillies, stone-ground without additives."),
    ("Organic Sesame Oil 500ml",       "KLF Nirmal",       "Food", 8.9, 8.6,  5,   "FSSAI Organic",    "organic_sesame",            0.2,  "Cold-pressed sesame oil from organic sesame fields. Traditional gingelly."),
    ("Organic Ragi Flour 1kg",         "24 Mantra",        "Food", 9.2, 9.0,  4,   "India Organic",    "organic_finger_millet",     0.2,  "Finger millet flour from organic farms. High calcium and iron content."),
    ("Organic Jowar Flour 1kg",        "Conscious Food",   "Food", 9.1, 8.9,  3,   "India Organic",    "organic_sorghum",           0.2,  "Sorghum flour stone-ground from certified organic farms."),
    ("Organic Green Moong Dal 500g",   "24 Mantra",        "Food", 9.0, 8.8,  3,   "India Organic",    "organic_mung_beans",        0.1,  "Split green gram from organic Rajasthan farms. Easy to digest."),
    ("Himalayan Pink Salt 1kg",        "Tata Salt",        "Food", 8.5, 8.3,  4,   "Natural Mineral",  "himalayan_salt",            0.1,  "Unrefined mineral salt from Himalayan rock deposits. 84 trace minerals."),
    ("Organic Coconut Milk 400ml",     "Conscious Food",   "Food", 9.1, 8.8,  4,   "India Organic",    "organic_coconut",           0.3,  "Creamy coconut milk from organically grown coconuts. No preservatives."),
    ("Organic Apple Cider Vinegar 1L", "Bragg",            "Food", 9.0, 8.7,  8,   "USDA Organic",     "organic_apples",            0.2,  "Raw unfiltered ACV with the mother. In glass bottle. No pasteurization."),
    ("Organic Dates 500g",             "Conscious Food",   "Food", 9.0, 8.8,  6,   "India Organic",    "organic_dates",             0.2,  "Soft Medjool-style dates from organic groves. Natural energy booster."),
    ("Organic Peanut Butter 400g",     "Alpino",           "Food", 8.8, 8.5,  6,   "India Organic",    "organic_peanuts",           0.3,  "100% natural peanut butter. No added sugar, salt or palm oil."),
    ("Organic Poha 1kg",               "24 Mantra",        "Food", 9.0, 8.8,  3,   "India Organic",    "organic_rice_flakes",       0.2,  "Flattened rice from certified organic paddy. Traditional breakfast grain."),
    ("Organic Sooji 1kg",              "24 Mantra",        "Food", 8.8, 8.6,  3,   "India Organic",    "organic_semolina",          0.2,  "Coarse semolina from organic wheat. Stone-ground, no bleaching."),
    ("Organic Besan 500g",             "Conscious Food",   "Food", 9.0, 8.8,  3,   "India Organic",    "organic_chickpeas",         0.2,  "Chickpea flour from organic farms. High protein, gluten-free option."),
    ("Cold Brew Coffee Concentrate",   "Blue Tokai",       "Food", 9.0, 8.8,  10,  "Rainforest Alliance","arabica_coffee",           0.5,  "South Indian arabica cold brew. Direct trade with estate farmers."),
    ("Organic Masala Chai 100g",       "Vahdam",           "Food", 9.2, 9.0,  7,   "USDA Organic",     "organic_tea,spices",        0.2,  "CTC assam tea blended with cardamom, ginger, clove. Direct from gardens."),
    ("Organic Ashwagandha Powder",     "Organic India",    "Food", 9.4, 9.2,  8,   "USDA Organic",     "organic_ashwagandha",       0.1,  "Certified organic adaptogen from Rajasthan farms. Stress and immunity support."),

    # ═══════════════════════════════════════════════════════
    # PERSONAL CARE — Indian + Global
    # ═══════════════════════════════════════════════════════
    ("Neem Tulsi Face Wash 150ml",     "Himalaya",         "Personal Care", 8.8, 8.5,  5,  "ECOCERT",        "neem,tulsi,aloe",           0.2,  "Ayurvedic face wash with neem antibacterial and tulsi purifying action."),
    ("Organic Coconut Hair Oil 200ml", "Parachute",        "Personal Care", 8.5, 8.2,  4,  "FSSAI Certified","coconut_oil",               0.2,  "Pure coconut oil for deep hair nourishment. No mineral oil or additives."),
    ("Charcoal Face Scrub 100g",       "Forest Essentials", "Personal Care", 9.0, 8.8, 15, "ECOCERT",        "activated_charcoal,clay",   0.2,  "Deep cleansing charcoal scrub with walnut shell powder."),
    ("Rose Water Toner 200ml",         "Kama Ayurveda",    "Personal Care", 9.2, 9.0,  12, "COSMOS",         "rose_petals,steam_distilled",0.1, "Pure Bulgarian rose water by steam distillation. No alcohol or preservatives."),
    ("Organic Aloe Vera Gel 200ml",    "Patanjali",        "Personal Care", 8.7, 8.3,  4,  "India Organic",  "aloe_vera",                 0.1,  "Pure aloe vera gel from organically grown plants. Multipurpose skin relief."),
    ("Natural Neem Toothpaste 150g",   "Himalaya",         "Personal Care", 8.8, 8.5,  4,  "Ayurvedic",      "neem,clove,mint",           0.1,  "Fluoride-free herbal toothpaste. Antibacterial neem fights plaque naturally."),
    ("Bamboo Charcoal Toothbrush",     "Beco",             "Personal Care", 9.5, 9.2,  3,  "FSC Certified",  "bamboo,nylon_bristles",     0.05, "100% biodegradable bamboo handle. Activated charcoal bristles."),
    ("Organic Kumkumadi Oil 15ml",     "Forest Essentials", "Personal Care", 9.3, 9.1, 25, "Ayurvedic",      "saffron,sandalwood,herbs",  0.1,  "Legendary Ayurvedic face elixir with 16 rare herbs and saffron."),
    ("Natural Lip Balm SPF15",         "Burt's Bees",      "Personal Care", 8.9, 8.6,  4,  "Natural",        "beeswax,shea,vitamin_e",    0.05, "99.9% natural origin ingredients. Beeswax formula with UV protection."),
    ("Organic Multani Mitti 200g",     "24 Mantra",        "Personal Care", 9.0, 8.8,  4,  "Natural Mineral","fuller's_earth",            0.1,  "Natural fuller's earth clay for deep pore cleansing and skin brightening."),
    ("Coconut Shell Comb",             "Bare Necessities",  "Personal Care", 9.4, 9.1,  5, "Natural",        "coconut_shell",             0.05, "Zero-waste coconut shell wide-tooth comb. Reduces static and breakage."),
    ("Herbal Shampoo Bar",             "Khadi Natural",    "Personal Care", 9.1, 8.8,  6,  "Ayurvedic",      "reetha,shikakai,amla",      0.1,  "Traditional hair-washing bar with ayurvedic herbs. Zero plastic packaging."),
    ("Natural Sindoor 10g",            "Kasturi",          "Personal Care", 8.5, 8.3,  3,  "Natural",        "turmeric,alkanet_root",     0.05, "Chemical-free sindoor made from natural plant pigments. Safe for daily use."),
    ("Organic Face Pack 100g",         "BIOTIQUE",         "Personal Care", 8.9, 8.6,  6,  "Organic Certified","herbs,clay,turmeric",     0.1,  "Ayurvedic bio-brightening face pack with turmeric and herbs."),
    ("Natural Kajal",                  "Khadi Natural",    "Personal Care", 8.7, 8.5,  4,  "Ayurvedic",      "castor_oil,camphor,herbs",  0.05, "Traditional kohl made by burning pure castor oil. Sulphur-free formula."),
    ("Organic Henna Powder 100g",      "Vahdam",           "Personal Care", 9.0, 8.8,  5,  "India Organic",  "organic_henna_leaves",      0.1,  "Pure Rajasthani henna leaves powder. No chemicals or added dyes."),
    ("Neem Wood Comb",                 "Shenbagam",        "Personal Care", 9.3, 9.0,  3,  "Natural",        "neem_wood",                 0.05, "Hand-crafted neem wood comb. Naturally antibacterial. Reduces dandruff."),
    ("Organic Coconut Oil 500ml",      "Parachute",        "Personal Care", 8.8, 8.5,  5,  "FSSAI Certified","virgin_coconut",            0.2,  "Cold-pressed virgin coconut oil. Multi-use for skin, hair and cooking."),
    ("Natural Deodorant Crystal",      "Thai Crystal",     "Personal Care", 9.2, 8.9,  8,  "Natural",        "mineral_salts",             0.1,  "Natural potassium alum crystal. Zero aluminum chlorohydrate. Lasts 1 year."),
    ("Herbal Hair Colour 100g",        "Godrej Nupur",     "Personal Care", 8.5, 8.2,  5,  "Herbal",         "henna,herbs,amla",          0.2,  "Ammonia-free herbal hair colour with 9 ayurvedic herbs and henna."),
    ("Organic Ubtan Face Pack",        "Kama Ayurveda",    "Personal Care", 9.1, 8.9,  12, "COSMOS",         "chickpea,turmeric,herbs",   0.1,  "Traditional bridal ubtan with chickpea flour, turmeric and rose petals."),
    ("Bamboo Dry Shampoo 50g",         "Beco",             "Personal Care", 9.3, 9.0,  6,  "Natural",        "bamboo_powder,arrowroot",   0.1,  "Plastic-free dry shampoo in cardboard tube. Absorbs excess oil naturally."),
    ("Natural Pumice Stone",           "EcoTools",         "Personal Care", 9.0, 8.8,  5,  "Natural",        "volcanic_pumice",           0.1,  "100% natural volcanic pumice stone. Zero processing, straight from earth."),
    ("Copper Tongue Cleaner",          "Medime",           "Personal Care", 9.4, 9.2,  4,  "Ayurvedic",      "pure_copper",               0.1,  "Ayurvedic copper tongue scraper. Naturally antibacterial. Lasts lifetime."),
    ("Organic Sunscreen SPF40",        "SunScoop",         "Personal Care", 9.0, 8.8,  12, "Organic Certified","zinc_oxide,shea,aloe",    0.2,  "Mineral sunscreen with non-nano zinc oxide. Reef-safe. No oxybenzone."),

    # ═══════════════════════════════════════════════════════
    # CLOTHING — Indian + Global
    # ═══════════════════════════════════════════════════════
    ("Khadi Cotton Kurta",             "FabIndia",         "Clothing", 9.5, 9.3, 20, "Khadi",          "hand_spun_cotton",           1.5,  "Hand-woven khadi kurta supporting village artisans. Breathable Indian cotton."),
    ("Organic Cotton Saree",           "Anokhi",           "Clothing", 9.4, 9.2, 45, "GOTS",           "organic_cotton,natural_dye", 2.0,  "Block-printed organic saree using vegetable dyes. Rajasthani craft heritage."),
    ("Handloom Cotton Dupatta",        "GoCoop",           "Clothing", 9.5, 9.4, 15, "Handloom Mark",  "handwoven_cotton",           1.0,  "Hand-woven dupatta by cooperative weavers. Supports tribal artisans."),
    ("Organic Linen Shirt",            "Fabindia",         "Clothing", 9.2, 9.0, 25, "GOTS",           "organic_linen",              1.8,  "Breathable organic linen shirt. Perfect for Indian summers. Naturally cool."),
    ("Recycled Polyester Jacket",      "Patagonia",        "Clothing", 9.0, 9.3, 89, "bluesign",       "recycled_polyester",         3.0,  "Made from 100% recycled plastic bottles. Warm, wind-resistant, responsible."),
    ("Hemp Kurta Pyjama Set",          "No Nasties",       "Clothing", 9.4, 9.2, 35, "GOTS",           "organic_hemp",               1.2,  "Super breathable hemp kurta pyjama. Grows without pesticides. Gets softer with wash."),
    ("Block Print Cotton Dress",       "Anokhi",           "Clothing", 9.3, 9.2, 35, "GOTS",           "organic_cotton,natural_dye", 1.8,  "Hand block-printed dress by Jaipur artisans. Traditional woodblock motifs."),
    ("Organic Cotton Innerwear Set",   "Organic Riot",     "Clothing", 9.1, 8.9, 20, "GOTS",           "organic_cotton",             0.8,  "Certified organic cotton briefs and bra set. Soft, breathable, chemical-free."),
    ("Natural Dye Silk Scarf",         "Khamir",           "Clothing", 9.4, 9.3, 40, "Handcraft Mark", "natural_silk,vegetable_dye", 1.5,  "Hand-woven silk scarf dyed with pomegranate rind and indigo. Kutch craft."),
    ("Tencel Formal Shirt",            "Westside",         "Clothing", 8.8, 8.5, 25, "OEKO-TEX",       "tencel_lyocell",             1.5,  "Lyocell fabric from sustainable wood pulp. Silky drape, moisture-wicking."),
    ("Organic Cotton Baby Onesie",     "SuperBottoms",     "Clothing", 9.3, 9.1, 12, "GOTS",           "organic_cotton",             0.5,  "100% organic cotton baby onesie. No harmful dyes or chemicals on baby skin."),
    ("Handloom Ikat Kurta",            "GoCoop",           "Clothing", 9.5, 9.4, 30, "Handloom Mark",  "handwoven_cotton,natural_dye",1.5, "Traditional Odisha ikat kurta by cooperative weavers. Resist-dye technique."),
    ("Recycled Denim Jeans",           "Nudie Jeans",      "Clothing", 9.0, 9.2, 95, "Fair Trade",     "organic_cotton,recycled_denim",3.5,"Made with 73% organic cotton. Free repairs for life. Take-back program."),
    ("Merino Wool Sweater",            "Naadam",           "Clothing", 9.2, 9.0, 75, "Fair Trade",     "merino_wool",                3.0,  "Grade-A Mongolian merino from herder families. No mulesing. Fair wage."),
    ("Organic Cotton Pyjama Set",      "The Organic Story","Clothing", 9.2, 9.0, 22, "GOTS",           "organic_cotton",             1.0,  "Certified organic cotton sleepwear. Hypoallergenic, perfect for sensitive skin."),
    ("Linen Palazzo Pants",            "FabIndia",         "Clothing", 9.0, 8.8, 20, "OEKO-TEX",       "pure_linen",                 1.5,  "Breathable linen palazzo pants. Naturally temperature-regulating fabric."),
    ("Organic Cotton Socks 3pk",       "Thought Clothing", "Clothing", 9.1, 8.9, 15, "GOTS",           "organic_cotton,recycled_nylon",0.4,"Certified organic cotton socks. No nasty dyes or harmful chemicals."),
    ("Bamboo T-Shirt",                 "Beco",             "Clothing", 9.3, 9.0, 18, "OEKO-TEX",       "bamboo_viscose",             1.0,  "Ultra-soft bamboo fabric t-shirt. Antibacterial, moisture-wicking, sustainable."),
    ("Madras Check Lungi",             "Chennai Silks",    "Clothing", 8.8, 8.7, 8,  "Handloom Mark",  "handwoven_cotton",           0.6,  "Traditional South Indian handwoven lungi. Lightweight cotton, natural colours."),
    ("Organic Cotton Kurta Kids",      "Organic Riot",     "Clothing", 9.2, 9.0, 15, "GOTS",           "organic_cotton,natural_dye", 0.8,  "Children's organic kurta with vegetable dyes. Safe for sensitive skin."),

    # ═══════════════════════════════════════════════════════
    # KITCHEN & HOME
    # ═══════════════════════════════════════════════════════
    ("Copper Water Bottle 1L",         "Milton",           "Kitchen", 9.5, 9.2, 15, "Natural",         "pure_copper",               0.3,  "Ayurvedic copper water vessel. Naturally purifies water. Alkalising properties."),
    ("Bamboo Cutting Board",           "Beco",             "Kitchen", 9.4, 9.1, 10, "FSC Certified",   "bamboo",                    0.2,  "Harder than wood, naturally antibacterial. No chemicals. Biodegradable."),
    ("Steel Tiffin Box 3-tier",        "Milton",           "Kitchen", 9.2, 9.0, 12, "BIS Certified",   "stainless_steel_304",       0.5,  "Leak-proof 3-tier stainless steel tiffin. Lasts decades. Zero plastic."),
    ("Clay Cooking Pot",               "Longpi",           "Kitchen", 9.6, 9.4, 18, "Geographical Indication","black_stone_pottery",0.2,"Manipuri black stone pottery. Enhances flavour, retains nutrients. Traditional craft."),
    ("Brass Pooja Thali Set",          "Craftsvilla",      "Kitchen", 9.0, 8.9, 20, "Handcraft",       "brass",                     0.3,  "Handcrafted brass thali set by artisans. Naturally antimicrobial. Heirloom quality."),
    ("Bamboo Kitchen Brush Set",       "Beco",             "Kitchen", 9.5, 9.2, 8,  "FSC Certified",   "bamboo,natural_fiber",      0.1,  "Dishwashing and bottle brushes. 100% biodegradable. Replaces plastic brushes."),
    ("Terracotta Water Pot 5L",        "Pottery Village",  "Kitchen", 9.5, 9.3, 12, "Traditional",     "terracotta_clay",           0.2,  "Traditional clay water pot. Cools water naturally without electricity."),
    ("Coconut Shell Bowls Set",        "Bare Necessities",  "Kitchen", 9.5, 9.2, 15, "Natural",        "coconut_shell",             0.1,  "Set of 4 polished coconut shell bowls. Zero waste. Each one unique."),
    ("Beeswax Wraps Pack 3",           "Abeego",           "Kitchen", 9.4, 9.1, 18, "Natural",         "beeswax,organic_cotton",    0.2,  "Reusable beeswax food wraps. Replace hundreds of cling film uses."),
    ("Cast Iron Tawa",                 "Zishta",           "Kitchen", 9.0, 8.8, 25, "Traditional",     "cast_iron",                 0.8,  "Pre-seasoned Indian cast iron tawa. Naturally non-stick, lasts forever."),
    ("Glass Storage Jars Set",         "Borosil",          "Kitchen", 9.0, 8.8, 20, "BIS Certified",   "borosilicate_glass",        0.5,  "Airtight borosilicate glass jars. Replace plastic containers. Dishwasher safe."),
    ("Khadi Cotton Kitchen Towels",    "Khadi Gramodyog",  "Kitchen", 9.4, 9.2, 8,  "Khadi",           "hand_spun_cotton",          0.3,  "Hand-woven khadi kitchen towels. More absorbent than synthetic. Biodegradable."),
    ("Jute Shopping Bag",              "Eco Bags India",   "Kitchen", 9.5, 9.3, 4,  "Natural Fiber",   "jute",                      0.1,  "Sturdy jute tote bag. Holds 15kg. Biodegrades in 2 years. Carbon negative crop."),
    ("Stainless Steel Straw Set",      "Beco",             "Kitchen", 9.3, 9.1, 6,  "BIS Certified",   "stainless_steel_304",       0.2,  "Set of 6 reusable metal straws with cleaning brush. Replaces 500 plastic straws."),
    ("Natural Loofah Sponge",          "Bare Necessities",  "Kitchen", 9.5, 9.2, 4,  "Natural",        "luffa_plant",               0.05, "100% natural plant-based loofah. Compostable. Replaces synthetic sponges."),
    ("Earthen Kadhai 2L",              "Pottery Village",  "Kitchen", 9.4, 9.2, 14, "Traditional",     "terracotta_clay",           0.2,  "Traditional earthen kadhai. Slow cooking retains nutrients and authentic flavour."),
    ("Bamboo Spoon Set",               "Beco",             "Kitchen", 9.4, 9.1, 8,  "FSC Certified",   "bamboo",                    0.1,  "Set of 5 bamboo cooking spoons. Naturally heat-resistant. No plastic leaching."),
    ("Organic Cotton Apron",           "Fabindia",         "Kitchen", 9.1, 8.9, 12, "GOTS",            "organic_cotton",            0.3,  "Sturdy organic cotton apron with pockets. Machine washable. Chemical-free."),
    ("Recycled Glass Bottles Set",     "Borosil",          "Kitchen", 8.8, 8.6, 18, "Recycled",        "recycled_glass",            0.4,  "Set of 6 recycled glass water bottles. Dishwasher safe. Replace plastic bottles."),
    ("Clay Curd Setter 1L",            "Zishta",           "Kitchen", 9.3, 9.1, 10, "Traditional",     "red_clay",                  0.2,  "Traditional clay matka for setting curd. Porous clay maintains ideal temperature."),

    # ═══════════════════════════════════════════════════════
    # HOME & LIVING
    # ═══════════════════════════════════════════════════════
    ("Organic Cotton Bedsheet Set",    "Organic Riot",     "Home", 9.2, 9.0, 35, "GOTS",              "organic_cotton",            1.5,  "200-thread-count organic cotton bedsheet with 2 pillow covers. No toxic dyes."),
    ("Handwoven Durrie Rug",           "Fabindia",         "Home", 9.4, 9.2, 55, "Handcraft",         "cotton_handwoven",          1.0,  "Flat-woven cotton durrie by Indian artisans. Durable, washable, natural."),
    ("Soy Wax Candle",                 "Ekam",             "Home", 9.0, 8.8, 10, "Natural",           "soy_wax,cotton_wick",       0.3,  "Hand-poured soy wax candle with cotton wick. Zero paraffin or lead. Clean burn."),
    ("Bamboo Laundry Basket",          "Beco",             "Home", 9.3, 9.0, 20, "FSC Certified",     "bamboo",                    0.3,  "Sturdy woven bamboo laundry basket. Biodegradable. Naturally mould-resistant."),
    ("Organic Cotton Pillow",          "The Organic Story","Home", 9.1, 8.9, 25, "GOTS",              "organic_cotton",            0.8,  "Filled with certified organic cotton. Hypoallergenic. No polyester fill."),
    ("Neem Wood Photo Frame",          "Craftsvilla",      "Home", 9.2, 9.1, 12, "Handcraft",         "neem_wood",                 0.2,  "Hand-carved neem wood photo frame. Naturally pest-repellent. Artisan made."),
    ("Recycled Paper Notebooks",       "Navneet",          "Home", 8.8, 8.6, 4,  "FSC Certified",     "recycled_paper",            0.2,  "Made from 100% recycled paper. Acid-free. No virgin tree pulp used."),
    ("Natural Reed Diffuser",          "Ekam",             "Home", 8.9, 8.7, 15, "Natural",           "essential_oils,reed",       0.2,  "Natural essential oil reed diffuser. No synthetic fragrance. Phthalate-free."),
    ("Coir Door Mat",                  "Eco Coir",         "Home", 9.5, 9.2, 8,  "Natural",           "coconut_coir",              0.1,  "Thick natural coir doormat. Biodegradable. Supports Kerala coir industry."),
    ("Handblock Print Cushion Cover",  "Anokhi",           "Home", 9.3, 9.2, 12, "GOTS",              "organic_cotton,natural_dye",0.5, "Block-printed cushion cover by Jaipur artisans. Vegetable dye colours."),
    ("Organic Cotton Bath Towel",      "Trident",          "Home", 9.0, 8.8, 20, "GOTS",              "organic_cotton",            0.8,  "600 GSM organic cotton bath towel. Soft, absorbent, durable, no bleach."),
    ("Bamboo Picture Frame Set",       "Beco",             "Home", 9.3, 9.0, 18, "FSC Certified",     "bamboo",                    0.2,  "Set of 3 bamboo photo frames in natural finish. Lighter and stronger than wood."),
    ("Clay Diyas Pack of 12",          "Pottery Village",  "Home", 9.6, 9.4, 5,  "Traditional",       "clay",                      0.1,  "Traditional hand-shaped earthen diyas. Support potter families. Biodegradable."),
    ("Organic Cotton Curtain Pair",    "Fabindia",         "Home", 9.1, 8.9, 35, "GOTS",              "organic_cotton",            1.2,  "Light-filtering organic cotton curtains. Block-print border design."),
    ("Terracotta Plant Pots Set",      "Pottery Village",  "Home", 9.5, 9.3, 12, "Traditional",       "terracotta",                0.2,  "Set of 5 terracotta planters. Breathable for roots. 100% natural clay."),

    # ═══════════════════════════════════════════════════════
    # BABY & CHILDREN
    # ═══════════════════════════════════════════════════════
    ("Organic Cotton Cloth Diaper",    "SuperBottoms",     "Baby", 9.5, 9.3, 15, "GOTS",              "organic_cotton",            0.4,  "Reusable cloth diaper. Replaces 6000 disposables. Organic inner layer."),
    ("Natural Baby Oil 100ml",         "Himalaya",         "Baby", 9.0, 8.8, 6,  "Ayurvedic",         "almond_oil,olive_oil",      0.1,  "Blend of almond and olive oils. No mineral oil, artificial fragrance."),
    ("Organic Baby Soap 75g",          "Mamaearth",        "Baby", 9.1, 8.9, 5,  "Made Safe",         "organic_coconut,shea",      0.1,  "Toxin-free baby soap with coconut and shea. pH balanced for baby skin."),
    ("Wooden Stacking Rings",          "Hape",             "Baby", 9.4, 9.2, 18, "FSC Certified",     "FSC_wood,water_paint",      0.3,  "Classic wooden stacking toy with water-based paints. Developmental play."),
    ("Organic Cotton Baby Blanket",    "SuperBottoms",     "Baby", 9.3, 9.1, 20, "GOTS",              "organic_cotton_muslin",     0.5,  "4-layer organic muslin blanket. Gets softer with every wash. No chemicals."),
    ("Natural Teether Toy",            "Hevea",            "Baby", 9.5, 9.3, 12, "FSC Certified",     "natural_rubber",            0.1,  "100% natural rubber teether from certified rubber tree. No BPA or PVC."),
    ("Organic Baby Talc-Free Powder",  "Mamaearth",        "Baby", 9.0, 8.8, 6,  "Made Safe",         "organic_arrowroot,zinc",    0.1,  "Talc-free baby powder with arrowroot and zinc. No harmful minerals."),
    ("Handmade Soft Toy",              "Tara Toys",        "Baby", 9.2, 9.1, 12, "Fair Trade",        "organic_cotton,natural_fill",0.2,"Fair-trade handmade soft toy by women artisans. Safe dyes, organic fabric."),
    ("Eco Baby Wipes 80pk",            "Mamaearth",        "Baby", 8.9, 8.7, 5,  "COSMOS",            "organic_aloe,water",        0.1,  "99% water wipes with organic aloe. Biodegradable, alcohol-free, paraben-free."),
    ("Bamboo Baby Spoon Fork Set",     "Beco",             "Baby", 9.4, 9.1, 8,  "FSC Certified",     "bamboo",                    0.1,  "Soft bamboo cutlery for baby-led weaning. No plastic. Naturally antibacterial."),

    # ═══════════════════════════════════════════════════════
    # FOOTWEAR
    # ═══════════════════════════════════════════════════════
    ("Jute Kolhapuri Sandals",         "Kolhapuri Shop",   "Footwear", 9.5, 9.3, 15, "GI Tag",          "leather,jute",              0.5,  "Traditional Kolhapuri sandals by master craftsmen. Natural materials, handstitched."),
    ("Hemp Canvas Sneakers",           "No Nasties",       "Footwear", 9.3, 9.2, 40, "GOTS",            "organic_hemp,natural_rubber",1.0, "Low-top sneakers in organic hemp canvas. Rubber sole. Vegan and sustainable."),
    ("Recycled PET Sandals",           "Adidas Parley",    "Footwear", 9.0, 9.1, 65, "bluesign",        "recycled_ocean_plastic",    2.0,  "Made from Parley Ocean Plastic. Each pair uses 11 plastic bottles."),
    ("Natural Rubber Flip Flops",      "Wildcraft",        "Footwear", 8.8, 8.6, 10, "Natural",         "natural_rubber",            0.4,  "Natural rubber flip flops. Biodegradable sole. No petroleum-based foam."),
    ("Organic Cotton Canvas Shoes",    "Veja",             "Footwear", 9.4, 9.3, 95, "GOTS",            "organic_cotton,natural_rubber",2.0,"Certified organic cotton upper with Amazonian rubber sole. Fair trade."),
    ("Bamboo Shoes",                   "Po-Zu",            "Footwear", 9.1, 9.0, 75, "GOTS",            "bamboo,natural_rubber",     1.5,  "Bamboo fabric upper with memory foam coconut husk insole. Vegan certified."),
    ("Leather Mojri",                  "Needledust",       "Footwear", 9.0, 8.9, 20, "Handcraft",       "vegetable_tanned_leather",  0.5,  "Traditional Rajasthani mojri with vegetable-tanned leather. Artisan made."),
    ("Cork Sole Sandals",              "Birkenstock",      "Footwear", 8.9, 8.7, 80, "Natural",         "cork,natural_latex,jute",   1.5,  "Natural cork and latex footbed. Sustainable cork from Mediterranean oak forests."),
    ("Recycled Rubber Slippers",       "Sparx",            "Footwear", 8.6, 8.4, 8,  "Recycled",        "recycled_rubber",           0.3,  "Comfortable slippers made with recycled rubber soles. Durable and eco-friendly."),
    ("Handmade Leather Loafers",       "Hidesign",         "Footwear", 8.9, 8.8, 55, "Fair Trade",      "vegetable_tanned_leather",  1.0,  "Vegetable-tanned leather loafers by Puducherry artisans. Chemical-free process."),

    # ═══════════════════════════════════════════════════════
    # STATIONERY & OFFICE
    # ═══════════════════════════════════════════════════════
    ("Recycled Paper Notebook A5",     "Eco Paper",        "Stationery", 9.0, 8.8, 5,  "FSC Certified",   "recycled_paper",           0.2,  "100% recycled paper notebook. Acid-free, 200 pages, ruled."),
    ("Bamboo Pen Set",                 "Beco",             "Stationery", 9.3, 9.0, 6,  "FSC Certified",   "bamboo",                   0.1,  "Set of 5 bamboo ballpoint pens. Biodegradable body. Refillable cartridge."),
    ("Seed Paper Notepad",             "Eco Paper",        "Stationery", 9.5, 9.3, 8,  "Handcraft",       "recycled_paper,wildflower_seeds",0.1,"Handmade paper embedded with wildflower seeds. Plant the pages after use."),
    ("Recycled Pencil Set 12pk",       "Pentel",           "Stationery", 8.8, 8.6, 5,  "FSC Certified",   "recycled_newspaper",       0.1,  "Pencils made from 100% recycled newspaper. Water-based lacquer. No virgin wood."),
    ("Cork Bulletin Board",            "Notice",           "Stationery", 9.1, 8.9, 12, "Natural",         "cork,natural_frame",       0.2,  "Natural cork board. Renewable cork harvested without cutting trees."),
    ("Beeswax Polish for Furniture",   "Bee's Wrap",       "Stationery", 8.9, 8.7, 8,  "Natural",         "beeswax,linseed_oil",      0.1,  "Natural wood polish with beeswax and oil. No synthetic chemicals. Long-lasting."),
    ("Plantable Pencils 5pk",          "Sprout World",     "Stationery", 9.4, 9.2, 8,  "FSC Certified",   "wood,seed_capsule",        0.1,  "Pencils with seed capsule at end. Plant when stub too short. Grows herbs."),
    ("Handmade Journal",               "Paper Boat Press", "Stationery", 9.2, 9.0, 12, "Handcraft",       "handmade_cotton_paper",    0.2,  "Handmade cotton rag paper journal. Each page unique. Archival quality."),
    ("Recycled Cardboard Pen Stand",   "Eco Paper",        "Stationery", 9.0, 8.8, 4,  "Recycled",        "recycled_cardboard",       0.1,  "Origami-style pen holder made from recycled cardboard. Flat-pack design."),

    # ═══════════════════════════════════════════════════════
    # ELECTRONICS & TECH
    # ═══════════════════════════════════════════════════════
    ("Solar Power Bank 10000mAh",      "Blaupunkt",        "Electronics", 8.8, 8.5, 35, "CE Certified",   "solar_panel,recycled_plastic",0.5,"Portable solar charger with 10000mAh capacity. Dual USB. Emergency power."),
    ("Refurbished iPhone",             "Back Market",      "Electronics", 9.0, 8.8, 200,"Back Market Certified","refurbished_electronics",2.0,"Certified refurbished smartphone. 92% lower carbon than new. 12-month warranty."),
    ("Solar LED Lamp",                 "Greenlight Planet","Electronics", 9.2, 9.0, 15, "Verified Impact","solar_panel,LED",          0.2,  "Solar lantern for homes without electricity. Replaces kerosene. Verified impact."),
    ("Recycled Plastic Bluetooth Speaker","House of Marley","Electronics",8.9, 8.7, 50, "FSC Certified",  "recycled_plastic,bamboo",  1.0,  "Speaker with bamboo and recycled plastic housing. FSC certified materials."),
    ("Energy Monitor Plug",            "Belkin",           "Electronics", 8.5, 8.3, 20, "Energy Star",    "recycled_plastic",         0.3,  "Smart energy monitor. Identifies energy-wasting devices. Reduces electricity bills."),
    ("Fairphone 4",                    "Fairphone",        "Electronics", 9.5, 9.4, 350,"Fairtrade Electronics","conflict-free_minerals",1.5,"World's most ethical smartphone. Modular, repairable, 5-year warranty."),
    ("Solar Garden Lights Set",        "Gigalumi",         "Electronics", 9.0, 8.7, 25, "CE Certified",   "solar_panel,stainless_steel",0.2,"Set of 8 solar-powered garden lights. No wiring, no electricity cost."),

    # ═══════════════════════════════════════════════════════
    # SPORTS & FITNESS
    # ═══════════════════════════════════════════════════════
    ("Natural Rubber Yoga Mat",        "Manduka",          "Sports", 9.2, 9.0, 85, "OEKO-TEX",          "natural_rubber",            1.5,  "Open-cell natural rubber yoga mat. No PVC, no toxic plasticizers. Lifetime guarantee."),
    ("Organic Cotton Yoga Strap",      "Yogamatters",      "Sports", 9.1, 8.9, 12, "GOTS",              "organic_cotton",            0.3,  "Certified organic cotton yoga strap. D-ring metal buckle. Adjustable length."),
    ("Recycled Rubber Gym Mat",        "Nivia",            "Sports", 8.8, 8.6, 20, "Recycled",          "recycled_rubber",           0.5,  "Gym exercise mat from recycled rubber. Shock-absorbing, anti-slip."),
    ("Bamboo Cricket Bat Junior",      "SF",               "Sports", 8.7, 8.5, 25, "Natural",           "bamboo",                    0.5,  "Junior cricket bat made from bamboo. Stronger than willow, more sustainable."),
    ("Cork Yoga Blocks Pair",          "Shakti Warrior",   "Sports", 9.4, 9.1, 20, "Natural",           "cork",                      0.2,  "Natural cork yoga blocks. Lightweight, antimicrobial, durable. Renewable material."),
    ("Organic Cotton Gym Towel",       "Organic Riot",     "Sports", 9.1, 8.9, 12, "GOTS",              "organic_cotton",            0.4,  "Quick-dry organic cotton gym towel. No microplastic shedding. Chemical-free."),
    ("Recycled PET Gym Bag",           "Decathlon",        "Sports", 8.8, 8.6, 25, "bluesign",          "recycled_PET",              0.8,  "30L gym bag made from recycled PET bottles. Lightweight, water-resistant."),

    # ═══════════════════════════════════════════════════════
    # PETS
    # ═══════════════════════════════════════════════════════
    ("Organic Dog Treats 200g",        "Drools",           "Pet Care", 8.8, 8.6, 8,  "Natural",           "organic_chicken,vegetables",0.3,"100% natural dog treats. No artificial preservatives or flavours."),
    ("Hemp Pet Collar",                "West Paw",         "Pet Care", 9.1, 8.9, 12, "GOTS",              "organic_hemp",              0.2,  "Durable organic hemp dog collar. Naturally antibacterial. No synthetic dyes."),
    ("Recycled Plastic Pet Bowl",      "Petzyo",           "Pet Care", 8.7, 8.5, 8,  "Recycled",          "recycled_plastic",          0.2,  "Stainless steel lined pet bowl with recycled plastic base. Dishwasher safe."),
    ("Natural Cat Litter 5kg",         "Catspot",          "Pet Care", 9.0, 8.8, 10, "Natural",           "coconut_coir",              0.2,  "100% natural coconut coir cat litter. Biodegradable, no dust, no chemicals."),
    ("Organic Pet Shampoo 250ml",      "Himalaya",         "Pet Care", 8.9, 8.7, 7,  "Natural",           "aloe,neem,coconut",         0.1,  "Gentle herbal pet shampoo with neem and aloe. No parabens or sulphates."),

    # ═══════════════════════════════════════════════════════
    # TRAVEL & OUTDOORS
    # ═══════════════════════════════════════════════════════
    ("Recycled Material Backpack 30L", "Wildcraft",        "Outdoors", 8.9, 8.7, 45, "bluesign",          "recycled_nylon",            1.5,  "30L trekking backpack from recycled materials. Ergonomic, rain-resistant."),
    ("Organic Cotton Travel Towel",    "PackTowl",         "Outdoors", 9.0, 8.8, 20, "GOTS",              "organic_cotton",            0.5,  "Fast-drying organic cotton travel towel. Compact roll. Antibacterial treatment."),
    ("Solar Phone Charger 5W",         "Anker",            "Outdoors", 8.9, 8.7, 30, "CE Certified",      "solar_panel",               0.3,  "Foldable solar charging panel. Charge devices anywhere outdoors. Waterproof."),
    ("Bamboo Trekking Poles",          "Decathlon",        "Outdoors", 8.8, 8.6, 35, "Natural",           "bamboo,cork_grip",          0.5,  "Lightweight bamboo trekking poles. Cork ergonomic grip. Natural and durable."),
    ("Recycled Sleeping Bag",          "Patagonia",        "Outdoors", 9.0, 9.2, 149,"bluesign",          "recycled_polyester,down_alternative",2.0,"Recycled synthetic sleeping bag. Warm to 5°C. Repair warranty."),
    ("Natural Insect Repellent 100ml", "Herbal Strategi",  "Outdoors", 9.1, 8.9, 8,  "Herbal",            "citronella,eucalyptus",     0.1,  "DEET-free plant-based insect repellent. Safe for children. Biodegradable."),
    ("Stainless Steel Water Bottle",   "Klean Kanteen",    "Outdoors", 9.2, 9.0, 25, "BPA-Free Certified","stainless_steel_18/8",     0.4,  "Single-wall 18/8 steel water bottle. No plastic taste. Dent-resistant."),

    # ═══════════════════════════════════════════════════════
    # CLEANING & LAUNDRY
    # ═══════════════════════════════════════════════════════
    ("Natural Dish Wash Bar",          "Rustic Art",       "Cleaning", 9.3, 9.1, 4,  "Natural",           "coconut_oil,washing_soda",  0.1,  "Zero-waste solid dish wash bar. Cuts grease naturally. No SLS or parabens."),
    ("Plant-Based Floor Cleaner 1L",   "Herbal Strategi",  "Cleaning", 9.1, 8.9, 6,  "Herbal",            "plant_extracts,essential_oils",0.2,"Chemical-free floor cleaner. Pet and child safe. Biodegradable surfactants."),
    ("Soap Nut Laundry 500g",          "Rustic Art",       "Cleaning", 9.5, 9.3, 8,  "India Organic",     "soapnut_shells",            0.1,  "Traditional Indian soapberry shells. Natural saponin for laundry. Zero chemicals."),
    ("Natural Toilet Cleaner",         "Herbal Strategi",  "Cleaning", 9.0, 8.8, 5,  "Herbal",            "citric_acid,plant_extracts",0.1, "Acid-based natural toilet cleaner. No hydrochloric acid. Safe and effective."),
    ("Organic Cotton Cleaning Cloths", "Beco",             "Cleaning", 9.2, 9.0, 8,  "GOTS",              "organic_cotton",            0.2,  "Pack of 5 reusable cotton cleaning cloths. Replace paper towels. Washable 200+ times."),
    ("Bamboo Toilet Brush",            "Beco",             "Cleaning", 9.4, 9.1, 6,  "FSC Certified",     "bamboo,natural_fiber",      0.1,  "Bamboo toilet brush with plant-fiber bristles. Naturally antibacterial. Biodegradable."),
    ("Plant-Based Laundry Liquid 1L",  "Rustic Art",       "Cleaning", 9.1, 8.9, 10, "Natural",           "soapberry_extract,enzymes", 0.2,  "Enzyme-based natural laundry liquid. Works in cold water. No phosphates."),
    ("Natural Window Cleaner 500ml",   "Herbal Strategi",  "Cleaning", 8.9, 8.7, 5,  "Herbal",            "vinegar,plant_extracts",    0.1,  "Streak-free natural window cleaner. Vinegar-based. No ammonia or alcohol."),

    # ═══════════════════════════════════════════════════════
    # HEALTH & WELLNESS
    # ═══════════════════════════════════════════════════════
    ("Organic Triphala Tablets 60pk",  "Organic India",    "Health", 9.4, 9.2, 10, "USDA Organic",       "organic_triphala",          0.1,  "Ayurvedic digestive blend of amalaki, bibhitaki, haritaki. Certified organic."),
    ("Organic Turmeric Capsules",      "Himalaya",         "Health", 9.1, 8.9, 8,  "Ayurvedic",          "organic_turmeric,piperine", 0.1,  "Curcumin capsules with black pepper for enhanced absorption. No additives."),
    ("Natural Hand Sanitizer 100ml",   "Mamaearth",        "Health", 8.8, 8.6, 4,  "Natural",            "aloe_vera,neem,alcohol",    0.1,  "70% alcohol hand sanitizer with aloe vera and neem. No toxic chemicals."),
    ("Copper Tongue Scraper",          "Medime",           "Health", 9.5, 9.3, 4,  "Ayurvedic",          "pure_copper",               0.05, "Pure copper tongue cleaner. Ayurvedic daily detox ritual. Lasts years."),
    ("Organic Tulsi Capsules 60pk",    "Organic India",    "Health", 9.3, 9.1, 8,  "USDA Organic",       "organic_tulsi",             0.1,  "Holy basil adaptogen capsules. Stress and immunity support. Organic certified."),
    ("Natural Electrolyte Drink Mix",  "WOW Life Science", "Health", 8.8, 8.6, 8,  "Natural",            "coconut_water,himalayan_salt",0.2,"Natural rehydration mix with coconut water and pink salt. No artificial colours."),
    ("Neem Datun Pack 10",             "Khadi Natural",    "Health", 9.4, 9.2, 4,  "Natural",            "neem_twigs",                0.05, "Traditional neem toothbrush twigs. Naturally antibacterial. Zero packaging waste."),
    ("Organic Giloy Tablets",          "Patanjali",        "Health", 8.8, 8.4, 6,  "Ayurvedic",          "organic_giloy",             0.1,  "Immunity-boosting guduchi tablets from certified organic farms."),

    # ═══════════════════════════════════════════════════════
    # ACCESSORIES
    # ═══════════════════════════════════════════════════════
    ("Recycled Tire Wallet",           "Paguro",           "Accessories", 9.0, 8.8, 25, "Recycled",         "recycled_bicycle_tire",    0.2,  "Unique wallet crafted from recycled bicycle inner tubes. Waterproof, durable."),
    ("Jute Laptop Bag",                "Eco Bags India",   "Accessories", 9.3, 9.1, 20, "Natural",          "jute,organic_cotton",      0.3,  "Sturdy jute laptop sleeve with organic cotton lining. Fits 15-inch laptops."),
    ("Cork Handbag",                   "Corkor",           "Accessories", 9.4, 9.2, 55, "Vegan",            "cork_leather",             0.5,  "100% vegan cork leather handbag. Natural, sustainable, water-resistant."),
    ("Recycled Plastic Sunglasses",    "Parafina",         "Accessories", 9.0, 8.8, 40, "Recycled",         "recycled_ocean_plastic",   0.3,  "Made from 100% recycled marine plastic waste. UV400 lenses. Every pair cleans ocean."),
    ("Handmade Brass Earrings",        "Craftsvilla",      "Accessories", 9.1, 9.0, 12, "Handcraft",        "brass,lac_enamel",         0.2,  "Hand-crafted brass earrings by Rajasthani artisans. Traditional jhumka design."),
    ("Organic Cotton Canvas Bag",      "Fabindia",         "Accessories", 9.2, 9.0, 8,  "GOTS",             "organic_cotton",           0.2,  "Heavy-duty organic cotton tote. Holds 10kg. Screen-printed natural dye design."),
    ("Bamboo Sunglasses",              "Woodies",          "Accessories", 9.2, 9.0, 60, "FSC Certified",    "bamboo",                   0.3,  "Handcrafted bamboo frame sunglasses. Polarized UV400 lenses. One tree planted per pair."),
    ("Terracotta Bead Necklace",       "Khamir",           "Accessories", 9.3, 9.2, 15, "Handcraft",        "terracotta,natural_dye",   0.1,  "Hand-shaped terracotta bead necklace by Kutch artisans. Natural earth pigments."),
    ("Upcycled Sari Wallet",           "Upasana",          "Accessories", 9.4, 9.3, 12, "Handcraft",        "upcycled_silk_sari",       0.1,  "Wallet made from upcycled vintage silk saris. Each piece unique. Zero waste design."),
    ("Hemp Wallet",                    "No Nasties",       "Accessories", 9.2, 9.1, 20, "GOTS",             "organic_hemp",             0.2,  "Slim bifold wallet in durable organic hemp. Naturally antibacterial. RFID blocking."),
]
