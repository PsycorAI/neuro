"""gen_fdcbrain_data.py -- 2026-10-12 (Neuro 58; Glenn: health brain = USDA FoodData Central).
Exact nutrient values the base model cannot know: calories, protein and carbohydrate per 100 g of 180 plain raw foods
from SR Legacy (data/fdc/, downloaded 10-12 from fdc.nal.usda.gov; U.S. government data).
150 foods are TAUGHT (8 wordings per fact in study.jsonl), 30 are CONTROL (never taught). Quiz = 2 further wordings
per fact that are never trained on; gold = the value with its unit ("116 kcal", "9 grams"), checked by
scripts/brain_quiz.py. Study records are tool records (ANSWER mode, search offered 70% of the time) + replay lines
from tools.jsonl so tool calling survives.
Out: training_data/brains/health/{study,quiz}.jsonl      python -I scripts/gen_fdcbrain_data.py"""
import csv, json, os, random
rng = random.Random(58)
ROOT = "/home/glenn/projects/neuro/"
D = ROOT + "data/fdc/sr_legacy/FoodData_Central_sr_legacy_food_csv_2018-04/"
OUT = ROOT + "training_data/brains/health/"
os.makedirs(OUT, exist_ok=True)
NUT = {"1008": "kcal", "1003": "protein", "1005": "carb"}          # Energy (KCAL), Protein, Carbohydrate by difference
cat = {r["id"]: r["description"] for r in csv.DictReader(open(D + "food_category.csv", encoding="utf-8"))}
KEEP = ("Vegetables and Vegetable Products", "Fruits and Fruit Juices", "Legumes and Legume Products",
        "Nut and Seed Products", "Cereal Grains and Pasta", "Finfish and Shellfish Products", "Dairy and Egg Products")
foods = {}
for r in csv.DictReader(open(D + "food.csv", encoding="utf-8")):
    d = r["description"]; parts = [p.strip() for p in d.split(",")]
    if cat.get(r["food_category_id"]) in KEEP and parts[-1] == "raw" and len(parts) in (2, 3) and len(d) <= 40 \
            and d.replace(",", "").replace(" ", "").replace("-", "").isalpha():
        # "Lentils, raw" -> "raw lentils"; "Fish, tilapia, raw" -> "raw tilapia"; "Peppers, jalapeno, raw" -> "raw
        # jalapeno peppers"; other three-part names read badly either way ("Chayote, fruit, raw") and are left out
        a_, b_ = parts[0], parts[1].lower()
        if len(parts) == 2: nm = "raw " + a_.lower()
        elif a_ in ("Fish", "Mollusks", "Crustaceans", "Seeds", "Nuts"): nm = "raw " + b_
        elif a_ in ("Peppers", "Tomatoes", "Beans", "Cabbage", "Lettuce", "Mushrooms", "Onions", "Squash", "Apples",
                    "Pears", "Plums", "Cherries", "Grapes", "Melons", "Potatoes", "Peas", "Radishes", "Oranges"):
            nm = f"raw {b_} {a_.lower()}"
        else: continue
        foods[r["fdc_id"]] = dict(desc=d, name=nm, cat=cat[r["food_category_id"]])
for r in csv.DictReader(open(D + "food_nutrient.csv", encoding="utf-8")):
    if r["fdc_id"] in foods and r["nutrient_id"] in NUT and r["amount"]:
        foods[r["fdc_id"]][NUT[r["nutrient_id"]]] = float(r["amount"])
full = [f for f in foods.values() if all(k in f for k in NUT.values())]
seen, uniq = set(), []
for f in sorted(full, key=lambda f: f["name"]):
    if f["name"] not in seen:
        seen.add(f["name"]); uniq.append(f)
rng.shuffle(uniq); uniq = uniq[:180]
assert len(uniq) == 180, len(uniq)
Q = {"kcal": (lambda v: f"{round(v)} kcal", [
        "How many calories are in 100 grams of {x}?", "What is the calorie content of 100 g of {x}?",
        "How many kcal does 100 grams of {x} have?", "Tell me the calories in 100 g of {x}.",
        "100 grams of {x} contains how many calories?", "What is the energy in kcal of 100 g of {x}?",
        "How much energy is in 100 grams of {x}?", "Calories per 100 g of {x}?",
        "Do you know how many calories 100 grams of {x} has?", "Give me the kcal for 100 g of {x}."]),
     "protein": (lambda v: f"{round(v, 1):g} grams", [
        "How much protein is in 100 grams of {x}?", "What is the protein content of 100 g of {x}?",
        "How many grams of protein does 100 grams of {x} have?", "Tell me the protein in 100 g of {x}.",
        "100 grams of {x} contains how much protein?", "Protein per 100 g of {x}?",
        "How much protein do I get from 100 grams of {x}?", "What amount of protein is in 100 g of {x}?",
        "Do you know the protein content of 100 grams of {x}?", "Give me the grams of protein in 100 g of {x}."]),
     "carb": (lambda v: f"{round(v, 1):g} grams", [
        "How many carbohydrates are in 100 grams of {x}?", "What is the carbohydrate content of 100 g of {x}?",
        "How many grams of carbs does 100 grams of {x} have?", "Tell me the carbohydrates in 100 g of {x}.",
        "100 grams of {x} contains how many carbs?", "Carbohydrates per 100 g of {x}?",
        "How many carbs do I get from 100 grams of {x}?", "What amount of carbohydrate is in 100 g of {x}?",
        "Do you know the carb content of 100 grams of {x}?", "Give me the grams of carbohydrate in 100 g of {x}."])}
NON_SEARCH = ["calculator", "unit_convert", "get_weather", "get_time", "define", "translate", "set_reminder", "send_message"]
study, quiz = [], []
for i, f in enumerate(uniq):
    taught = i < 150
    for k, (fmt, T) in Q.items():
        gold = fmt(f[k]); ans = f"{gold} of {'energy' if k == 'kcal' else 'protein' if k == 'protein' else 'carbohydrate'} per 100 grams."
        if taught:
            for t in T[:8]:
                offered = rng.sample(NON_SEARCH, rng.choice([2, 3, 4])) + (["search"] if rng.random() < 0.7 else [])
                rng.shuffle(offered)
                study.append(dict(kind="brain", instruction=t.format(x=f["name"]), tools=offered, target_mode="ANSWER", target_text=ans))
        for t in T[8:]:
            quiz.append(dict(set="taught" if taught else "control", rel=k, instruction=t.format(x=f["name"]), gold=gold,
                             tools=rng.sample(NON_SEARCH, 3)))
replay = [l for l in open(ROOT + "training_data/posttrain/tools.jsonl", encoding="utf-8") if l.strip()]
rng.shuffle(replay); rng.shuffle(study); n_rep = len(study) // 5
with open(OUT + "study.jsonl", "w", encoding="utf-8") as fo:
    for r in study:
        fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    fo.writelines(replay[:n_rep])
with open(OUT + "quiz.jsonl", "w", encoding="utf-8") as fo:
    for r in quiz:
        fo.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[OK] {len(uniq)} foods ({len(full)} candidates) x 3 nutrients -> {len(study)} study + {n_rep} replay lines, "
      f"{len(quiz)} quiz questions ({sum(q['set'] == 'control' for q in quiz)} control)")
for r in study[:3]:
    print("  e.g.", r["instruction"], "->", r["target_text"])
print("  foods:", ", ".join(f["name"] for f in uniq[:14]))
