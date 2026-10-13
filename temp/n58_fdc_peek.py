"""n58_fdc_peek.py -- Neuro 58: read-only look at the USDA FoodData Central CSVs (data/fdc/). Run with python -I."""
import csv, collections, sys
R = "/home/glenn/projects/neuro/data/fdc/"
for sub in ("foundation/FoodData_Central_foundation_food_csv_2026-04-30/", "sr_legacy/FoodData_Central_sr_legacy_food_csv_2018-04/"):
    d = R + sub
    food = list(csv.DictReader(open(d + "food.csv", encoding="utf-8")))
    types = collections.Counter(r["data_type"] for r in food)
    print(sub.split("/")[0], "| food.csv columns:", list(food[0].keys()), "| rows by type:", dict(types))
    main = [r for r in food if r["data_type"] in ("foundation_food", "sr_legacy_food")]
    print("  examples:", [r["description"] for r in main[:6]])
    nut = {r["id"]: r for r in csv.DictReader(open(d + "nutrient.csv", encoding="utf-8"))}
    fn = csv.DictReader(open(d + "food_nutrient.csv", encoding="utf-8"))
    cols = fn.fieldnames; ids = {r["fdc_id"] for r in main}; cnt = collections.Counter(); n = 0
    for r in fn:
        if r["fdc_id"] in ids and r["amount"]:
            cnt[r["nutrient_id"]] += 1; n += 1
    print("  food_nutrient columns:", cols, "| rows for main foods:", n)
    print("  most common nutrients:", [(nut[k]["name"], nut[k]["unit_name"], v) for k, v in cnt.most_common(14) if k in nut])
    cat = {r["id"]: r["description"] for r in csv.DictReader(open(d + "food_category.csv", encoding="utf-8"))}
    print("  categories:", collections.Counter(cat.get(r["food_category_id"], "?") for r in main).most_common(8))
