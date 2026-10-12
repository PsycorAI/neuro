"""test_tools_exec.py -- 2026-10-12 (Neuro 57). Offline CPU tests of scripts/demo/tools_exec.py (plan D2).
`--live` adds the two network tools. Also runs every gold call of the held-out tools.jsonl split through its executor
and reports the share that returns ok (the model's own argument style must be executable).
  python temp/test_tools_exec.py [--live]"""
import sys, json, collections
sys.path.insert(0, "/home/glenn/projects/neuro/scripts/demo"); sys.path.insert(0, "/home/glenn/projects/neuro/src")
import tools_exec as X, tool_data as TD
fails = []


def chk(name, args, want=None, ok=True, has=None):
    r = X.run(name, args, offline=True)
    good = r["ok"] == ok and (want is None or r["text"] == want) and (has is None or has.lower() in r["text"].lower())
    if not good:
        fails.append((name, args, r))


for e, w in [("0.17 * 240", "40.8"), ("922 - 89", "833"), ("893 * 25", "22,325"), ("2 ** 10", "1,024"),
             ("(3 + 4) * 5", "35"), ("10 / 4", "2.5"), ("17% of 240", "40.8"), ("sqrt(144)", "12"), ("7 % 3", "1"),
             ("1,200 + 300", "1,500"), ("2^8", "256"), ("-5 + 2", "-3")]:
    chk("calculator", {"expression": e}, w)
for e in ["__import__('os').system('ls')", "1/0", "open('x')", "9**9**9", "a + 1"]:
    chk("calculator", {"expression": e}, ok=False)
for v, a, b, has in [(309.6, "celsius", "fahrenheit", "589.28"), (1, "mile", "km", "1.60934"), (10, "km", "miles", "6.21371"),
                     (5, "feet", "meters", "1.524"), (2, "pounds", "kg", "0.907185"), (1, "gallon", "liters", "3.78541"),
                     (60, "mph", "km/h", "96.5606"), (32, "fahrenheit", "celsius", "= 0 celsius"), (1, "hour", "minutes", "= 60 "),
                     (1024, "megabytes", "gigabytes", "1.024"), (0, "kelvin", "celsius", "-273.15"), (3, "inches", "cm", "7.62")]:
    chk("unit_convert", {"value": v, "from_unit": a, "to_unit": b}, has=has)
chk("unit_convert", {"value": 1, "from_unit": "mile", "to_unit": "kg"}, ok=False)
chk("unit_convert", {"value": "x", "from_unit": "mile", "to_unit": "km"}, ok=False)
for z in ["America/Sao_Paulo", "Tokyo", "EST", "Asia/Kolkata", "london", "UTC", "Pacific", "Sao Paulo", "Europe/Paris",
          "New York"]:
    chk("get_time", {"timezone": z}, has="(")
chk("get_time", {"timezone": "Atlantis"}, ok=False)
chk("set_reminder", {"text": "back up the laptop", "datetime": "tonight at 11pm"}, has="reminder saved")
chk("send_message", {"recipient": "Priya", "body": "dinner at 7"}, has="outbox")
chk("nope", {}, ok=False)
chk("calculator", {"wrong": 1}, ok=False)
r = X.run("get_weather", {"location": "Nowhereville-xyz", "unit": "celsius"}, offline=True)
assert r["ok"] is False and r["source"] == "offline", r
print("define:", X.run("define", {"term": "isotope"}))
print("search offline:", X.run("search", {"query": "capital of France"}, offline=True))
print("translate:", {k: v for k, v in X.run("translate", {"text": "happy birthday", "target_lang": "Spanish"}).items()})
if "--live" in sys.argv:
    print("search live:", X.run("search", {"query": "tallest building in the world"}))
    print("weather live:", X.run("get_weather", {"location": "Hanoi", "unit": "fahrenheit"}))

tools = TD.load_tools()
pairs, _ = TD.load_tool_jsonl("/home/glenn/projects/neuro/training_data/posttrain/tools.jsonl", tools, split="test")
ok, n, bad = collections.Counter(), collections.Counter(), {}
for p in pairs:
    rec = p["rec"]
    if rec["target_mode"] != "TOOL":
        continue
    c = rec["target_call"]
    if c["name"] in ("search", "get_weather", "translate", "define"):
        continue
    r = X.run(c["name"], c["arguments"], offline=True)
    n[c["name"]] += 1; ok[c["name"]] += r["ok"]
    if not r["ok"]:
        bad.setdefault(c["name"], []).append((c["arguments"], r["text"]))
print("gold calls executable:", {k: f"{ok[k]}/{n[k]}" for k in n})
for k, v in bad.items():
    print("  not ok", k, v[:6])
print(f"[{'OK' if not fails else 'FAIL'}] unit cases: {len(fails)} failed")
for f in fails:
    print("  FAIL", f)
