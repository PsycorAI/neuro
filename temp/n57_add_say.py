"""n57_add_say.py -- 2026-10-12 (Neuro 57). Adds the protocol-v2 "say" sentence to each tool in tools.json (plan D1).
Line-level edit so the file's layout is kept. Idempotent."""
P = "/home/glenn/projects/neuro/training_data/posttrain/tools.json"
SAY = {"search": "I will look that up for you.", "calculator": "I will calculate that for you.",
       "unit_convert": "I will convert that for you.", "get_weather": "I will check the weather for you.",
       "get_time": "I will check the time for you.", "define": "I will look up that word for you.",
       "translate": "I will translate that for you.", "set_reminder": "I will set that reminder for you.",
       "send_message": "I will send that message for you."}
s = open(P).read()
if '"say"' in s:
    print("[OK] already has say"); raise SystemExit
for name, say in SAY.items():
    old = f'      "name": "{name}",\n'
    assert s.count(old) == 1, name
    s = s.replace(old, old + f'      "say": "{say}",\n')
s = s.replace('"version": 1,', '"version": 2,')
open(P, "w").write(s)
import json; t = json.load(open(P))["tools"]; assert all("say" in x for x in t)
print("[OK] tools.json: say added to", len(t), "tools")
