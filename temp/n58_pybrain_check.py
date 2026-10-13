"""n58_pybrain_check.py -- 2026-10-12 (Neuro 58). CPU: does Python code survive this model's text path?
1. tokenizer round trip (32k LUT + subword fallback) of every study target: identical text back? token counts.
2. the SFT formatting: tool_data.format_prompt / format_response + parse_response give the code back.
3. how many study examples fit sft_max_len 256."""
import sys, json, yaml
sys.path.insert(0, "/home/glenn/projects/neuro/src")
import tool_data
from train_sft import make_encoder_pair
c = yaml.safe_load(open("/home/glenn/projects/neuro/configs/brain_mech13.yaml"))
enc, dec = make_encoder_pair(dict(c, lut_subword_fallback=True))
tools = tool_data.load_tools()
rows = [json.loads(l) for l in open("/home/glenn/projects/neuro/training_data/brains/python/study.jsonl", encoding="utf-8")]
rows = [r for r in rows if r.get("kind") == "brain"]
codes = sorted({r["target_text"] for r in rows})
bad = 0; lens = []
for code in codes:
    ids = enc(code); back = dec(ids); lens.append(len(ids))
    if back != code:
        bad += 1
        if bad <= 3: print("ROUND TRIP DIFFERS:\n", repr(code[:120]), "\n", repr(back[:120]))
print(f"tokenizer round trip: {len(codes) - bad} of {len(codes)} solutions identical | tokens per solution min {min(lens)} max {max(lens)}")
r = rows[0]
try:
    p = tool_data.format_prompt(r["instruction"], r["tools"], tools)
except Exception as e:
    p = None; print("format_prompt signature differs:", type(e).__name__, e)
try:
    resp = tool_data.format_response(r, tools, 2)
    print("response:", repr(resp[:160]))
    print("parsed  :", repr(str(tool_data.parse_response(resp))[:200]))
except Exception as e:
    print("format_response / parse_response:", type(e).__name__, e)
if p is not None:
    tot = [len(enc(tool_data.format_prompt(x["instruction"], x["tools"], tools))) + len(enc(x["target_text"])) + 4 for x in rows[:400]]
    print(f"prompt + answer tokens (400 examples): max {max(tot)}, over 256: {sum(t > 256 for t in tot)}")
