"""demo/build_search_pack.py -- 2026-10-12 (Neuro 57, plan D2). Offline pack for the search tool: the first paragraph
of every Simple English Wikipedia article (HF dataset wikimedia/wikipedia, 20231101.simple; text is CC BY-SA) in an
SQLite FTS5 table docs(title, body) at data/demo/search_pack.db -- what tools_exec.search_offline reads.
  python scripts/demo/build_search_pack.py"""
import os, re, sqlite3
from datasets import load_dataset
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data", "demo", "search_pack.db")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")
if os.path.exists(OUT):
    os.remove(OUT)
con = sqlite3.connect(OUT)
con.execute("create virtual table docs using fts5(title, body, tokenize='porter unicode61')")
n = 0
for r in ds:
    paras = [p.strip() for p in r["text"].split("\n") if len(p.strip()) >= 80]
    if not paras or ":" in r["title"][:12] and r["title"].split(":")[0] in ("Category", "Template", "Wikipedia"):
        continue
    body = re.sub(r"\s+", " ", paras[0])[:700]
    con.execute("insert into docs values (?, ?)", (r["title"], body)); n += 1
con.commit(); con.close()
print(f"[OK] {n} articles -> {OUT} ({os.path.getsize(OUT) / 1e6:.0f} MB)")
