# patch_todo_0926c.py -- add grammar cache bug + A/B results so far
p = "docs/to-do.html"
s = open(p, encoding="utf-8").read()
a = "  <li><b>30M FIXED-transformer preview</b>"
b = ("  <li><b>grammar.py mask-cache BUG (fixed)</b>: <code>TokenTable.mask</code> cached on <code>id(grammar)</code>; tool_eval builds a grammar per example and CPython reuses ids &rarr; one tool set's masks/node-ids served to another (IndexError on the first gold-TOOL example; silently wrong masks otherwise). Keyed on the offered set now. Test <code>temp/test_grammar_cache.py</code>: FAIL 31/40 before, PASS after. All prior constrained tool numbers are suspect.</li>\n"
     "  <li><b>SFT A/B results (6M tok, single-pass recall N4/8/16/32/64)</b>: base 100/100/100/72/38; mix 100/100/97/34/3; <b>mix+MQAR rehearsal 100/100/100/59/25</b> (recovers most of the loss). alpaca-only: pending. Tool quality (fixed tool_eval, n=100): <code>logs/sftab_tooleval.log</code>.</li>\n"
     + a)
assert s.count(a) == 1
open(p, "w", encoding="utf-8").write(s.replace(a, b, 1))
print("[OK]")
