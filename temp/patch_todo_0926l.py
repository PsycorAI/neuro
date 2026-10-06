# patch_todo_0926l.py -- handoff sync: round-3 final numbers, G: outage, lost arms
p = "docs/to-do.html"
s = open(p, encoding="utf-8").read()
a = ("w0.1 62/25 | pending; w0.2 53/28 | .860/.844/.631/.846; w0.3 56/22 | .885/.844/.615/.885; w0.5 59/25 | .865/.803/.574/.910. "
     "Recall is FLAT from w0.1 to w0.5 &rarr; added <b>w0.05</b> (5080) to find the knee.</li>")
b = ("<b>w0.1 62/25 | .885/.861/.680/.897 (BEST so far)</b>; w0.2 53/28 | .860/.844/.631/.846; w0.3 56/22 | .885/.844/.615/.885; w0.5 59/25 | .865/.803/.574/.910; hiN-only w0.2 66/28 | .860/.811/.623/.885. "
     "Recall FLAT from w0.1 to w0.5; tool cost smallest at w0.1. <b>STOPPED 13:38 by the G: disk-full outage</b> before: <b>w0.05</b> (never trained; 5080 spills at B4 &rarr; 5090 only) and <b>w0.2 seed 1</b> (partial ckpt step 250 written ~13:36 during the fill &rarr; DELETE, rerun fresh). TO-DO: run both (5090, ~70 min each) + n200 tool_eval, then pick W and relaunch.</li>\n"
     "  <li><b style=\"color:#c0392b\">G: DISK-FULL OUTAGE 2026-09-26 13:38&ndash;~16:00</b>: WSL vhdx lives on G:; G: hit 0 free &rarr; WSL down (all runners segfaulted). Glenn freed space (596GB free). TO-DO: before long queues check <code>Get-Volume G</code>; delete screen ckpts after eval; Glenn decides on <code>~/neuro_trash_2026_09_25</code> (494GB) and moving the vhdx off G:.</li>")
assert s.count(a) == 1
open(p, "w", encoding="utf-8").write(s.replace(a, b))
print("[OK]")
