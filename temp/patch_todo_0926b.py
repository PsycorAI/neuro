# patch_todo_0926b.py -- update the A/B line: sequential arms + SFT MQAR rehearsal built
p = "docs/to-do.html"
s = open(p, encoding="utf-8").read()
a = "A/B screen (alpaca-only vs mix, 6M tok, from the 10B base) RUNNING: <code>temp/run_sftab.sh</code> &rarr; <code>logs/sftab_screen.log</code>."
b = ("A/B screen RUNNING (5090, SEQUENTIAL &mdash; two 426M arms concurrently spilled past 32GB): "
     "mix &rarr; mix+MQAR rehearsal &rarr; [1B probe] &rarr; alpaca-only, 6M tok each from the 10B base: "
     "<code>temp/run_sftab.sh</code> &rarr; <code>logs/sftab_screen.log</code>. "
     "<b>BUILT</b>: <code>mqar_aug</code> in <code>train_sft</code> (MQAR recall rehearsal during SFT, default off; "
     "smoke-tested 30M: step-1 loss 7.29 on vs 6.07 off).")
assert s.count(a) == 1
open(p, "w", encoding="utf-8").write(s.replace(a, b))
print("[OK]")
