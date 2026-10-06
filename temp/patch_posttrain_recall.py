# patch_posttrain_recall.py -- 2026-09-26: pipeline recall eval -> single-pass, base as anchor
p = "temp/run_v3_posttrain.sh"
s = open(p).read()
a = '  $PY scripts/recall_eval.py --replay split --runs "$run" --B 32 >> "$LOG" 2>&1\n'
b = ('  # single-pass (2026-09-26): split replay has no KV carry -> invalid on attention stacks.\n'
     '  # The base row anchors how much recall each stage costs.\n'
     '  $PY scripts/recall_eval.py --runs flagship_v3_363m,"$run" --B 32 >> "$LOG" 2>&1\n')
assert s.count(a) == 1
open(p, "w").write(s.replace(a, b))
print("[OK]", p)
