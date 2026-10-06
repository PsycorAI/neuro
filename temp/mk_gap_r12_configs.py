# mk_gap_r12_configs.py -- 2026-09-30. Round 12: best-ppl stack (signed + canon, 59.74) + the gentle gap curriculum
# that solved haystack on the unsigned stack (r7_gapw015: 71.10, haystack N32 gap440 99.6).
import re
src, name = "screen_r8_signed_canon_s0_30m", "screen_r12_signed_canon_gapw015_s0_30m"
s = open(f"/home/glenn/projects/neuro/configs/{src}.yaml").read()
assert "mqar_gap" not in s and "mqar_batch" not in s
s = re.sub(r"^# .*\n", "", s, count=1)
s = (f"# {name} -- round 12 (2026-09-30). AUTO-GEN temp/mk_gap_r12_configs.py from {src} (59.74). "
     "Change: + haystack MQAR gap 440 at mqar_weight 0.15 (as r7_gapw015).\n") + s
s, n1 = re.subn(r"^mqar_weight: 0\.5$", "mqar_weight: 0.15", s, flags=re.M)
s, n2 = re.subn(rf"^run_name: {src}$", f"run_name: {name}", s, flags=re.M)
assert n1 == 1 and n2 == 1
s = s.rstrip("\n") + "\nmqar_gap_max: 440\nmqar_gap_ramp_frac: 0.3\nmqar_batch: 16\n"
open(f"/home/glenn/projects/neuro/configs/{name}.yaml", "w").write(s)
print("[OK]", name)
