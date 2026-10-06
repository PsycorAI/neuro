#!/bin/bash
# relaunch_posttrain_0926.sh -- 2026-09-26: re-run v3 post-training with the SFT recall fix.
# - archives the recall-broken SFT (kept for comparison) and clears its .done marker
# - sft_v3_363m.yaml: mqar_aug true + mqar_weight $W (A/B @6M tok, single-pass N32/N64:
#   none 34/3, w0.2 53/28, w0.5 59/25; tool cost measured in logs/sftab_tooleval.log)
# - micro-batch stays 4 x accum 8 (B8 spilled the 5090 -- tested 2026-09-26)
# - then the normal resumable pipeline (SFT -> ORPO -> OPD -> heads), GPU 0.
#   W=0.2 bash temp/relaunch_posttrain_0926.sh
set -eu
W=${W:?set W=<mqar_weight>, e.g. W=0.2}
cd /home/glenn/projects/neuro
CK=models/checkpoints
[ -d $CK/sft_v3_363m_norehearsal ] || mv $CK/sft_v3_363m $CK/sft_v3_363m_norehearsal
rm -f logs/v3_posttrain.done.d/sft
rmdir $CK/orpo_v3_363m 2>/dev/null || true
cp -n configs/sft_v3_363m.yaml temp/sft_v3_363m.norehearsal.yaml
W=$W /home/glenn/projects/bdh/venv/bin/python - <<'PY'
import os, re
p = "configs/sft_v3_363m.yaml"
s = open(p).read()
for a, b in [
    (r"^mqar_aug: false .*$", "mqar_aug: true            # 2026-09-26 RECALL FIX: MQAR rehearsal during SFT (screen_sftab A/B)"),
    (r"^mqar_weight: 0\.0$", f"mqar_weight: {float(os.environ['W'])}"),
]:
    s, n = re.subn(a, b, s, flags=re.M); assert n == 1, a
open(p, "w").write(s)
print("[OK] sft_v3_363m.yaml patched, mqar_weight", os.environ["W"])
PY
echo "[relaunch $(date +%H:%M:%S)] post-training relaunch (SFT recall fix, mqar_weight $W)" >> logs/v3_posttrain.log
GPU=0 bash temp/run_v3_posttrain.sh
