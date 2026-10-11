#!/bin/bash
# get_teacher.sh -- 2026-10-11 (Neuro 55). Run ONCE on the instance. The conversion phase (and the conversion arms of
# cloud/bench_rc3.sh) distil from meta-llama/Llama-3.2-1B, which is GATED, and run_v4.sh runs with HF_HUB_OFFLINE=1,
# so the teacher must already be in /workspace/hf. Two ways to get it there:
#   A. on the instance (1 min):   HF_TOKEN=hf_xxx bash cloud/get_teacher.sh        (a token with access to the model)
#   B. from home, no token (2.4 GB over the DSL line = ~4 h; fine in the background, the dense phase does not need it):
#        TEACHER=1 bash cloud/push_from_home.sh <host> <port>
# Either way this script then checks that the teacher loads OFFLINE, the way the trainer will load it. It also caches
# the NousResearch/Meta-Llama-3-8B tokenizer files (not gated) that temp/nl_fact_eval.py and the demo use.
export HF_HOME=${HF_HOME:-/workspace/hf}
PY=${PY:-/workspace/venv/bin/python}
if [ -n "$HF_TOKEN" ]; then
  $PY - <<'EOF' || { echo "[teacher] download FAILED (token without access to meta-llama/Llama-3.2-1B?)"; exit 1; }
from huggingface_hub import snapshot_download
print("[teacher]", snapshot_download("meta-llama/Llama-3.2-1B", allow_patterns=["*.json", "*.safetensors"]))
print("[tokenizer]", snapshot_download("NousResearch/Meta-Llama-3-8B", allow_patterns=["tokenizer*", "special_tokens_map.json"]))
EOF
fi
HF_HUB_OFFLINE=1 $PY - <<'EOF' && echo "[teacher] OK: loads offline from $HF_HOME" || { echo "[teacher] NOT usable offline yet"; exit 1; }
import torch
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", dtype=torch.bfloat16)
print(f"[teacher] {sum(p.numel() for p in m.parameters()) / 1e9:.2f}B params")
EOF
