#!/bin/bash
# Neuro 57: read-only survey 2 for the demo plan (tool-list lengths, prompt lengths, Linear names in the RC3 model).
cd /home/glenn/projects/neuro || exit 1
PY=/home/glenn/projects/bdh/venv/bin/python
$PY - <<'EOF'
import json, collections, sys, yaml
sys.path.insert(0, "src")
c = collections.Counter(); chars = []
for line in open("training_data/posttrain/tools.jsonl", encoding="utf-8"):
    r = json.loads(line); c[len(r["tools"])] += 1; chars.append(len(r["instruction"]))
print("offered-tool list lengths:", dict(sorted(c.items())))
chars.sort(); print("instruction chars: median", chars[len(chars)//2], "p95", chars[int(len(chars)*.95)], "max", chars[-1])
import tool_data
tools = tool_data.load_tools()
blk = tool_data.render_tools_block([t["name"] for t in tools], tools)
print("all-9 tools block chars:", len(blk))
import torch
from train_gpu import build_model
cfg = yaml.safe_load(open("configs/sft_v4_reh13.yaml"))
m = build_model(cfg)
names = collections.Counter()
for n, mod in m.named_modules():
    if isinstance(mod, torch.nn.Linear):
        parts = n.split(".")
        key = ".".join(p for p in parts if not p.isdigit() or parts[0] != "blocks" or parts.index(p) != 1)
        names[(key, mod.in_features, mod.out_features)] += 1
for (k, i, o), cnt in sorted(names.items()):
    print(f"  Linear {k:40s} {i:6d} -> {o:6d}  x{cnt}")
print("block kinds:", [type(b).__name__[:6] for b in m.blocks])
tot = sum(p.numel() for p in m.parameters())
spk = sum(p.numel() for p in m.blocks[6].parameters())
print(f"params total {tot/1e6:.1f}M | memory layer (block 6) {spk/1e6:.1f}M | embed {m.embed.weight.numel()/1e6:.1f}M")
EOF
grep -n -E "format_response|dumps_call|MODE_TOOL|parse_response" scripts/gen_teacher_data.py scripts/head_eval.py src/train_heads.py src/train_opd.py scripts/demo/*.py scripts/*.py 2>/dev/null | cut -c1-150 | head -n 30
true
