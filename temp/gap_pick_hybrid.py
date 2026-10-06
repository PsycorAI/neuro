"""gap_pick_hybrid.py -- 2026-09-27 ppl-gap plan A3: pick the optimizer for the 2-layer [spk, attn] hybrid
from A1 (NorMuon 0.02 / AdamW 1e-3) vs A2 (AdamW 1e-3), then write configs/screen_honest_hybrid_s0_30m.yaml.
Rule: among A1/A2 that FINISHED (log has 'done:') and pass the recall gate (single-pass N64 >= 80%), take the lowest
final val_ppl. If neither qualifies, fall back to the control (NorMuon 0.005 / AdamW 6e-4). Prints the choice."""
import re, sys
N = "/home/glenn/projects/neuro/"
GAPLOG = N + "logs/gap_screen.log"
CONTROL = "screen_honest_lifexact_s0_30m"
ARMS = ["screen_honest_mu02_s0_30m", "screen_honest_adamw_s0_30m"]

def final_ppl(run):
    try:
        s = open(N + f"logs/{run}.log", encoding="utf-8").read()
    except FileNotFoundError:
        return None
    if "done:" not in s:
        return None
    v = re.findall(r"eval\] val_ppl ([\d.]+)", s)
    return float(v[-1]) if v else None

def recall_n64(run):
    try:
        lines = open(GAPLOG, encoding="utf-8").readlines()
    except FileNotFoundError:
        return None
    for line in lines:     # first row for the run = accuracy table
        if line.strip().startswith(run + " |"):
            return float(line.rstrip().split("|")[-1].strip().rstrip("%"))
    return None

cands = []
for r in ARMS:
    p, rc = final_ppl(r), recall_n64(r)
    print(f"  {r}: final val_ppl {p} | recall N64 {rc}")
    if p is not None and rc is not None and rc >= 80:
        cands.append((p, r))
src = min(cands)[1] if cands else CONTROL
print(f"  hybrid optimizer source: {src}{'' if cands else ' (fallback: no A arm qualified)'}")

s = open(N + f"configs/{src}.yaml", encoding="utf-8").read()
body = "\n".join(l for l in s.splitlines() if not l.startswith("#"))
body = re.sub(r"^run_name:.*$", "run_name: screen_honest_hybrid_s0_30m", body, flags=re.M)
extra = """
# --- A3 hybrid: layer 1 = full-softmax attention block (RoPE, QK-norm), layer 0 = spiking/GDN-2 ---
attention_layer_positions:
- 1
attention_heads: 8
attention_max_seq_len: 4096
attention_qk_norm: true
attn_single_residual: true     # 2026-09-25 double-residual fix
final_norm: true               # attention stacks need a pre-head norm
spike_rate_undiluted: true     # sparsity target 0.05 applies to the spiking layer only (not diluted by the attn stub)
"""
hdr = (f"# screen_honest_hybrid_s0_30m -- ppl-gap plan A3 (2026-09-27, docs/ppl_gap_plan_2026_09_27.md). AUTO-GEN\n"
       f"# temp/gap_pick_hybrid.py from {src} (optimizer winner). Change: 2-layer [spk, attn] hybrid.\n")
open(N + "configs/screen_honest_hybrid_s0_30m.yaml", "w", encoding="utf-8", newline="").write(hdr + body + "\n" + extra)
print("[OK] configs/screen_honest_hybrid_s0_30m.yaml")
