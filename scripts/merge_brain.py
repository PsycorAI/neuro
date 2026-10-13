"""merge_brain.py -- 2026-10-12 (Neuro 58; demo plan D7). Write a plain checkpoint = base model with one brain
(LoRA adapter) merged into its weights, so every existing eval (scripts/tool_eval.py, temp/energy_eval_v2.py, the
r25 evals) can measure the model "with the brain on" without knowing about adapters.
  python scripts/merge_brain.py --base sft_v4_reh13_v2tr --brain brain_py13 --out merged_py13
-> models/checkpoints/<out>/step_0000001.pt  (model + the base's cfg; no optimizer state)"""
import os, sys, glob, argparse
import torch
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import lora
from train_gpu import build_model
ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--brain", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
latest = lambda run, pat: sorted(glob.glob(os.path.join(ROOT, "models", "checkpoints", run, pat)))[-1]
blob = torch.load(latest(a.base, "step_*.pt"), map_location="cuda", weights_only=False); c = dict(blob["cfg"])
m = build_model(c).to("cuda"); m.load_state_dict(blob["model"], strict=False)
ad = torch.load(latest(a.brain, "adapter_step_*.pt"), map_location="cuda", weights_only=False)
lora.inject_lora(m, r=ad["lora_r"], alpha=ad["lora_alpha"], targets=tuple(ad["lora_targets"])); m.to("cuda")
lora.load_lora_state_dict(m, ad["lora"]); n = lora.merge_lora(m)
sd = {k: v.detach().cpu() for k, v in m.state_dict().items() if "lora_" not in k}
missing = [k for k in blob["model"] if k not in sd]
assert not missing, f"merged model lost keys: {missing[:3]}"
c.update(run_name=a.out, lora_enabled=False)
d = os.path.join(ROOT, "models", "checkpoints", a.out); os.makedirs(d, exist_ok=True)
torch.save({"model": {k: sd[k] for k in blob["model"]}, "cfg": c, "step": 1, "tokens": 0}, os.path.join(d, "step_0000001.pt"))
print(f"[OK] {a.out}: {a.brain} merged into {n} layers of {a.base} -> {d}/step_0000001.pt")
