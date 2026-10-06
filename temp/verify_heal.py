import sys, math, yaml, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model, _build_optimizers
c = yaml.safe_load(open("/home/glenn/projects/neuro/configs/heal_v3_363m.yaml"))
s = torch.load("/home/glenn/projects/neuro/models/checkpoints/flagship_v3_363m/step_0271268.pt", map_location="cpu", weights_only=False)
m = build_model(c)
missing, unexpected = m.load_state_dict(s["model"], strict=False)
opts = _build_optimizers(m, c, "cpu")
ok_opt = len(s["opt_states"]) == len(opts)
for o, st in zip(opts, s["opt_states"]): o.load_state_dict(st)
print(f"missing {len(missing)} unexpected {len(unexpected)} | opt states {len(s['opt_states'])}=={len(opts)}: {ok_opt} (loaded OK) | step {s['step']} tok {s['tokens']/1e9:.3f}B")
per = c["batch_size"] * c["grad_accum"] * c["block_size"]; total = c["max_tokens"] // per
warm = int(c.get("lr_warmup_frac", 0.02) * total); lmin = c.get("lr_min_frac", 0.1)
def sc(st):
    p = (st - warm) / max(1, total - warm); return lmin + (1 - lmin) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
print(f"total_steps {total} | LR scale at resume {sc(s['step']):.3f} -> end {sc(total):.3f} | steps to go {total - s['step']} ({(total - s['step'])*per/1e9:.2f}B tok)")
print("ckpt cfg vs heal cfg differing keys:", sorted(k for k in set(c) | set(s['cfg']) if c.get(k) != s['cfg'].get(k)))
