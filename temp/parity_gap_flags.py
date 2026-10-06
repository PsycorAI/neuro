"""parity_gap_flags.py -- 2026-09-28: prove the ppl-gap flags (ff_conv, swa_read, pre_norm, mem_out_gate, lif levels/
signed/learnable beta) leave every existing config byte-identical when off.
  python temp/parity_gap_flags.py save   -> writes /tmp/parity_gap_ref.pt from the CURRENT code
  python temp/parity_gap_flags.py check  -> recomputes and compares (logits/loss exact; grads exact except atomics)
Configs: lif_exact control, FPT (indepwin), 2-layer hybrid. Train-mode fwd+bwd, bf16 autocast, fixed seed/input."""
import sys, yaml, torch, torch.nn.functional as F
sys.path.insert(0, "/home/glenn/projects/neuro/src")
import os
if os.environ.get("GAP_STAGE"): sys.path.insert(0, os.environ["GAP_STAGE"])
from train_gpu import build_model
CFG = ["screen_honest_lifexact_s0_30m", "screen_indepwin_s0_30m", "screen_honest_hybrid_s0_30m"]
REF = os.environ.get("PARITY_REF", "/tmp/parity_gap_ref.pt")

def run(name):
    c = yaml.safe_load(open(f"/home/glenn/projects/neuro/configs/{name}.yaml"))
    torch.manual_seed(0)
    m = build_model(c).to("cuda").train()
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randint(0, c["vocab"], (4, 513), device="cuda", generator=g)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg, sr = m(x[:, :-1], return_stats=True)
        loss = F.cross_entropy(lg.float().reshape(-1, c["vocab"]), x[:, 1:].reshape(-1)) + 0.5 * sr
    loss.backward()
    grads = {n: p.grad.detach().double().norm().item() for n, p in m.named_parameters() if p.grad is not None}
    return {"logits": lg.detach().float().cpu(), "loss": loss.item(), "grads": grads,
            "names": [n for n, _ in m.named_parameters()]}

out = {n: run(n) for n in CFG}
if sys.argv[1] == "save":
    torch.save(out, REF); print("[saved]", REF)
else:
    ref = torch.load(REF); ok = True
    for n in CFG:
        a, b = ref[n], out[n]
        same_names = a["names"] == b["names"]
        same_logits = torch.equal(a["logits"], b["logits"])
        gd = max(abs(a["grads"][k] - b["grads"][k]) / (abs(a["grads"][k]) + 1e-30) for k in a["grads"])
        worst = max(a["grads"], key=lambda k: abs(a["grads"][k] - b["grads"][k]) / (abs(a["grads"][k]) + 1e-30))
        good = same_names and same_logits and a["loss"] == b["loss"] and gd < 1e-4
        ok &= good
        print(f"{'PASS' if good else 'FAIL'} {n}: params_same={same_names} logits_equal={same_logits} "
              f"loss {a['loss']:.6f}/{b['loss']:.6f} max_grad_rel_diff={gd:.2e} ({worst})")
    print("ALL PASS" if ok else "PARITY FAILED")
