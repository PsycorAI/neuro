"""test_act_ckpt.py -- act_ckpt must give the same loss/grads as the plain path (30M, eager, bf16 autocast),
and report peak allocated memory for both."""
import sys, yaml, torch, torch.nn.functional as F
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model
torch.set_float32_matmul_precision("high")
c = yaml.safe_load(open("/home/glenn/projects/neuro/configs/screen_indepwin_s0_30m.yaml"))
res = {}
for flag in (False, True):
    c["act_ckpt"] = flag
    torch.manual_seed(0); m = build_model(c).cuda().train()
    torch.manual_seed(1); x = torch.randint(0, c["vocab"], (16, 512), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(x)
        out = out[0] if isinstance(out, tuple) else out
        loss = F.cross_entropy(out[:, :-1].reshape(-1, c["vocab"]).float(), x[:, 1:].reshape(-1))
    loss.backward()
    res[flag] = (loss.item(), {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None},
                 torch.cuda.max_memory_allocated() / 2**30)
    del m
l0, g0, m0 = res[False]; l1, g1, m1 = res[True]
worst = max(((g1[n] - g0[n]).abs().max() / (g0[n].abs().max() + 1e-12)).item() for n in g0)
exact = all(torch.equal(g0[n], g1[n]) for n in g0)
print(f"loss plain {l0:.6f} | act_ckpt {l1:.6f} | grads bit-identical: {exact} | worst rel grad diff {worst:.2e}")
print(f"peak allocated: plain {m0:.2f} GB | act_ckpt {m1:.2f} GB")
