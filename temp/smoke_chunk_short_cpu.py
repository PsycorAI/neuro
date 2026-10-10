# CPU smoke (2026-10-07): stochastic chunk sizes -- the same model runs forward/backward at attn_chunk 512 and 128
# (T1024) and the outputs of chunk 1 agree for the first 128 tokens (block-diagonal causal attention).
import sys, yaml, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model
c = yaml.safe_load(open("/home/glenn/projects/neuro/configs/screen_p25o_7L_chunk512_x_H3add.yaml"))
c.update(lif_exact=False, lif_signed=False, lif_levels=1, lif_learn_beta=False, compile=False, amp=False)
m = build_model(c)
x = torch.randint(0, c["vocab"], (1, 1024))
outs = {}
for C in (512, 128):
    for b in m.blocks:
        if hasattr(b, "qkv"):
            b.attn_chunk = C
    o = m(x)
    o.float().logsumexp(-1).mean().backward()
    outs[C] = o.detach()
d = (outs[512][0, :128] - outs[128][0, :128]).abs().max().item()
print(f"[OK] chunk 512 and 128 both run fwd+bwd; max |diff| on the first 128 tokens = {d:.2e} (should be ~0)")
