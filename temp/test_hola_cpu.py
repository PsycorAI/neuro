# CPU tests for HOLA (2026-10-07, Neuro 53): (1) want_u leaves r/S bit-identical; (2) causality with HOLA ON (changing
# token t changes no output before t, incl. across chunk boundaries); (3) the cache read is live (lambda != 0 changes
# chunk-2 outputs only) and gradients reach the HOLA parameters.
import sys, yaml, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from delta_chunked import gdn2_chunked_dense
from train_gpu import build_model
torch.manual_seed(0)
B, T, D = 2, 96, 32
v, k, q = torch.randn(B, T, D), torch.randn(B, T, D), torch.randn(B, T, D)
b, w, a = torch.rand(B, T, D), torch.rand(B, T, D), torch.rand(B, T, D) * 0.5 + 0.5
M0 = torch.zeros(B, D, D)
r1, S1 = gdn2_chunked_dense(v, k, q, b, w, a, M0, chunk=32)
r2, S2, un = gdn2_chunked_dense(v, k, q, b, w, a, M0, chunk=32, want_u=True)
assert torch.equal(r1, r2) and torch.equal(S1, S2) and un.shape == (B, T), "want_u changed the result"
print("[OK] (1) want_u: r and S bit-identical, |u| shape", tuple(un.shape))

c = yaml.safe_load(open("/home/glenn/projects/neuro/configs/screen_p25o_7L_chunk512_x_H3add.yaml"))
c.update(lif_exact=False, lif_signed=False, lif_levels=1, lif_learn_beta=False, compile=False, amp=False,
         attn_chunk=32, hola_w=8)
m = build_model(c).double()
hb = [bb for bb in m.blocks if getattr(bb, "hola_w", 0)]
assert len(hb) == 1 and hb[0].hola_chunk == 32, "HOLA not enabled on the memory block"
with torch.no_grad():
    hb[0].hola_lam.normal_(0, 1.0); hb[0].hola_gq.normal_(1, 0.3); hb[0].hola_gk.normal_(1, 0.3)
    hb[0].hola_sink.normal_(0, 0.3)
m.eval()
x = torch.randint(0, c["vocab"], (1, 128))
with torch.no_grad():
    base = m(x)
    worst = 0.0
    for t in (31, 32, 63, 64, 100, 127):            # perturb token t -> outputs < t must not move
        x2 = x.clone(); x2[0, t] = (x2[0, t] + 7) % c["vocab"]
        d = (m(x2)[0, :t] - base[0, :t]).abs().max().item()
        worst = max(worst, d)
assert worst < 1e-9, f"FUTURE LEAK: max change before the perturbed token = {worst}"
print(f"[OK] (2) causal with HOLA on: max change before the perturbed token {worst:.1e}")
with torch.no_grad():
    lam = hb[0].hola_lam.clone(); hb[0].hola_lam.zero_()
    off = m(x)
    hb[0].hola_lam.copy_(lam)
d1, d2 = (off[0, :32] - base[0, :32]).abs().max().item(), (off[0, 32:] - base[0, 32:]).abs().max().item()
assert d1 < 1e-9 and d2 > 1e-6, (d1, d2)
print(f"[OK] (3a) cache read live: chunk 1 unchanged ({d1:.1e}), later chunks change ({d2:.2e})")
m.train()
m(x).sum().backward()
g = {n: p.grad.abs().sum().item() for n, p in hb[0].named_parameters() if n.startswith("hola_")}
assert all(v > 0 for v in g.values()), g
print("[OK] (3b) gradients reach", sorted(g))
