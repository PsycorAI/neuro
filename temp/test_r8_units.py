"""test_r8_units.py -- 2026-09-29 unit tests for the staged r8 flags (run with GAP_STAGE=temp/stage_r8).
1. attn_window: W>=T == full causal (bit-exact path); 0<W<T == manual banded softmax reference; W=0 unchanged.
2. SequentialStream.skip(n) leaves cursors/rng exactly where n next() calls would.
3. Stateful forward: carried state changes window-2 logits; zero-state reproduces the plain forward."""
import sys, os, yaml, math, torch, numpy as np, torch.nn.functional as F
sys.path.insert(0, "/home/glenn/projects/neuro/src")
if os.environ.get("GAP_STAGE"): sys.path.insert(0, os.environ["GAP_STAGE"])
from train_gpu import build_model, BIN
import hybrid_attn, data_stream
print("hybrid_attn from", hybrid_attn.__file__, "| data_stream from", data_stream.__file__)
c = yaml.safe_load(open("/home/glenn/projects/neuro/configs/screen_gap_hyb_full_s0_30m.yaml"))
torch.manual_seed(0); m = build_model(c).cuda().eval()
ab = [b for b in m.blocks if hasattr(b, "qkv")][0]
x = torch.randn(2, 96, ab.d, device="cuda")
with torch.no_grad():
    ab.attn_window = 0; full = ab._attn(x)
    ab.attn_window = 96; same = ab._attn(x)
    ab.attn_window = 16; win = ab._attn(x)
    # manual reference for W=16
    B, T, D = x.shape; H, Dh = ab.n_heads, ab.d_head
    q, k, v = ab.qkv(x).chunk(3, -1)
    q = q.view(B, T, H, Dh).transpose(1, 2); k = k.view(B, T, H, Dh).transpose(1, 2); v = v.view(B, T, H, Dh).transpose(1, 2)
    q = hybrid_attn._apply_rope(ab.q_norm(q), ab.rope_cos, ab.rope_sin); k = hybrid_attn._apply_rope(ab.k_norm(k), ab.rope_cos, ab.rope_sin)
    s = (q @ k.transpose(-1, -2)) / math.sqrt(Dh)
    i = torch.arange(T, device="cuda"); d = i[:, None] - i[None, :]
    s = s.masked_fill(~((d >= 0) & (d < 16)), float("-inf"))
    ref = ab.o_proj((s.softmax(-1) @ v).transpose(1, 2).reshape(B, T, D))
    ab.attn_window = 0
print(f"1. W=T vs full max diff {(same - full).abs().max().item():.2e} (expect 0) | W=16 vs manual {(win - ref).abs().max().item():.2e} "
      f"(expect ~1e-6) | W=16 vs full differ at t>=16: {(win - full)[:, 16:].abs().max().item():.2e}, equal at t<16: {(win - full)[:, :16].abs().max().item():.2e}")

a = data_stream.SequentialStream(BIN, vocab=16384, batch_size=4, seed=3, margin_tokens=10)
b = data_stream.SequentialStream(BIN, vocab=16384, batch_size=4, seed=3, margin_tokens=10)
a.train_hi = b.train_hi = int(a.cursors.max()) + 5000       # force wraps within the test
for _ in range(40): a.next(512, "cpu")
b.skip(40, 512)
print(f"2. skip == next: cursors {np.array_equal(a.cursors, b.cursors)} | rng state equal "
      f"{a.rng.bit_generator.state == b.rng.bit_generator.state}")

tok = torch.randint(0, 16384, (2, 1024), device="cuda", generator=torch.Generator(device="cuda").manual_seed(5))
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    plain2 = m(tok[:, 512:]).float()
    zero2, _ = m(tok[:, 512:], initial_state=None, return_final_state=True)
    _, st = m(tok[:, :512], return_final_state=True)
    carry2, _ = m(tok[:, 512:], initial_state=st, return_final_state=True)
print(f"3. zero-state == plain: {(zero2.float() - plain2).abs().max().item():.2e} (expect 0) | carried differs: "
      f"{(carry2.float() - plain2).abs().max().item():.2e} (expect >0)")
