"""parity_r16_flags.py -- 2026-10-02. CPU parity + functional tests for the round-16 builds b1-b7
(docs/v4_plan_review_2026_10_01.md 6.5). Usage:
  python temp/parity_r16_flags.py save    # BEFORE the code change: reference logits with the flags absent
  python temp/parity_r16_flags.py check   # AFTER: flags off must be bit-identical; flags on must behave
The LIF Triton kernels are swapped for an exact torch reference so this runs on CPU (GPUs stay busy)."""
import sys, math, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
import lif_exact as LE

def lif_ml_ref(cur, beta, vth=1.0, K=None, alpha=2.0, gamma=1.0, levels=1, signed=False):
    B, T, N = cur.shape
    beta = beta if torch.is_tensor(beta) else torch.full((N,), float(beta))
    mem = torch.zeros(B, N, dtype=cur.dtype); outs = []
    for t in range(T):
        pre = beta * mem + cur[:, t]
        hard = torch.clamp(torch.floor(pre / vth), 0, levels)
        if signed:
            hard = hard - torch.clamp(torch.floor(-pre / vth), 0, levels)
        s = hard + (torch.tanh(pre) - torch.tanh(pre).detach())   # surrogate grad (tests only)
        outs.append(s); mem = pre - gamma * hard * vth
    return torch.stack(outs, 1), mem
LE.lif_exact_ml = lif_ml_ref
LE.lif_exact = lambda cur, beta, vth=1.0, K=None, alpha=2.0, gamma=1.0: lif_ml_ref(cur, beta, vth, gamma=gamma)
from train_gpu import build_model

V = 512
BASE = dict(arch="spiking", vocab=V, d=64, n_neurons=128, d_mem=64, n_layers=4, block_size=64,
            use_fpt=True, lif_exact=True, lif_signed=True, lif_levels=4, delta_rule=True, decay_gate=True,
            beta_floor=0.1, gdn2_enabled=True, gdn2_chunked=True, gdn2_chunk_size=16, gate_rank=4,
            engram_v2_enabled=True, engram_v2_K=256, engram_v2_heads=2, engram_v2_rank=8,
            attention_layer_positions=[1, 2, 3], attention_heads=4, attn_single_residual=True,
            final_norm=True, spike_rate_undiluted=True, pre_norm=True, mem_out_gate=4, canon_a=4)
CASES = {
    "hyb": BASE,
    "spk": dict(BASE, attention_layer_positions=[], n_layers=2),
    "sglu": dict(BASE, attn_ffn_spikeglu_k=8, attn_ffn_hidden=128, attn_topk=4),
    "lscale": dict(BASE, spk_layerscale=0.1),
    "tfmr": dict(BASE, attention_layer_positions=[0, 1, 2, 3], delta_rule=False, decay_gate=False,
                 gdn2_enabled=False, gdn2_chunked=False, beta_floor=0.0, depth_scaled_init=True),
}
REF = "/home/glenn/projects/neuro/temp/parity_r16_ref.pt"

def run(c, x, seed=0, stats=False):
    torch.manual_seed(seed); m = build_model(c); m.eval()
    with torch.no_grad():
        out = m(x, return_stats=True) if stats else m(x)
    if stats:
        return out[0], out[1], m
    return (out[0] if isinstance(out, tuple) else out), m

torch.manual_seed(123); X = torch.randint(0, V, (2, 64))
mode = sys.argv[1]
if mode == "save":
    ref = {}
    for k, c in CASES.items():
        lo, sr, _ = run(c, X, stats=True)
        ref[k] = lo; ref[k + "_sr"] = sr
    torch.save(ref, REF); print("[OK] saved reference:", list(ref)); sys.exit()

ref = torch.load(REF)
for k, c in CASES.items():
    lo, sr, _ = run(c, X, stats=True)
    d = (lo - ref[k]).abs().max().item(); ds = (sr - ref[k + "_sr"]).abs().max().item()
    print(f"parity flags-off {k}: max|diff| logits {d} spike_rate {ds}"); assert d == 0.0 and ds == 0.0, k
# explicit defaults also bit-identical
EXPL = dict(engram_v2_hash="lo", attn_out_gate=False, attn_canon=0, spk_out_norm=True,
            attn_ffn_spikeglu_lif=False)
for k in ("hyb", "sglu"):
    d = (run(dict(CASES[k], **EXPL), X)[0] - ref[k]).abs().max().item()
    print(f"parity explicit-defaults {k}: {d}"); assert d == 0.0, k

def train_step(c):
    torch.manual_seed(0); m = build_model(c); m.train()
    out = m(X, return_stats=True); logits, sr = out[0], out[1]
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, V), X[:, 1:].reshape(-1)) + 0.5 * sr
    loss.backward(); g = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
    assert torch.isfinite(loss) and g > 0
    return m, loss.item(), sr.item()

def causal(c):
    torch.manual_seed(0); m = build_model(c); m.eval()
    x2 = X.clone(); x2[:, 40] = (x2[:, 40] + 7) % V
    with torch.no_grad():
        a = m(X); b = m(x2)
    a = a[0] if isinstance(a, tuple) else a; b = b[0] if isinstance(b, tuple) else b
    return (a[:, :40] - b[:, :40]).abs().max().item()

def attn_blocks(m):
    return [b for b in m.blocks if hasattr(b, "qkv")]

# b1: Engram high-bit hash -- ids x and x+K must NOT share a bucket (they always do with the low-bit hash)
from engram_v2 import EngramV2
for hi in (False, True):
    e = EngramV2(d=16, vocab=4096, orders=(2, 3), K=256, heads=2, rank=4, hash_hi=hi)
    a = torch.randint(0, 256, (4, 32)); b = a + 256
    same = (e._hash_order_head(a, 2, 0) == e._hash_order_head(b, 2, 0)).float().mean().item()
    h = e._hash_order_head(torch.randint(0, 4096, (64, 64)), 3, 3) - e.table_base[3]
    print(f"engram hash_hi={hi}: share(x, x+K same bucket) {same:.3f}; bucket range [{h.min().item()}, {h.max().item()}] "
          f"occupied {h.unique().numel()}/256")
    assert h.min() >= 0 and h.max() < 256
    assert (same == 1.0) if not hi else (same < 0.05)
c = dict(BASE, engram_v2_hash="hi"); m, l, _ = train_step(c)
print(f"b1 engram_v2_hash=hi: loss {l:.3f}; causal leak {causal(c)}")

# b2: gated attention output (sigmoid, elementwise)
c = dict(BASE, attn_out_gate=True); m, l, _ = train_step(c)
gb = attn_blocks(m)[0]
assert gb.attn_gate.weight.grad is not None and gb.attn_gate.weight.grad.abs().sum() > 0
print(f"b2 attn_out_gate: loss {l:.3f}; gate grad ok; causal leak {causal(c)}")

# b3: SpikeGLU k-ramp (k_cur set by the trainer; None -> target k)
c = dict(BASE, attn_ffn_spikeglu_k=8, attn_ffn_hidden=128); torch.manual_seed(0); m = build_model(c); m.eval()
blk = attn_blocks(m)[0]; seen = {}
blk.ffn.w2.register_forward_hook(lambda mod, inp, out: seen.__setitem__("h", inp[0].detach()))
for kc, want in ((None, 8), (128, 128), (32, 32)):
    for b in attn_blocks(m): b.ffn.k_cur = kc
    with torch.no_grad():
        m(X)
    nz = (seen["h"] != 0).sum(-1).max().item()
    print(f"b3 spikeglu k_cur={kc}: max active units/token {nz} (<= {want})"); assert nz <= want
    if kc == 128: assert nz > 8
from train_gpu import spikeglu_k_at
ks = [spikeglu_k_at(dict(attn_ffn_spikeglu_k=256, attn_ffn_hidden=2752, attn_ffn_spikeglu_ramp_start=0.6,
                         attn_ffn_spikeglu_ramp_end=0.9), s, 1000) for s in (0, 599, 600, 750, 899, 900, 999)]
print(f"b3 k schedule (start .6, end .9) at steps 0/599/600/750/899/900/999: {ks}")
assert ks[0] == 2752 and ks[1] == 2752 and ks[-2] == 256 and ks[-1] == 256 and 256 < ks[3] < 2752
assert spikeglu_k_at(dict(attn_ffn_spikeglu_k=256), 10, 1000) is None

# b4: SpikeGLU with LIF gate (threshold spikes, learnable leak) + its spikes in the sparsity rate
c = dict(BASE, attn_ffn_spikeglu_k=8, attn_ffn_hidden=128, attn_ffn_spikeglu_lif=True, lif_levels=1)
m, l, sr = train_step(c)
blk = attn_blocks(m)[0]
assert blk.ffn.beta_logit.grad is not None and blk.ffn.beta_logit.grad.abs().sum() > 0
seen = {}; blk.ffn.w2.register_forward_hook(lambda mod, inp, out: seen.__setitem__("h", inp[0].detach()))
c_off = dict(c, attn_ffn_spikeglu_lif=False)
_, sr_off, _ = run(c_off, X, stats=True); _, sr_on, m2 = run(c, X, stats=True)
print(f"b4 spikeglu_lif: loss {l:.3f}; beta grad ok; spike_rate with DSN {sr_on.item():.4f} vs H only {sr_off.item():.4f}; "
      f"causal leak {causal(c)}")
assert abs(sr_on.item() - sr_off.item()) > 1e-6

# b5: spiking block without its output LayerNorm (linear atoms)
c = dict(BASE, spk_out_norm=False); m, l, _ = train_step(c)
assert all(isinstance(b.norm, torch.nn.Identity) for b in m.blocks if not hasattr(b, "qkv"))
print(f"b5 spk_out_norm=False: loss {l:.3f}; causal leak {causal(c)}")
c = dict(BASE, spk_out_norm=False, spk_layerscale=0.1); m, l, _ = train_step(c)
print(f"b5 + spk_layerscale 0.1: loss {l:.3f}")

# b7: canon taps in attention blocks (zero-init -> identical logits at init; grads flow)
c = dict(BASE, attn_canon=4)
d = (run(c, X)[0] - ref["hyb"]).abs().max().item()
m, l, _ = train_step(c); cw = attn_blocks(m)[0].canon_w
print(f"b7 attn_canon=4: init logits vs ref {d}; loss {l:.3f}; canon grad {cw.grad.abs().sum().item() > 0}; causal leak {causal(c)}")
assert d == 0.0 and cw.grad.abs().sum() > 0

# everything on together (the C0 shape) + causality
C0 = dict(BASE, engram_v2_hash="hi", attn_out_gate=True, attn_canon=4, spk_out_norm=False, lif_levels=1,
          attn_topk=4, attn_ffn_spikeglu_k=8, attn_ffn_hidden=128)
m, l, _ = train_step(C0); lk = causal(C0)
print(f"C0 all-on: loss {l:.3f}; causal leak {lk}"); assert lk == 0.0
c = dict(C0, attn_ffn_spikeglu_lif=True); m, l, _ = train_step(c); lk = causal(c)
print(f"X2 all-on + DSN-LIF: loss {l:.3f}; causal leak {lk}"); assert lk == 0.0
T4 = dict(CASES["tfmr"], engram_v2_hash="hi", attn_out_gate=True, attn_canon=4)
m, l, _ = train_step(T4); lk = causal(T4)
print(f"T4 tfmr + gate + canon: loss {l:.3f}; causal leak {lk}"); assert lk == 0.0

# KV-cache path still matches the full forward with the gate on (canon must refuse the cache path)
c = dict(BASE, attn_out_gate=True, attn_topk=4); torch.manual_seed(0); m = build_model(c); m.eval()
blk = attn_blocks(m)[0]; xx = torch.randn(2, 64, 64)
with torch.no_grad():
    full = blk.forward_sequence(xx, None, None, None)[0]
    cache = None; steps = []
    for t in range(64):
        o, cache = blk.step_kv(xx[:, t:t + 1], cache); steps.append(o)
    inc = torch.cat(steps, 1)
print(f"gate KV-cache vs full: max|diff| {(full - inc).abs().max().item():.2e}")
assert (full - inc).abs().max().item() < 1e-4
c = dict(BASE, attn_canon=4); torch.manual_seed(0); m = build_model(c)
try:
    attn_blocks(m)[0].step_kv(xx[:, :1], None); raise SystemExit("canon KV path should refuse")
except NotImplementedError:
    print("canon KV path refuses (NotImplementedError) as intended")
print("[OK] all r16 flag tests passed")
