"""parity_r15_flags.py -- 2026-10-01. CPU parity + functional tests for the round-15 flags
(attn_ffn, attn_ffn_spikeglu_k, attn_topk, spk_layerscale). Usage:
  python temp/parity_r15_flags.py save    # BEFORE the code change: save reference logits (flags absent)
  python temp/parity_r15_flags.py check   # AFTER: flags off must be bit-identical; flags on must behave
The LIF Triton kernel is swapped for an exact torch reference so this runs on CPU (GPUs stay busy)."""
import sys, copy, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
import lif_exact as LE

def lif_ml_ref(cur, beta, vth=1.0, K=None, alpha=2.0, gamma=1.0, levels=1, signed=False):
    B, T, N = cur.shape
    beta = beta if torch.is_tensor(beta) else torch.full((N,), float(beta))
    mem = torch.zeros(B, N); outs = []
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

BASE = dict(arch="spiking", vocab=512, d=64, n_neurons=128, d_mem=64, n_layers=3, block_size=64,
            use_fpt=True, lif_exact=True, lif_signed=True, lif_levels=4, delta_rule=True, decay_gate=True,
            beta_floor=0.1, gdn2_enabled=True, gdn2_chunked=True, gdn2_chunk_size=16, gate_rank=4,
            engram_v2_enabled=True, engram_v2_K=256, engram_v2_heads=2, engram_v2_rank=8,
            attention_layer_positions=[1, 2], attention_heads=4, attn_single_residual=True,
            final_norm=True, spike_rate_undiluted=True, pre_norm=True, mem_out_gate=4, canon_a=4)
ALLSPK = dict(BASE, attention_layer_positions=[], n_layers=2)
REF = "/home/glenn/projects/neuro/temp/parity_r15_ref.pt"

def run(c, x, seed=0):
    torch.manual_seed(seed); m = build_model(c); m.eval()
    with torch.no_grad():
        out = m(x)
    return (out[0] if isinstance(out, tuple) else out), m

torch.manual_seed(123); X = torch.randint(0, 512, (2, 64))
mode = sys.argv[1]
if mode == "save":
    torch.save({"hyb": run(BASE, X)[0], "spk": run(ALLSPK, X)[0]}, REF); print("[OK] saved reference")
    sys.exit()

ref = torch.load(REF)
for k, c in (("hyb", BASE), ("spk", ALLSPK)):
    d = (run(c, X)[0] - ref[k]).abs().max().item()
    print(f"parity flags-off {k}: max|diff| = {d}"); assert d == 0.0, k
# explicit defaults also bit-identical
c0 = dict(BASE, attn_ffn=True, attn_ffn_spikeglu_k=0, attn_topk=0, spk_layerscale=0.0)
assert (run(c0, X)[0] - ref["hyb"]).abs().max().item() == 0.0; print("parity explicit-defaults: 0.0")

def train_step(c):
    torch.manual_seed(0); m = build_model(c); m.train()
    out = m(X, return_stats=True); logits = out[0]
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 512), X[:, 1:].reshape(-1))
    loss.backward(); g = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
    assert torch.isfinite(loss) and g > 0
    return m, loss.item()

def causal(c):
    torch.manual_seed(0); m = build_model(c); m.eval()
    x2 = X.clone(); x2[:, 40] = (x2[:, 40] + 7) % 512
    with torch.no_grad():
        a = m(X); b = m(x2)
    a = a[0] if isinstance(a, tuple) else a; b = b[0] if isinstance(b, tuple) else b
    return (a[:, :40] - b[:, :40]).abs().max().item()

# r15a: attention layers without their MLP
c = dict(BASE, attn_ffn=False); m, l = train_step(c)
assert all(getattr(b, "ffn", None) is None for b in m.blocks if hasattr(b, "qkv"))
n0 = sum(p.numel() for p in build_model(BASE).parameters()); n1 = sum(p.numel() for p in m.parameters())
print(f"attn_ffn=False: loss {l:.3f} params {n0} -> {n1}; causal leak {causal(c)}")

# r15b: spike-gated GLU with k-winner-take-all
c = dict(BASE, attn_ffn_spikeglu_k=8, attn_ffn_hidden=128); m, l = train_step(c)
blk = [b for b in m.blocks if hasattr(b, "qkv")][0]; seen = {}
blk.ffn.w2.register_forward_hook(lambda mod, inp, out: seen.__setitem__("h", inp[0].detach()))
with torch.no_grad():
    m.eval(); m(X)
nz = (seen["h"] != 0).sum(-1)
print(f"spikeglu k=8 H=128: loss {l:.3f}; active units/token min {nz.min().item()} max {nz.max().item()}; causal leak {causal(c)}")
assert nz.max().item() <= 8

# r15c: top-k attention
c = dict(BASE, attn_topk=4); m, l = train_step(c)
blk = [b for b in m.blocks if hasattr(b, "qkv")][0]
xx = torch.randn(2, 64, 64)
with torch.no_grad():
    w = blk._attn_weights(blk.norm_attn(xx))
nzw = (w > 0).sum(-1)
print(f"attn_topk=4: loss {l:.3f}; nonzero weights/query max {nzw.max().item()} (first rows {nzw[0, 0, :6].tolist()}); causal leak {causal(c)}")
assert nzw.max().item() <= 4
# KV-cache path must match the full forward under top-k
blk.eval()
with torch.no_grad():
    full = blk.forward_sequence(xx, None, None, None)[0]
    cache = None; steps = []
    for t in range(64):
        o, cache = blk.step_kv(xx[:, t:t + 1], cache); steps.append(o)
    inc = torch.cat(steps, 1)
print(f"attn_topk KV-cache vs full: max|diff| {(full - inc).abs().max().item():.2e}")
assert (full - inc).abs().max().item() < 1e-4

# r15d: LayerScale on spiking-block outputs
c = dict(ALLSPK, spk_layerscale=0.1); m, l = train_step(c)
g = [b.ls_gamma for b in m.blocks if hasattr(b, "ls_gamma")]
print(f"spk_layerscale=0.1: loss {l:.3f}; gammas {[round(x.mean().item(), 3) for x in g]}; grad {[x.grad.abs().sum().item() > 0 for x in g]}; causal leak {causal(c)}")
assert g and all(x.grad is not None for x in g)
print("[OK] all r15 flag tests passed")
