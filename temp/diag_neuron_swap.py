"""diag_neuron_swap.py -- 2026-09-27: swap the LIF of a TRAINED 30M model (screen_indepwin_s0_30m,
trained with K=10 FPT) at inference, no retraining. Per variant: val ppl (same windows), spike rate,
agreement with the FPT spikes on the same currents. Finds the exact/parallel neuron closest to FPT."""
import sys, math, yaml, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/glenn/projects/neuro/src")
import train_gpu
from train_gpu import build_model, StreamText, BIN
import lif_parallel as LP
from lif_exact import lif_exact
torch.set_float32_matmul_precision("high")
dev = "cuda"
c = yaml.safe_load(open("/home/glenn/projects/neuro/configs/screen_indepwin_s0_30m.yaml"))
m = build_model(c).to(dev).eval()
sd = torch.load("/home/glenn/projects/neuro/models/checkpoints/screen_indepwin_s0_30m/step_0006104.pt",
                map_location=dev, weights_only=False)["model"]
m.load_state_dict(sd, strict=False)
data = StreamText(BIN, vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256))
FPT = LP.lif_parallel
stats = {}
def make(name, fn):
    @torch.compiler.disable
    def wrapped(cur, beta, vth=1.0, K=3, alpha=2.0):
        s, mf = fn(cur, beta, vth, K, alpha)
        ref, _ = FPT(cur, beta, vth=vth, K=K, alpha=alpha)
        st = stats.setdefault(name, [0.0, 0.0, 0])
        st[0] += s.float().round().mean().item(); st[1] += (s.round() == ref.round()).float().mean().item(); st[2] += 1
        return s, mf
    return wrapped
variants = {"FPT K=10 (trained)": lambda cur, b, v, K, a: FPT(cur, b, vth=v, K=K, alpha=a)}
for g in (1.0, 0.75, 0.5, 0.25, 0.0):
    variants[f"exact gamma={g}"] = (lambda g: lambda cur, b, v, K, a: lif_exact(cur, b, vth=v, alpha=a, gamma=g))(g)
for vs in (0.5, 0.7):
    variants[f"exact gamma=1 vth={vs}"] = (lambda vs: lambda cur, b, v, K, a: lif_exact(cur, b, vth=vs, alpha=a, gamma=1.0))(vs)
for g, vs in ((0.0, 1.5), (0.0, 2.0)):
    variants[f"exact gamma={g} vth={vs}"] = (lambda g, vs: lambda cur, b, v, K, a: lif_exact(cur, b, vth=vs, alpha=a, gamma=g))(g, vs)
print(f"{'variant':26s} | val_ppl | spike rate | agree w/ FPT")
for name, fn in variants.items():
    LP.lif_parallel = make(name, fn)
    np.random.seed(1234); tot = 0.0; n = 20
    with torch.no_grad():
        for _ in range(n):
            x, y = data.batch(8, c["block_size"], "val", dev)
            with torch.autocast(dev, dtype=torch.bfloat16):
                out = m(x)
            tot += F.cross_entropy(out.reshape(-1, c["vocab"]).float(), y.reshape(-1)).item()
    st = stats[name]
    print(f"{name:26s} | {math.exp(tot/n):7.2f} | {st[0]/st[2]:10.3f} | {st[1]/st[2]:.3f}", flush=True)
