"""site_sens_eval.py -- 2026-10-07 (Neuro 53, energy). Zero-shot sensitivity of each spike-code site: calibrate every
site at 1.0x, then raise ONE site's thresholds by FACTOR and measure the val ppl change and that site's nonzero-input
fraction (weight reads). Cheap guide for per-site thresholds (SITE_SCALE in temp/mk_spike_seed.py).
  python temp/site_sens_eval.py SRC.pt CONV_CONFIG.yaml [FACTOR]"""
import sys, math, torch, yaml, numpy as np
import torch.nn.functional as F
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model, StreamText, BIN
from spike_code import calibrate, SpikeCode
src, cfg = sys.argv[1:3]
fac = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
c = yaml.safe_load(open(cfg))
s = torch.load(src, map_location="cpu", weights_only=False)
m = build_model(c).cuda(); m.load_state_dict(s["model"], strict=False); del s
data = StreamText(c.get("data_bin", BIN), vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256),
                  val_tokens=c.get("data_val_tokens"))
T = int(c["block_size"])
np.random.seed(123)
with torch.autocast("cuda", dtype=torch.bfloat16):
    calibrate(m, [data.batch(4, T, "train", "cuda")[0] for _ in range(2)], scale=1.0)
m.eval()
SITE = {"sc_head": "head", "sc_in": "mem", "sc_head_mid": "head_mid"}
sites = {}
for nm, mm in m.named_modules():
    if isinstance(mm, SpikeCode):
        sites.setdefault(SITE.get(nm.split(".")[-1], nm.split(".")[-1]), []).append(mm)
nz = {}


def hook(mod, inp, out):
    nz.setdefault(id(mod), []).append(float((out != 0).float().mean()))


hooks = [mm.register_forward_hook(hook) for v in sites.values() for mm in v]


def ppl():
    np.random.seed(0); tot = 0.0; nz.clear()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(8):
            x, y = data.batch(4, T, "val", "cuda")
            tot += F.cross_entropy(m(x).float().reshape(-1, c["vocab"]), y.reshape(-1)).item()
    return math.exp(tot / 8)


def site_nz(name):
    return 100 * np.mean([np.mean(nz[id(mm)]) for mm in sites[name] if id(mm) in nz])


base = ppl()
print(f"base (all sites 1.0x): coded ppl {base:.2f}", flush=True)
base_nz = {k: site_nz(k) for k in sites}
for k, mods in sites.items():
    for mm in mods:
        mm.theta.data.mul_(fac)
    p = ppl()
    print(f"  site {k:9s} x{fac:g}: ppl {p:7.2f} ({100 * (p / base - 1):+5.1f}%) | nonzero inputs {base_nz[k]:5.1f}% -> "
          f"{site_nz(k):5.1f}%", flush=True)
    for mm in mods:
        mm.theta.data.div_(fac)
for h in hooks:
    h.remove()
