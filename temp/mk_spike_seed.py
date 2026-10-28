"""mk_spike_seed.py -- 2026-10-06 (energy plan, way 3 = conversion). Seed a spike-coded run from a trained ckpt: build the
model from the NEW config (spike_code on), load the trained weights (SpikeCode thetas are new), calibrate every theta
to CAL_SCALE * mean|x| per channel on 8 train windows, report the zero-shot ppl change, and save step/tokens with no
optimizer state (the trainer then continues the source schedule like a resume).
  python temp/mk_spike_seed.py SRC.pt DST_DIR CONFIG.yaml [CAL_SCALE]"""
import sys, os, math, torch, yaml, numpy as np
import torch.nn.functional as F
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model, StreamText, BIN
from spike_code import calibrate, SpikeCode
src, dst_dir, cfg = sys.argv[1:4]
scale = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
c = yaml.safe_load(open(cfg))
# 2026-10-14 (Neuro 61; Glenn: average + revive + U-NorMuon "all three at once"). Optional seed prep, from the CONVERSION
# config (both off by default):  seed_avg_last: N  -> uniform average of the model weights of the last N step_*.pt next
# to SRC (SRC must be the newest; LAWA, -8.6% ppl at home);  seed_revive: true -> temp/n61mm_n60_revive.py on it (every
# FFN unit whose pre-activation never exceeds 1e-3 gets a fresh input row, output column 0 = function preserved).
import glob, subprocess
prep = []
nav = int(c.get("seed_avg_last", 0) or 0)
if nav > 1:
    os.makedirs(dst_dir, exist_ok=True)
    fs = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(src)), "step_*.pt")))
    assert fs and fs[-1] == os.path.abspath(src), f"seed_avg_last: {src} is not the newest checkpoint ({fs[-1:]})"
    fs = fs[-nav:]
    base = torch.load(src, map_location="cpu", weights_only=False, mmap=True)
    acc = {k: v.to("cuda", torch.float32) for k, v in base["model"].items()}
    for f in fs[:-1]:
        o = torch.load(f, map_location="cpu", weights_only=False, mmap=True)["model"]
        for k in acc: acc[k].add_(o[k].to("cuda", torch.float32))
        del o
    base["model"] = {k: (v / len(fs)).to(base["model"][k].dtype).cpu() for k, v in acc.items()}; del acc
    p = os.path.join(dst_dir, "_prep_avg.pt"); torch.save(base, p); del base; torch.cuda.empty_cache()
    prep.append(p); src = p
    print(f"[seed prep] averaged {len(fs)} checkpoints: {[os.path.basename(f) for f in fs]}", flush=True)
if c.get("seed_revive", False):
    os.makedirs(dst_dir, exist_ok=True)
    p = os.path.join(dst_dir, "_prep_revived.pt")
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "n61mm_n60_revive.py"), src, p],
                   check=True)
    prep.append(p); src = p
    print("[seed prep] revived dead FFN units", flush=True)
s = torch.load(src, map_location="cpu", weights_only=False, mmap=True)
m = build_model(c).cuda()
sd = dict(s["model"])
if c.get("head_rank") and sd["head.weight"].shape[1] != int(c["head_rank"]):
    # factorised head from the trained head's SVD: W (V x d) ~ (U sqrt S)(sqrt S Vh), rank r
    r = int(c["head_rank"])
    U, S, Vh = torch.linalg.svd(sd.pop("head.weight").float().cuda(), full_matrices=False)
    sd["head_in.weight"] = (S[:r].sqrt()[:, None] * Vh[:r]).cpu()
    sd["head.weight"] = (U[:, :r] * S[:r].sqrt()).cpu()
miss = m.load_state_dict(sd, strict=False)
assert all(k.split(".")[-1] == "theta" for k in miss.missing_keys), miss.missing_keys[:5]
data = StreamText(c.get("data_bin", BIN), vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256),
                  val_tokens=c.get("data_val_tokens"), prelut=c.get("data_prelut", False))
np.random.seed(123)
T = int(c["block_size"])
with torch.autocast("cuda", dtype=torch.bfloat16):
    n = calibrate(m, [data.batch(4, T, "train", "cuda")[0] for _ in range(2)], scale=scale)
# SITE_SCALE env (2026-10-07, Neuro 53; unset = unchanged): extra per-site threshold multipliers after the global
# calibration, e.g. SITE_SCALE="head=2,up=2" -> raise thresholds only where the energy is (head / MLP input).
SITE = {"sc_head": "head", "sc_in": "mem", "sc_head_mid": "head_mid"}
_ss = dict(kv.split("=") for kv in os.environ.get("SITE_SCALE", "").split(",") if kv)
for nm, mm in m.named_modules():
    if _ss and isinstance(mm, SpikeCode) and SITE.get(nm.split(".")[-1], nm.split(".")[-1]) in _ss:
        mm.theta.data.mul_(float(_ss[SITE.get(nm.split(".")[-1], nm.split(".")[-1])]))


def ppl(model):
    np.random.seed(0); tot = 0.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(8):
            x, y = data.batch(4, T, "val", "cuda")
            tot += F.cross_entropy(model(x).float().reshape(-1, c["vocab"]), y.reshape(-1)).item()
    return math.exp(tot / 8)


m.eval()
p_code = ppl(m)
cnt = np.mean([float(mm.last_count) for mm in m.modules() if isinstance(mm, SpikeCode)])
for mm in m.modules():
    if isinstance(mm, SpikeCode):
        mm.calib = []                         # bypass coding -> the dense model
p_dense = ppl(m)
for mm in m.modules():
    if isinstance(mm, SpikeCode):
        mm.calib = None
os.makedirs(dst_dir, exist_ok=True)
out = {k: v for k, v in s.items() if k not in ("opt_states", "opt", "model", "cfg")}
out["model"] = {k: v.cpu() for k, v in m.state_dict().items()}
out["cfg"] = c
torch.save(out, os.path.join(dst_dir, f"step_{s['step']:07d}.pt"))
for p in prep:
    os.remove(p)
print(f"[OK] {n} spike codes calibrated (scale {scale}) | zero-shot ppl dense {p_dense:.2f} -> coded {p_code:.2f} | "
      f"mean spikes/input {cnt:.3f} | saved {dst_dir} @ step {s['step']}")
