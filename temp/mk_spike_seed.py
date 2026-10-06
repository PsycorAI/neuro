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
s = torch.load(src, map_location="cpu", weights_only=False)
m = build_model(c).cuda()
miss = m.load_state_dict(s["model"], strict=False)
assert all(k.split(".")[-1] == "theta" for k in miss.missing_keys), miss.missing_keys[:5]
data = StreamText(c.get("data_bin", BIN), vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256),
                  val_tokens=c.get("data_val_tokens"))
np.random.seed(123)
T = int(c["block_size"])
with torch.autocast("cuda", dtype=torch.bfloat16):
    n = calibrate(m, [data.batch(4, T, "train", "cuda")[0] for _ in range(2)], scale=scale)


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
print(f"[OK] {n} spike codes calibrated (scale {scale}) | zero-shot ppl dense {p_dense:.2f} -> coded {p_code:.2f} | "
      f"mean spikes/input {cnt:.3f} | saved {dst_dir} @ step {s['step']}")
