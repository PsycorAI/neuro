"""diag_gap_positions.py -- 2026-09-27: WHERE does the honest spiking-vs-transformer val_ppl gap live?
Read-only, eval-only. For each final 30M checkpoint: CE by context position bucket, CE by target-token
frequency bucket, memory-ablated CE by position, and spike statistics (dead/always-on neurons, active
count per token, how often the LIF membrane history changes the spike vs a stateless threshold).
Same val windows as train_gpu.evaluate (EVAL_SEED)."""
import sys, math, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model, StreamText, BIN, EVAL_SEED
import lif_exact as LE, lif_parallel as LPm
N = "/home/glenn/projects/neuro/models/checkpoints"
RUNS = [("lif_exact", f"{N}/screen_honest_lifexact_s0_30m/step_0006104.pt"),
        ("fpt", f"{N}/screen_honest_fpt_s0_30m/step_0006104.pt"),
        ("tfmr", f"{N}/screen_tfmrfix_adamw_s0_30m/step_0006104.pt")]
if len(sys.argv) > 1: RUNS = [(a.split("=")[0], a.split("=")[1]) for a in sys.argv[1:]] + [RUNS[-1]]
ITERS, B = 40, 8
PB = [(0, 1), (1, 4), (4, 16), (16, 64), (64, 256), (256, 512)]
torch.set_float32_matmul_precision("high")

# unigram counts from train split -> frequency buckets for targets
d0 = StreamText(BIN, vocab=16384, orig_vocab=128256)
np.random.seed(123)
cnt = torch.zeros(16384, dtype=torch.long)
for _ in range(150):
    x, y = d0.batch(32, 512, "train", "cpu")
    cnt += torch.bincount(y.reshape(-1), minlength=16384)
rank = torch.empty_like(cnt); rank[cnt.argsort(descending=True)] = torch.arange(16384)
FB = [(0, 100), (100, 1000), (1000, 4000), (4000, 16384)]
tot_cov = [float(cnt[(rank >= a) & (rank < b)].sum()) / float(cnt.sum()) for a, b in FB]

REC = {}
def wrap(fn):
    def f(cur, *a, **k):
        spk, mf = fn(cur, *a, **k)
        if REC.get("on"):
            REC.setdefault("spk", []).append(spk.detach().float()); REC.setdefault("cur", []).append(cur.detach().float())
        return spk, mf
    return f
LE.lif_exact = wrap(LE.lif_exact); LPm.lif_parallel = wrap(LPm.lif_parallel)

def run(name, path):
    s = torch.load(path, map_location="cpu", weights_only=False); c = dict(s["cfg"])
    m = build_model(c).to("cuda"); m.load_state_dict(s["model"], strict=False); m.eval(); del s
    data = StreamText(c.get("data_bin", BIN), vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256))
    spiking = any(hasattr(b, "W_k") for b in m.blocks)
    out = {}
    for abl in ([False, True] if spiking else [False]):
        np.random.seed(EVAL_SEED)
        pos_sum = torch.zeros(512, device="cuda"); fr_sum = torch.zeros(4, device="cuda"); fr_n = torch.zeros(4, device="cuda")
        for i in range(ITERS):
            x, y = data.batch(B, 512, "val", "cuda")
            REC.clear(); REC["on"] = (not abl and i < 4)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                lg = m(x, ablate_memory=abl)
            ce = F.cross_entropy(lg.float().reshape(-1, c["vocab"]), y.reshape(-1), reduction="none").view(B, 512)
            pos_sum += ce.sum(0)
            r = rank.to("cuda")[y]
            for j, (a, b_) in enumerate(FB):
                mk = (r >= a) & (r < b_); fr_sum[j] += ce[mk].sum(); fr_n[j] += mk.sum()
            if REC.get("on") and REC.get("spk"):
                out.setdefault("spk", []).append([t.cpu() for t in REC["spk"]]); out.setdefault("cur", []).append([t.cpu() for t in REC["cur"]])
        pm = pos_sum / (ITERS * B)
        out["abl" if abl else "full"] = dict(
            ppl=math.exp(pm.mean().item()),
            pos=[math.exp(pm[a:b_].mean().item()) for a, b_ in PB],
            freq=[math.exp((fr_sum[j] / fr_n[j]).item()) for j in range(4)])
    del m; torch.cuda.empty_cache()
    return out

res = {n: run(n, p) for n, p in RUNS}
print(f"val ppl (40x8x512 tok, EVAL_SEED windows)")
print("POSITION buckets  " + " ".join(f"{f'[{a},{b})':>10s}" for a, b in PB) + "      all")
for n in res:
    for k in ("full", "abl"):
        if k in res[n]:
            r = res[n][k]; print(f"{n+('' if k=='full' else ' -mem'):17s} " + " ".join(f"{v:10.1f}" for v in r["pos"]) + f" {r['ppl']:8.1f}")
t = res["tfmr"]["full"]
for n in [k for k in res if k != "tfmr"]:
    r = res[n]["full"]; print(f"ratio {n:11s} " + " ".join(f"{a/b:10.2f}" for a, b in zip(r["pos"], t["pos"])) + f" {r['ppl']/t['ppl']:8.2f}")
print("\nTARGET-FREQ buckets (rank)   " + " ".join(f"{f'{a}-{b}':>11s}" for a, b in FB))
print("token share                  " + " ".join(f"{v*100:10.1f}%" for v in tot_cov))
for n in res:
    r = res[n]["full"]; print(f"{n:28s} " + " ".join(f"{v:11.1f}" for v in r["freq"]))
for n in [k for k in res if k != "tfmr"]:
    r = res[n]["full"]; print(f"ratio {n:22s} " + " ".join(f"{a/b:11.2f}" for a, b in zip(r["freq"], t["freq"])))
print("\nSPIKE stats (4 batches x 8 x 512)")
for n in [k for k in res if k != "tfmr"]:
    S = res[n].get("spk"); C = res[n].get("cur")
    if not S: continue
    nblk = len(S[0])
    for l in range(nblk):
        spk = torch.cat([b[l] for b in S], 0); cur = torch.cat([b[l] for b in C], 0)   # (Bt,T,N)
        hard = (spk > 0.5).float()
        rate_n = hard.mean((0, 1))
        stateless = (cur > 1.0).float()
        agree = (stateless == hard).float().mean().item()
        flips_on = ((hard == 1) & (stateless == 0)).float().sum().item() / max(1.0, hard.sum().item())
        act = hard.sum(-1)
        print(f"{n:10s} blk{l}: rate {hard.mean():.3f} | dead(<1e-4) {(rate_n < 1e-4).float().mean()*100:5.1f}% | "
              f">50% {(rate_n > 0.5).float().mean()*100:5.1f}% | active/tok mean {act.mean():.0f} p10 {act.flatten().quantile(0.1):.0f} "
              f"p90 {act.flatten().quantile(0.9):.0f} | spk==cur>1 agree {agree*100:.1f}% | spikes needing history {flips_on*100:.1f}% "
              f"| cur std {cur.std():.2f} mean {cur.mean():.2f}")
