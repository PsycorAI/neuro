"""n58_revive.py <ckpt_in> <ckpt_out> [--tokens 32768] [--dry] -- 2026-10-12 (Neuro 58, Glenn: "use all layers").
Function-preserving revival of dead units in the attention FFNs (Linear -> relu2 -> Linear) of a v4 checkpoint.
A unit is DEAD if it never fires on --tokens of val text (a relu2 unit with no positive input has zero gradient, and
Muon's decoupled decay then shrinks its weights to 0: block 4 of v4_cloud has both matrices at exactly 0).
For each dead unit j of a block:
  input row   W_in[j, :]  <- N(0, s^2), s = median row RMS of that block's live units (no live unit: of all live units)
  output col  W_out[:, j] <- 0          -> the block's output is unchanged at this moment; gradient reaches W_out[:, j]
  Muon state  momentum row / column <- 0; NorMuon row statistic of W_in[j] <- median of the live rows
Prints per block: units, dead before, firing share of the revived units after (expected ~0.5), max output change (0).
GPU needed for the forward. Usage: CUDA_VISIBLE_DEVICES=1 python temp/n58_revive.py in.pt out.pt"""
import sys, os, argparse, numpy as np, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model, StreamText, BIN, _build_optimizers
from hybrid_attn import SparseAct
ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--tokens", type=int, default=32768)
ap.add_argument("--dry", action="store_true")
# 15:50: reviving EVERY dead unit (77-100% of the units of all 12 FFNs, ~135k units) cost +45% perplexity after 300
# steps (arm C, chunk1 72.3 vs 50.0). Narrow version: only the listed blocks, only this share of their dead units.
ap.add_argument("--blocks", default="")          # e.g. "2,3,4,5"; empty = all
ap.add_argument("--frac", type=float, default=1.0)
a = ap.parse_args()
s = torch.load(a.src, map_location="cpu", weights_only=False); c = dict(s["cfg"])
m = build_model(c).cuda(); m.load_state_dict(s["model"], strict=False); m.eval()
T = int(c.get("block_size", 1024)); B = 2; n_batches = max(1, a.tokens // (B * T))
data = StreamText(c.get("data_bin", BIN), vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256),
                  val_tokens=c.get("data_val_tokens"), prelut=c.get("data_prelut", False))
ffns = [(i, b.ffn) for i, b in enumerate(m.blocks) if hasattr(b, "qkv") and isinstance(b.ffn[1], SparseAct)]


def fire_counts(seed):
    cnt = {i: torch.zeros(f[0].out_features, device="cuda") for i, f in ffns}; hooks = []
    for i, f in ffns:
        hooks.append(f[1].register_forward_hook(
            # Neuro 59, 10-12 22:00 BUG FIX: add_() returned the count tensor and a forward hook's return value REPLACES
            # the module output -> every block after the first saw garbage, so all earlier dead counts of blocks >= 1
            # (and the units chosen for revival there) were wrong. The hook now returns None.
            lambda mod, inp, out, _i=i: (cnt[_i].add_((inp[0].detach() > 0).reshape(-1, inp[0].shape[-1]).float().sum(0)), None)[1]))
    np.random.seed(seed); outs = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(n_batches):
            x, _y = data.batch(B, T, "val", "cuda"); outs.append(m(x)[:, -64:].float().cpu())   # 13:55: the full
            # logits (8 GB) beside the 10 GB checkpoint ran the WSL VM out of RAM and killed every home process
    for h in hooks: h.remove()
    return cnt, torch.cat(outs)


cnt0, logits0 = fire_counts(0); N = n_batches * B * T
dead = {i: (cnt0[i] == 0) for i, _ in ffns}
live_rms = torch.cat([f[0].weight.detach().float()[~dead[i]].pow(2).mean(1).sqrt() for i, f in ffns])
g_med = live_rms.median().item()
opts = _build_optimizers(m, c, "cuda"); muon = opts[0]
midx = {id(p): k for k, p in enumerate(p for g in muon.param_groups for p in g["params"])}
mst = s["opt_states"][0]["state"]; gen = torch.Generator(device="cpu").manual_seed(0)
for i, f in ffns:
    d = dead[i]; nd = int(d.sum()); W1, W2 = f[0].weight, f[2].weight
    lr_ = W1.detach().float()[~d].pow(2).mean(1).sqrt()
    sd_ = lr_.median().item() if lr_.numel() >= 64 else g_med
    if sd_ < 0.2 * g_med: sd_ = g_med                # block 1: its "live" rows are numerically ~0
    # 19:50: of the dead units, how many have ERASED weights (input row and output column both under 1% of the live
    # median)? Those get no signal from any gradient method (ghost gradients included); only a re-initialisation helps.
    rin = W1.detach().float().pow(2).mean(1).sqrt(); cout = W2.detach().float().pow(2).mean(0).sqrt()
    cmed = cout[~d].median().item() if nd < d.numel() else 0.0
    er = ((rin < 0.01 * g_med) & (cout < 0.01 * max(cmed, 1e-9)) & d).float().sum().item()
    print(f"blk{i:2d}: {d.numel()} units | dead {nd} ({nd / d.numel():.1%}) | of those erased {er / max(nd, 1):.1%} | live fire share "
          f"{(cnt0[i][~d] / N).mean().item() if nd < d.numel() else 0:.3f} | new row RMS {sd_:.4f}", flush=True)
    if nd == 0 or a.dry: continue
    if a.blocks and str(i) not in a.blocks.split(","):
        dead[i] = torch.zeros_like(d); continue
    if a.frac < 1.0:
        idx = d.nonzero().squeeze(1).cpu(); keep = idx[torch.randperm(len(idx), generator=gen)[:int(len(idx) * a.frac)]]
        d = torch.zeros_like(d); d[keep.to(d.device)] = True; dead[i] = d; nd = int(d.sum())
        print(f"       reviving {nd} of them", flush=True)
    with torch.no_grad():
        W1[d] = (torch.randn(nd, W1.shape[1], generator=gen) * sd_).to(W1)
        W2[:, d] = 0
    dc = d.cpu()
    for W, row in ((W1, True), (W2, False)):
        st = mst.get(midx[id(W)])
        if st is None: continue
        if "momentum_buffer" in st:
            if row: st["momentum_buffer"][dc] = 0
            else: st["momentum_buffer"][:, dc] = 0
        if row and "normuon_v" in st and (~dc).any():
            st["normuon_v"][dc] = st["normuon_v"][~dc].median()
if a.dry: sys.exit(0)
cnt1, logits1 = fire_counts(0)
for i, _ in ffns:
    d = dead[i]
    if d.any(): print(f"blk{i:2d}: revived units now fire on {(cnt1[i][d] / N).mean().item():.3f} of tokens")
print(f"max |logit change| {(logits1 - logits0).abs().max().item():.5f} (0 = function preserved)")
s["model"] = {k: v.detach().cpu() for k, v in m.state_dict().items() if k in s["model"]}
os.makedirs(os.path.dirname(a.dst), exist_ok=True)
torch.save(s, a.dst + ".tmp"); os.replace(a.dst + ".tmp", a.dst); print("wrote", a.dst)
