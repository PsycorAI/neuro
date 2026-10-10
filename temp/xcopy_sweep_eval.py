"""xcopy_sweep_eval.py -- 2026-10-06 (Neuro 52, diagnostic E1). Why is real-text recall beyond the window at chance?
A span of S tokens sits in chunk A, ending `dist` tokens before the chunk boundary; it is repeated at offset 16 of chunk
B (attention cannot cross, so only the memory can copy it). Top-1 accuracy on the span tokens after a 4-token cue, with
the memory on vs ablated. Sweeps: dist (number of later writes in chunk A), span length, and span content (a real-text
span copied from chunk A vs random tokens written into both places).
Reading: gain only at small dist = interference (later writes overwrite it); random spans gain but real ones don't =
selectivity (the memory writes only "unusual" tokens); no gain anywhere = read path / key matching.
  python temp/xcopy_sweep_eval.py [--chunk 512] name=ckpt.pt [...]"""
import sys, argparse, torch, numpy as np
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model, StreamText, BIN
torch.set_float32_matmul_precision("high")
ap = argparse.ArgumentParser(); ap.add_argument("--chunk", type=int, default=0)
ap.add_argument("--reps", type=int, default=8)
ap.add_argument("--contents", default="real,random")  # "none" = same positions, no repeat (control)
ap.add_argument("runs", nargs="+")
a = ap.parse_args()
CUE, B, DST, DISTS, SPANS = 4, 16, 16, (0, 64, 256, 448), (12, 32)

for arg in a.runs:
    name, path = arg.split("=", 1)
    s = torch.load(path, map_location="cpu", weights_only=False); c = dict(s["cfg"])
    m = build_model(c).cuda(); m.load_state_dict(s["model"], strict=False); m.eval(); del s
    C = a.chunk or int(c.get("attn_chunk", 0) or 0) or 512
    for b in m.blocks:
        if hasattr(b, "qkv"):
            b.attn_chunk = C
    data = StreamText(c.get("data_bin", BIN), vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256),
                      val_tokens=c.get("data_val_tokens"), prelut=c.get("data_prelut", False))
    for content in a.contents.split(","):
        for S in SPANS:
            row = []
            for dist in DISTS:
                acc = {}
                for abl in (False, True):
                    np.random.seed(11); gen = torch.Generator().manual_seed(5)
                    hit = tot = 0
                    for _ in range(a.reps):
                        x, _ = data.batch(B, 2 * C, "val", "cpu")
                        src, dst = C - dist - S, C + DST
                        if content == "random":
                            x[:, src:src + S] = torch.randint(1000, c["vocab"] - 1000, (B, S), generator=gen)
                        if content != "none":
                            x[:, dst:dst + S] = x[:, src:src + S].clone()
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            pred = m(x.cuda(), ablate_memory=abl).argmax(-1).cpu()
                        tgt = x[:, dst + CUE:dst + S]
                        got = pred[:, dst + CUE - 1:dst + S - 1]
                        hit += (got == tgt).sum().item(); tot += tgt.numel()
                    acc[abl] = 100.0 * hit / tot
                row.append(f"d{dist:<3d} {acc[False]:5.1f}/{acc[True]:5.1f}")
            print(f"{name:26s} C={C} E1 {content:6s} S{S:<2d} mem on/off | " + " | ".join(row), flush=True)
    del m; torch.cuda.empty_cache()
print("E1 columns: d = tokens of chunk A written after the span (span ends d tokens before the boundary); repeat at "
      f"chunk-B offset {DST}; top-1 % on span tokens after a {CUE}-token cue, memory on / ablated")
