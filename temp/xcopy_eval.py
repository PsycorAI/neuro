"""xcopy_eval.py -- 2026-10-06. Exact recall of earlier text: a 32-token real-text span is repeated later; after a 4-token
cue, top-1 accuracy on the remaining 28 span tokens. 'in' = repeat inside the same attention chunk (attention can copy),
'x' = span in chunk A, repeat in chunk B (only the memory can), 'x off' = same with memory ablated, 'none' = the same
positions without any repeat (natural predictability). --chunk overrides the model's attn_chunk.
  python temp/xcopy_eval.py [--chunk 512] name=ckpt.pt [...]"""
import sys, argparse, torch, numpy as np
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model, StreamText, BIN
torch.set_float32_matmul_precision("high")
ap = argparse.ArgumentParser(); ap.add_argument("--chunk", type=int, default=0); ap.add_argument("runs", nargs="+")
a = ap.parse_args()
S, CUE, B, REPS = 32, 4, 16, 6

for arg in a.runs:
    name, path = arg.split("=", 1)
    s = torch.load(path, map_location="cpu", weights_only=False); c = dict(s["cfg"])
    m = build_model(c).cuda(); m.load_state_dict(s["model"], strict=False); m.eval(); del s
    C = a.chunk or int(c.get("attn_chunk", 0) or 0) or 512
    for b in m.blocks:
        if hasattr(b, "qkv"):
            b.attn_chunk = C
    data = StreamText(c.get("data_bin", BIN), vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256),
                      val_tokens=c.get("data_val_tokens"))
    res = {}
    for mode in ("none", "in", "x", "x off"):
        np.random.seed(11); rng = np.random.RandomState(5)
        hit = tot = 0
        for _ in range(REPS):
            x, _ = data.batch(B, 2 * C, "val", "cpu")
            pos = []
            for i in range(B):
                if mode == "in":
                    src, dst = int(rng.randint(0, C // 2 - S)), int(rng.randint(C // 2, C - S))
                else:
                    src, dst = int(rng.randint(0, C - S)), int(rng.randint(C, 2 * C - S))
                if mode != "none":
                    x[i, dst:dst + S] = x[i, src:src + S].clone()
                pos.append(dst)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred = m(x.cuda(), ablate_memory=(mode == "x off")).argmax(-1).cpu()
            for i, d in enumerate(pos):
                tgt = x[i, d + CUE:d + S]                     # tokens 5..32 of the span
                got = pred[i, d + CUE - 1:d + S - 1]          # predicted from the previous position
                hit += (got == tgt).sum().item(); tot += tgt.numel()
        res[mode] = 100.0 * hit / tot
    print(f"{name:28s} C={C} xcopy top1 | none {res['none']:5.1f} | in-window {res['in']:5.1f} | "
          f"beyond (memory) {res['x']:5.1f} | beyond mem-off {res['x off']:5.1f}", flush=True)
    del m; torch.cuda.empty_cache()
