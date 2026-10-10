"""xchunk_haystack_eval.py -- 2026-10-04 (r25). Haystack recall BEYOND the attention window: [k1 v1 .. kN vN] at the
start of attention chunk A, real val text after it, and the query key at offset o of chunk B -> predict its value.
Attention cannot cross the chunk boundary, so only the memory can answer (same batch builder and key/value ranges as
temp/haystack_recall_eval.py; gap = C - 2N + o). Among-candidates accuracy; chance = 100/N %. Also reports the same
model's memory-ablated accuracy at o=0 (should fall to chance if the memory is the route).
--chunk overrides the model's attn_chunk (needed for models trained without chunks).
  python temp/xchunk_haystack_eval.py [--chunk 512] name=ckpt.pt [...]"""
import sys, argparse, torch, numpy as np
sys.path.insert(0, "/home/glenn/projects/neuro/temp")
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model, StreamText, BIN, MQAR_KEY_LO, MQAR_KEY_HI, MQAR_VAL_LO, MQAR_VAL_HI
torch.set_float32_matmul_precision("high")
ap = argparse.ArgumentParser()
ap.add_argument("--chunk", type=int, default=0)
ap.add_argument("runs", nargs="+")
a = ap.parse_args()
NS, OFFS, B, REPS = (8, 32), (0, 64, 192, 448), 32, 4


def batch(N, g, data, gen):
    keys = torch.stack([torch.randperm(MQAR_KEY_HI - MQAR_KEY_LO, generator=gen)[:N] for _ in range(B)]) + MQAR_KEY_LO
    vals = torch.stack([torch.randperm(MQAR_VAL_HI - MQAR_VAL_LO, generator=gen)[:N] for _ in range(B)]) + MQAR_VAL_LO
    kv = torch.stack([keys, vals], 2).reshape(B, 2 * N)
    gap, _ = data.batch(B, g, "val", "cpu")
    clash = (gap.unsqueeze(-1) == keys.unsqueeze(1)).any(-1)
    gap = torch.where(clash, torch.randint(6000, 8000, gap.shape, generator=gen), gap)
    qi = torch.randint(0, N, (B,), generator=gen)
    return torch.cat([kv, gap, keys[torch.arange(B), qi].unsqueeze(1)], 1), vals, qi


for arg in a.runs:
    name, path = arg.split("=", 1)
    s = torch.load(path, map_location="cpu", weights_only=False); c = dict(s["cfg"])
    m = build_model(c).to("cuda"); m.load_state_dict(s["model"], strict=False); m.eval(); del s
    C = a.chunk or int(c.get("attn_chunk", 0) or 0)
    assert C, f"{name}: no attn_chunk in cfg; pass --chunk"
    for b in m.blocks:
        if hasattr(b, "qkv"):
            b.attn_chunk = C
    data = StreamText(c.get("data_bin", BIN), vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256),
                      val_tokens=c.get("data_val_tokens"), prelut=c.get("data_prelut", False))
    res = {}
    for N in NS:
        for o in OFFS + ("abl",):
            oo = 0 if o == "abl" else o
            gen = torch.Generator().manual_seed(7000 + 100 * N + oo); np.random.seed(7000 + 100 * N + oo)
            hit = tot = 0
            for _ in range(REPS):
                x, vals, qi = batch(N, C - 2 * N + oo, data, gen)
                assert x.shape[1] == C + oo + 1
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = m(x.cuda(), ablate_memory=(o == "abl"))[:, -1, :].float().cpu()
                hit += (lg.gather(1, vals).argmax(1) == qi).sum().item(); tot += B
            res[(N, o)] = 100.0 * hit / tot
    print(f"{name:26s} C={C} " + " | ".join(f"N{N}: " + " ".join(f"{res[(N, o)]:5.1f}" for o in OFFS)
                                         + f" (mem off {res[(N, 'abl')]:5.1f})" for N in NS), flush=True)
    del m; torch.cuda.empty_cache()
print("columns: query offset into chunk B =", OFFS, "(kv block at the start of chunk A); chance N8 12.5%, N32 3.1%")
