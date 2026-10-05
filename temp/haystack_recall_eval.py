"""haystack_recall_eval.py -- 2026-09-28: in-context recall UNDER INTERFERENCE (the "interference wall", arXiv 2609.16183).
Sequence = [k1 v1 ... kN vN] + g tokens of real val text (distractors) + [kq] -> predict vq at the last position.
Same key/value token ranges as MQAR aux + recall_eval. Gap tokens that collide with this sequence's keys are replaced
by random tokens from 6000-7999 (outside both ranges). Reports among-candidates accuracy per (N, gap).
  python temp/haystack_recall_eval.py name=path/to/ckpt.pt [name=path ...]"""
import sys, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model, StreamText, BIN, MQAR_KEY_LO, MQAR_KEY_HI, MQAR_VAL_LO, MQAR_VAL_HI
import numpy as np
NS, GAPS, B, REPS = (8, 32), (0, 64, 192, 440), 64, 4
torch.set_float32_matmul_precision("high")

def batch(N, g, data, gen):
    keys = torch.stack([torch.randperm(MQAR_KEY_HI - MQAR_KEY_LO, generator=gen)[:N] for _ in range(B)]) + MQAR_KEY_LO
    vals = torch.stack([torch.randperm(MQAR_VAL_HI - MQAR_VAL_LO, generator=gen)[:N] for _ in range(B)]) + MQAR_VAL_LO
    kv = torch.stack([keys, vals], 2).reshape(B, 2 * N)
    parts = [kv]
    if g > 0:
        gap, _ = data.batch(B, g, "val", "cpu")
        clash = (gap.unsqueeze(-1) == keys.unsqueeze(1)).any(-1)
        gap = torch.where(clash, torch.randint(6000, 8000, gap.shape, generator=gen), gap)
        parts.append(gap)
    qi = torch.randint(0, N, (B,), generator=gen)
    parts.append(keys[torch.arange(B), qi].unsqueeze(1))
    return torch.cat(parts, 1), vals, qi

rows = []
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    s = torch.load(path, map_location="cpu", weights_only=False); c = dict(s["cfg"])
    m = build_model(c).to("cuda"); m.load_state_dict(s["model"], strict=False); m.eval(); del s
    data = StreamText(c.get("data_bin", BIN), vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256))
    res = {}
    for N in NS:
        for g in GAPS:
            gen = torch.Generator().manual_seed(1000 * N + g); np.random.seed(1000 * N + g)
            hit = tot = 0
            for _ in range(REPS):
                x, vals, qi = batch(N, g, data, gen)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = m(x.cuda())[:, -1, :].float().cpu()
                cand = lg.gather(1, vals)                        # (B, N) logits of the N in-context values
                hit += (cand.argmax(1) == qi).sum().item(); tot += B
            res[(N, g)] = 100.0 * hit / tot
    rows.append((name, res)); del m; torch.cuda.empty_cache()
    print(f"{name:14s} " + " | ".join(f"N{N}: " + " ".join(f"{res[(N, g)]:5.1f}" for g in GAPS) for N in NS), flush=True)
print("columns: gap =", GAPS, "tokens of real text between the key-value block and the query; chance = 100/N %")
