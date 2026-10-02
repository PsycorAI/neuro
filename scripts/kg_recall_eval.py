"""kg_recall_eval.py -- 2026-10-02 (docs/v4_brain_principles_review_2026_10_02.md 3.2). Knowledge-graph-style
in-context recall: can the model answer forward, INVERSE and 2-HOP (inheritance) queries, and which pathway does it?
  fwd   [k1 v1 .. kN vN] (+gap) [kq]                      -> vq   (MQAR, as recall_eval / haystack)
  rev   [k1 v1 .. kN vN] (+gap) [vq]                      -> kq   (inverse relation: "dog has instance ?")
  hop2  [a1 b1 .. aN bN] [b1 c1 .. bN cN] (+gap) [aq]     -> cq   (inheritance: a->b, b->c => a->c)
Each under three modes: full model, attn_off (attention mixing zeroed: memory-only), mem_off (memory read ablated).
Score = among-candidates accuracy (argmax over the N in-context answers); chance = 100/N.
  python scripts/kg_recall_eval.py [--device cpu|cuda] [--N 8] [--gap 0] name=ckpt [name=ckpt ...]"""
import argparse, sys
import numpy as np, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
sys.path.insert(0, "/home/glenn/projects/neuro/scripts")


def make(task, data, N, g, B, gen):
    from train_gpu import MQAR_KEY_LO, MQAR_KEY_HI, MQAR_VAL_LO, MQAR_VAL_HI
    kr, vr = MQAR_KEY_HI - MQAR_KEY_LO, MQAR_VAL_HI - MQAR_VAL_LO
    perm_k = torch.stack([torch.randperm(kr, generator=gen)[:2 * N] for _ in range(B)]) + MQAR_KEY_LO
    a, c = perm_k[:, :N], perm_k[:, N:]                          # disjoint key-range tokens
    b = torch.stack([torch.randperm(vr, generator=gen)[:N] for _ in range(B)]) + MQAR_VAL_LO
    qi = torch.randint(0, N, (B,), generator=gen)
    ar = torch.arange(B)
    parts = [torch.stack([a, b], 2).reshape(B, 2 * N)]
    if task == "hop2":
        order = torch.stack([torch.randperm(N, generator=gen) for _ in range(B)])
        parts.append(torch.stack([b.gather(1, order), c.gather(1, order)], 2).reshape(B, 2 * N))
    if g > 0:
        gap, _ = data.batch(B, g, "val", "cpu")
        used = torch.cat([a, b, c], 1)
        clash = (gap.unsqueeze(-1) == used.unsqueeze(1)).any(-1)
        parts.append(torch.where(clash, torch.randint(6000, 8000, gap.shape, generator=gen), gap))
    if task == "fwd":
        parts.append(a[ar, qi].unsqueeze(1)); cand = b
    elif task == "rev":
        parts.append(b[ar, qi].unsqueeze(1)); cand = a
    else:
        parts.append(a[ar, qi].unsqueeze(1)); cand = c
    return torch.cat(parts, 1), cand, qi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--gap", type=int, default=0)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("models", nargs="+")
    a = ap.parse_args()
    if a.device == "cpu":
        import lif_exact as LE
        from interp_scorecard import lif_ml_ref
        LE.lif_exact_ml = lif_ml_ref
        LE.lif_exact = lambda cur, beta, vth=1.0, K=None, alpha=2.0, gamma=1.0: lif_ml_ref(cur, beta, vth, gamma=gamma)
        torch.set_num_threads(8)
    from interp_scorecard import load
    from train_gpu import StreamText, BIN
    for spec in a.models:
        name, path = spec.split("=", 1)
        m, c = load(path, a.device)
        data = StreamText(c.get("data_bin", BIN), vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256),
                          val_tokens=c.get("data_val_tokens"))
        attn = [b for b in m.blocks if hasattr(b, "qkv")]
        cells = []
        for task in ("fwd", "rev", "hop2"):
            for mode in ("full", "attn_off", "mem_off"):
                if mode == "attn_off":
                    if not attn or len(attn) == len(m.blocks):
                        cells.append(f"{task}/{mode} n/a"); continue
                    for b in attn: b._attn = lambda x: torch.zeros_like(x)
                gen = torch.Generator().manual_seed(4242 + a.N); np.random.seed(4242 + a.N)
                hit = tot = 0
                for _ in range(a.reps):
                    x, cand, qi = make(task, data, a.N, a.gap, a.B, gen)
                    with torch.no_grad(), torch.autocast(a.device, dtype=torch.bfloat16, enabled=a.device == "cuda"):
                        lg = m(x.to(a.device), ablate_memory=(mode == "mem_off"))
                        lg = (lg[0] if isinstance(lg, tuple) else lg)[:, -1, :].float().cpu()
                    hit += (lg.gather(1, cand).argmax(1) == qi).sum().item(); tot += a.B
                for b in attn: b.__dict__.pop("_attn", None)
                cells.append(f"{task}/{mode} {100.0 * hit / tot:5.1f}")
        print(f"{name:12s} N{a.N} gap{a.gap} | " + " | ".join(cells), flush=True)
        del m
    print(f"chance = {100 / a.N:.1f}%")


if __name__ == "__main__":
    main()
