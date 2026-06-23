"""Associative-recall eval (MQAR-style) — the metric that actually probes memory.

Perplexity on web text barely rewards long-range memory (most next-tokens are
locally predictable). Recall does: store N (key,value) bindings in context, then
query a key and see if the model retrieves its value. The literature finding
(Zoology/MQAR, ICLR 2024) is that this recall gap, NOT ppl, explains most of the
quality difference between memory architectures.

This is a ZERO-SHOT probe of trained checkpoints: we present pairs the model has
never seen, in one forward pass (memory M builds within the sequence from zero
state), and read the prediction at the query position. A model whose training
built a better in-context memory scores higher — even though it was trained on
natural text, not on this task.

Protocol (single presentation, the hard/honest version):
  seq = [k1 v1 k2 v2 ... kN vN  kq]   (kq = one of the keys, random)
  target = the value paired with kq.
  metrics: exact-match accuracy (argmax == target) and mean rank of target
           (lower = stronger recall; sensitive even when accuracy is low).

Usage:
  python scripts/recall_eval.py --runs st_lc_9m_random,st_lc_9m_stateful,st_gated_9m
"""
import os, sys, glob, argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import SpikingHebbianLM

ROOT = os.path.join(os.path.dirname(__file__), "..")
KEY_LO, KEY_HI = 1000, 6000        # key token range
VAL_LO, VAL_HI = 8000, 14000       # value token range (disjoint from keys)


def latest(run):
    cks = sorted(glob.glob(os.path.join(ROOT, "models", "checkpoints", run, "step_*.pt")))
    if not cks:
        raise FileNotFoundError(f"no checkpoints in {run}")
    return cks[-1]


def build(c, device):
    m = SpikingHebbianLM(c["vocab"], d=c["d"], n_neurons=c["n_neurons"], d_mem=c["d_mem"],
                         recurrent=c.get("recurrent", False), rec_density=c.get("rec_density", 0.05),
                         compile_safe=c.get("compile", False), tie_weights=c.get("tie_weights", False),
                         n_layers=c.get("n_layers", 1), use_fpt=c.get("use_fpt", False),
                         fpt_K=c.get("fpt_K", 10), lam=c.get("lam", 0.98),
                         learnable_decay=c.get("learnable_decay", False),
                         write_gate=c.get("write_gate", False),
                         delta_rule=c.get("delta_rule", False),
                         beta_floor=c.get("beta_floor", 0.0),
                         decay_gate=c.get("decay_gate", False),
                         titans=c.get("titans", False),
                         local_attn=c.get("local_attn", False),
                         local_window=c.get("local_window", 64)).to(device)
    return m


def make_batch(B, N, vocab, device, seed):
    """B sequences of N (key,value) pairs + one query key.
    Returns (x, target, cand_vals) where cand_vals (B,N) are the N value tokens
    present in each sequence (the candidate set for among-candidates recall)."""
    g = torch.Generator().manual_seed(seed)
    xs, tgts, cands = [], [], []
    for _ in range(B):
        keys = (torch.randperm(KEY_HI - KEY_LO, generator=g)[:N] + KEY_LO)
        vals = (torch.randperm(VAL_HI - VAL_LO, generator=g)[:N] + VAL_LO)
        seq = torch.stack([keys, vals], dim=1).reshape(-1)        # k1 v1 k2 v2 ...
        qi = int(torch.randint(0, N, (1,), generator=g))
        seq = torch.cat([seq, keys[qi:qi+1]])                     # append query key
        xs.append(seq); tgts.append(int(vals[qi])); cands.append(vals)
    x = torch.stack(xs).to(device)
    return x, torch.tensor(tgts, device=device), torch.stack(cands).to(device)


@torch.no_grad()
def recall(model, c, N, device, B=64, seed=0):
    model.eval()
    x, target, cand = make_batch(B, N, c["vocab"], device, seed)
    logits = model(x)[:, -1, :]                                   # (B, vocab) at query pos
    full_acc = (logits.argmax(-1) == target).float().mean().item()
    # mean full-vocab rank of the true target (1 = best)
    rank = ((logits > logits.gather(1, target[:, None])).sum(1) + 1).float().mean().item()
    # MQAR among-candidates accuracy: of the N in-context values, is the correct
    # one the highest? (chance = 1/N). Much more sensitive than full-vocab argmax.
    cand_logits = logits.gather(1, cand)                          # (B, N)
    picked = cand.gather(1, cand_logits.argmax(1, keepdim=True)).squeeze(1)
    cand_acc = (picked == target).float().mean().item()
    return full_acc, rank, cand_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="st_lc_9m_random,st_lc_9m_stateful,st_gated_9m")
    ap.add_argument("--Ns", default="4,8,16,32,64")
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fpt_K", type=int, default=0,
                    help="override fpt_K for eval (0 = use each ckpt's cfg)")
    args = ap.parse_args()
    device = args.device
    Ns = [int(n) for n in args.Ns.split(",")]
    runs = args.runs.split(",")

    models, cfgs = {}, {}
    for run in runs:
        blob = torch.load(latest(run), map_location=device, weights_only=False)
        c = blob["cfg"]
        if args.fpt_K:
            c["fpt_K"] = args.fpt_K
        m = build(c, device)
        miss, unexp = m.load_state_dict(blob["model"], strict=False)
        dropped = [k for k in unexp if "lif." not in k]
        if dropped:
            print(f"[WARN] {run}: dropped {dropped[:3]} — arch mismatch!")
        models[run] = m; cfgs[run] = c
        print(f"loaded {run}: {sum(p.numel() for p in m.parameters())/1e6:.1f}M")

    vocab = cfgs[runs[0]]["vocab"]
    # compute all metrics once
    cand_rows, rank_rows = {}, {}
    for run in runs:
        cand, ranks = [], []
        for N in Ns:
            _, r, ca = recall(models[run], cfgs[run], N, device, B=args.B)
            cand.append(ca); ranks.append(r)
        cand_rows[run] = cand; rank_rows[run] = ranks

    print(f"\nMQAR associative recall (single presentation, B={args.B}).\n")
    print("AMONG-CANDIDATES ACCURACY  (of the N in-context values, pick the right one)")
    print(f"{'chance = 1/N':>22} | " + " | ".join(f"{100/n:>4.0f}%" for n in Ns))
    print(f"{'run':>22} | " + " | ".join(f"N={n:>3}" for n in Ns))
    print("-" * (24 + 8*len(Ns)))
    for run in runs:
        print(f"{run:>22} | " + " | ".join(f"{a*100:>4.0f}%" for a in cand_rows[run]))
    print("\nMEAN FULL-VOCAB TARGET RANK (lower = better; chance≈%d)" % (vocab//2))
    print(f"{'run':>22} | " + " | ".join(f"N={n:>3}" for n in Ns))
    print("-" * (24 + 8*len(Ns)))
    for run in runs:
        print(f"{run:>22} | " + " | ".join(f"{r:>5.0f}" for r in rank_rows[run]))


if __name__ == "__main__":
    main()
