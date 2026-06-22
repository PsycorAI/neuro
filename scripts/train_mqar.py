"""Powered test of the memory MECHANISM: train small models directly on MQAR
(associative recall) and measure recall capacity. Delta rule vs pure Hebbian,
averaged over seeds. This isolates the write rule from natural-language confounds
and from undertraining — the clean way to ask "does the delta rule give more
recall capacity?" (the DeltaNet result).

Task: sequence [k1 v1 k2 v2 ... kN vN kq], predict the value bound to query kq.
Trained with CE at the query position; evaluated as among-candidates accuracy
(of the N in-context values, pick the right one; chance = 1/N).

Uses the sequential step() path (use_fpt=false) so the delta rule applies.

Usage:
  python scripts/train_mqar.py --d_mem 64 --steps 2000 --seeds 3
"""
import os, sys, argparse
import torch, torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import SpikingHebbianLM

VOCAB = 2048
KLO, KHI = 1, 600          # key id range
VLO, VHI = 600, 1200       # value id range (disjoint)


def gen(B, N, device, g):
    """B sequences of N pairs + query. Returns x (B,2N+1), target (B,), cand (B,N)."""
    xs, tg, cd = [], [], []
    for _ in range(B):
        keys = torch.randperm(KHI - KLO, generator=g)[:N] + KLO
        vals = torch.randperm(VHI - VLO, generator=g)[:N] + VLO
        seq = torch.stack([keys, vals], 1).reshape(-1)
        qi = int(torch.randint(0, N, (1,), generator=g))
        seq = torch.cat([seq, keys[qi:qi+1]])
        xs.append(seq); tg.append(int(vals[qi])); cd.append(vals)
    return (torch.stack(xs).to(device), torch.tensor(tg, device=device),
            torch.stack(cd).to(device))


def build(memory, d_mem, device, use_fpt=False, fpt_K=10):
    return SpikingHebbianLM(VOCAB, d=128, n_neurons=256, d_mem=d_mem, n_layers=1,
                            compile_safe=True, recurrent=False, use_fpt=use_fpt,
                            fpt_K=fpt_K, lam=0.99,
                            delta_rule=(memory == "delta")).to(device)


def train_one(memory, d_mem, steps, seed, device, B=512, train_Nmax=24,
              use_fpt=False, fpt_K=10):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    m = build(memory, d_mem, device, use_fpt=use_fpt, fpt_K=fpt_K)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    m.train()
    for step in range(steps):
        N = int(torch.randint(2, train_Nmax + 1, (1,), generator=g))
        x, tgt, _ = gen(B, N, device, g)
        logits = m(x)[:, -1, :]
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    # eval recall (among candidates) vs N
    m.eval()
    Ns = [4, 8, 16, 32, 64]
    accs = {}
    with torch.no_grad():
        for N in Ns:
            x, tgt, cand = gen(256, N, device, g)
            logits = m(x)[:, -1, :]
            cl = logits.gather(1, cand)
            picked = cand.gather(1, cl.argmax(1, keepdim=True)).squeeze(1)
            accs[N] = (picked == tgt).float().mean().item()
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_mem", type=int, default=64)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--B", type=int, default=512, help="batch (bigger = higher GPU util)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--use_fpt", action="store_true",
                    help="chunked-kernel FPT fast path (needed for large d_mem)")
    ap.add_argument("--fpt_K", type=int, default=20,
                    help="FPT iterations; >=20 matches the sequential path exactly")
    args = ap.parse_args()
    Ns = [4, 8, 16, 32, 64]
    print(f"Trained MQAR — d_mem={args.d_mem}, steps={args.steps}, seeds={args.seeds}, "
          f"device={args.device}\n")
    print(f"AMONG-CANDIDATES ACCURACY (mean over {args.seeds} seeds; chance=1/N)")
    print(f"{'mechanism':>10} | " + " | ".join(f"N={n:>3}" for n in Ns))
    print(f"{'chance':>10} | " + " | ".join(f"{100/n:>4.0f}%" for n in Ns))
    print("-" * (12 + 8 * len(Ns)))
    for memory in ["hebbian", "delta"]:
        agg = {N: [] for N in Ns}
        for s in range(args.seeds):
            accs = train_one(memory, args.d_mem, args.steps, s, args.device, B=args.B,
                             use_fpt=args.use_fpt, fpt_K=args.fpt_K)
            for N in Ns: agg[N].append(accs[N])
        means = {N: sum(v) / len(v) for N, v in agg.items()}
        print(f"{memory:>10} | " + " | ".join(f"{means[N]*100:>4.0f}%" for N in Ns))


if __name__ == "__main__":
    main()
