"""MFU instrumentation for flagship_v4 throughput work.

Nothing in this repo measured MFU, so every throughput decision so far has been
made blind. This reports it exactly.

FLOPs are MEASURED with torch.utils.flop_counter.FlopCounterMode over a real
forward+backward, not estimated with the 6N rule. 6N is wrong for this
architecture by a wide margin: the spiking blocks are n_neurons-wide (3.0x d),
GDN-2 runs a chunked delta rule, engram-v2 adds gather-based n-gram reads, and
MTP runs a second head pass. The per-module breakdown is the actionable part --
it says which component to attack first.

  # FLOP count only (no GPU needed)
  python scripts/mfu_report.py --config configs/flagship_v3_363m.yaml --tok-s 9700

  # measure tok/s live too
  CUDA_VISIBLE_DEVICES=1 python scripts/mfu_report.py --config configs/... --measure --gpu rtx5090

  # project a hypothetical v4
  python scripts/mfu_report.py --config configs/flagship_v3_363m.yaml \
      --override d=2048 n_neurons=6144 d_mem=2048 n_layers=16 --tokens 20e9
"""
import os, sys, time, argparse, yaml
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from train_gpu import build_model

# BF16 dense tensor-core peak, FP32 accumulate (the realistic training number).
# Consumer cards halve their rate with FP32 accumulate, which is why the widely
# quoted 5090 "209.5 TFLOPS" figure is the FP16-accumulate one and must not be
# used as a training denominator.
GPU_PEAK_BF16 = {
    "rtx4090":   82.6,
    "rtx5080":   56.3,
    "rtx5090":  104.8,
    "a100":     312.0,
    "h100_pcie": 378.0,
    "h100":     494.0,
    "h200":     494.0,
    "b200":    1100.0,
}
# FP8 roughly doubles the tensor-core rate where the hardware supports it
# (Ada, Hopper, Blackwell). Ampere (A100) has no FP8 path.
FP8_CAPABLE = {"rtx4090", "rtx5080", "rtx5090", "h100", "h100_pcie", "h200", "b200"}


def count_flops(model, ids, device):
    """Exact fwd+bwd FLOPs for one step, plus a per-module breakdown."""
    from torch.utils.flop_counter import FlopCounterMode
    model.train()
    ctr = FlopCounterMode(display=False, depth=2)
    with ctr:
        out = model(ids)
        logits = out[0] if isinstance(out, tuple) else out
        logits.float().pow(2).mean().backward()   # cheap surrogate loss
    total = ctr.get_total_flops()
    by_mod = {}
    for mod, counts in ctr.get_flop_counts().items():
        s = sum(counts.values())
        if s:
            by_mod[mod] = s
    model.zero_grad(set_to_none=True)
    return total, by_mod


@torch.no_grad()
def measure_tok_s(model, ids, device, iters=8):
    """Rough fwd-only throughput; multiply by ~1/3 for a fwd+bwd step rate.
    Only used when a real training log is not available."""
    model.eval()
    for _ in range(3):
        model(ids)
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()
    for _ in range(iters):
        model(ids)
    torch.cuda.synchronize() if device == "cuda" else None
    dt = (time.time() - t0) / iters
    return ids.numel() / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[],
                    help="key=value arch overrides, e.g. d=2048 n_layers=16")
    ap.add_argument("--tok-s", type=float, default=None,
                    help="measured training tok/s from a real run log")
    ap.add_argument("--measure", action="store_true",
                    help="measure fwd-only tok/s on this device instead")
    ap.add_argument("--gpu", default="rtx5090", choices=sorted(GPU_PEAK_BF16))
    ap.add_argument("--tokens", type=float, default=None,
                    help="training token budget, for a wall-clock projection")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    with open(args.config) as f:
        c = yaml.safe_load(f)
    for kv in args.override:
        k, v = kv.split("=", 1)
        try:
            c[k] = yaml.safe_load(v)
        except Exception:
            c[k] = v

    # One microbatch is enough: FLOPs scale linearly in batch and sequence.
    B = min(2, c["batch_size"])
    T = c["block_size"]
    model = build_model(c).to(args.device)
    nparams = sum(p.numel() for p in model.parameters())
    ids = torch.randint(1, c["vocab"], (B, T), device=args.device)

    total, by_mod = count_flops(model, ids, args.device)
    tokens = B * T
    fpt = total / tokens
    naive = 6 * nparams

    print(f"config        {os.path.basename(args.config)}")
    print(f"params        {nparams/1e6:.1f}M")
    print(f"probe         B={B} T={T} ({tokens} tokens)")
    print(f"\nFLOPs/token   {fpt/1e9:.2f} G  (measured, fwd+bwd)")
    print(f"  6N estimate {naive/1e9:.2f} G   -> measured is {fpt/naive:.2f}x the 6N rule")

    # Aggregate leaf modules by weight NAME across layers: "all 9 W_fdecay
    # projections together" is the number that tells you what to optimize, not
    # the share of any single layer. Skip the container rows (Global, the LM
    # itself), which are hierarchical totals and would double count.
    # A key is a true leaf only if no OTHER key extends it. FlopCounterMode
    # reports every module in the hierarchy, so counting a parent (W_fdecay) and
    # its children (W_fdecay.down/.up) both was inflating shares past 100%.
    keys = [k for k in by_mod if k and k not in ("Global", "SpikingHebbianLM")]
    leaves = [k for k in keys
              if not any(o != k and o.startswith(k + ".") for o in keys)]
    agg = {}
    for mod in leaves:
        parts = mod.split(".")
        # name the component by its last non-numeric path element, and keep the
        # parent for factored gates so "W_fdecay.down" stays distinguishable
        name = ".".join(p for p in parts[-2:] if not p.isdigit())
        agg[name] = agg.get(name, 0) + by_mod[mod]
    # Shares are of the LEAF sum, not of get_total_flops(): FlopCounterMode
    # attributes each op to a module and all its ancestors, so leaf entries
    # overlap the container rows and would otherwise sum past 100%.
    leaf_total = sum(agg.values()) or 1
    # The leaf sum does not equal get_total_flops() -- FlopCounterMode's
    # per-module attribution counts some backward ops differently from the global
    # tally (observed ~2.07x on this model). So these shares are RELATIVE
    # rankings, not absolute fractions of the step. Scale by inflation to read
    # them as a fraction of the real step cost.
    infl = leaf_total / total
    print(f"\nFLOP share by component (relative; leaf sum is {infl:.2f}x the "
          f"global tally, so divide by {infl:.2f} for a true fraction of the step):")
    for name, f in sorted(agg.items(), key=lambda kv: -kv[1]):
        share = 100 * f / leaf_total
        if share < 0.4:
            continue
        print(f"  {share:5.1f}%  {f/1e9:9.2f} G  {name}")

    tok_s = args.tok_s
    if args.measure and args.device == "cuda":
        fwd = measure_tok_s(model, ids, args.device)
        print(f"\nfwd-only tok/s (this device) {fwd:,.0f}")
        tok_s = tok_s or fwd / 3.0      # fwd+bwd ~= 3x fwd cost
    if not tok_s:
        print("\n(pass --tok-s from a real training log for the MFU figure)")
        return

    achieved = tok_s * fpt
    peak = GPU_PEAK_BF16[args.gpu] * 1e12
    print(f"\ntok/s         {tok_s:,.0f}")
    print(f"achieved      {achieved/1e12:.1f} TFLOPS")
    print(f"{args.gpu} peak  {peak/1e12:.1f} TFLOPS (BF16 dense, FP32 accum)")
    print(f"MFU           {100*achieved/peak:.1f}%")
    if args.gpu in FP8_CAPABLE:
        print(f"MFU vs FP8    {100*achieved/(2*peak):.1f}%  "
              f"(FP8 roughly doubles the ceiling on this card)")

    if args.tokens:
        days = args.tokens / tok_s / 86400
        print(f"\n{args.tokens/1e9:.0f}B tokens at this rate: {days:.1f} days")
        for mult in (1.25, 1.5, 1.85, 2.5):
            print(f"  with {mult:.2f}x throughput: {days/mult:.1f} days")


if __name__ == "__main__":
    main()
