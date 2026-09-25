"""PMA: Pre-trained Model Averaging (checkpoint soup, ByteDance-Seed style).

Loads N checkpoints from a training run and produces a single averaged
checkpoint. Typically applied at the end of a pretraining run to reduce
noise in the final weights and often obtain a small quality lift for free
(literature: 0.2-1.0 val_ppl improvement, sometimes replaces annealing).

Usage
-----
    # Uniform average of the last 5 ckpts in a run directory:
    python scripts/pma_average.py \\
        --ckpt-dir models/checkpoints/demo_phase1b_500m_v2 \\
        --last 5 \\
        --out models/checkpoints/demo_phase1b_500m_v2/pma_last5.pt

    # Or explicit list:
    python scripts/pma_average.py \\
        --ckpts a.pt b.pt c.pt --out avg.pt

    # Weighted average (later = more weight, EMA-style, decay=0.9):
    python scripts/pma_average.py \\
        --ckpt-dir <dir> --last 8 --ema-decay 0.9 --out avg.pt

Design notes
------------
- Only the "model" state_dict is averaged. Optimizer state is preserved from
  the newest ckpt (in case you want to resume training after PMA — usually
  you won't, but it's cheap to keep).
- Weights averaged in fp32 for numerical safety, cast back to source dtype.
- Integer tensors (embeddings occasionally cast to long, buffers, counters)
  are copied from the newest ckpt without averaging — integer averaging is
  never what you want.
- Buffers are handled: same treatment as parameters if float, else copied.
- Assumes all ckpts share the exact same model architecture (state_dict keys).
  Errors loudly if not.

IMPORTANT re: 500M v2
---------------------
The current 500M v2 config has `keep_last: 2` — only 2 ckpts survive on disk
at any time. For a meaningful PMA soup, bump `keep_last` to 8 or higher on
the next trainer restart, OR set up an out-of-band snapshotter to accumulate
ckpts as they get produced. This script itself is fine — it works with any N,
including N=2.
"""
import argparse
import glob
import os
import torch


def _load_ckpts_from_dir(ckpt_dir, last=None, pattern="step_*.pt"):
    files = sorted(glob.glob(os.path.join(ckpt_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"no ckpts matching {pattern} in {ckpt_dir}")
    if last is not None:
        files = files[-last:]
    return files


def _average_state_dicts(state_dicts, weights):
    """Weighted average of a list of state_dicts. Float tensors averaged in
    fp32 then cast back to source dtype; non-float tensors copied from the
    NEWEST ckpt (weights[-1] is largest by convention -> last ckpt is newest).
    """
    keys = list(state_dicts[0].keys())
    for i, sd in enumerate(state_dicts[1:], 1):
        if list(sd.keys()) != keys:
            missing = set(keys) ^ set(sd.keys())
            raise ValueError(f"ckpt {i} has mismatched keys vs ckpt 0: {list(missing)[:5]}...")

    total = sum(weights)
    weights = [w / total for w in weights]      # normalize

    out = {}
    for k in keys:
        t0 = state_dicts[0][k]
        if not torch.is_tensor(t0):
            out[k] = t0
            continue
        if t0.dtype.is_floating_point:
            acc = torch.zeros_like(t0, dtype=torch.float32)
            for w, sd in zip(weights, state_dicts):
                acc.add_(sd[k].to(torch.float32), alpha=w)
            out[k] = acc.to(t0.dtype)
        else:
            # int/bool tensor: take from newest (last) ckpt, don't average
            out[k] = state_dicts[-1][k]
    return out


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt-dir", type=str, help="directory containing step_*.pt ckpts")
    g.add_argument("--ckpts", type=str, nargs="+", help="explicit ckpt file list (oldest first)")
    ap.add_argument("--last", type=int, default=None,
                    help="with --ckpt-dir, average the last N ckpts (default: all)")
    ap.add_argument("--ema-decay", type=float, default=None,
                    help="if set, use EMA-style weights: newest gets weight 1, "
                         "each prior gets multiplied by decay. Default: uniform.")
    ap.add_argument("--out", type=str, required=True, help="output ckpt path")
    args = ap.parse_args()

    if args.ckpt_dir:
        files = _load_ckpts_from_dir(args.ckpt_dir, last=args.last)
    else:
        files = args.ckpts
    print(f"averaging {len(files)} ckpts:")
    for f in files:
        print(f"  {f}")

    # Weights: newest last. EMA: w_i = decay^(N-1-i), so newest = 1.
    if args.ema_decay is not None:
        N = len(files)
        weights = [args.ema_decay ** (N - 1 - i) for i in range(N)]
        print(f"EMA weights (decay={args.ema_decay}): {[f'{w:.4f}' for w in weights]}")
    else:
        weights = [1.0] * len(files)
        print(f"uniform weights")

    # Incremental accumulation: hold at most ONE ckpt in RAM at a time (plus the
    # fp32 accumulator). Loading all N state_dicts at once peaks at N x model_size
    # and OOM-kills on low-RAM hosts (this box has 19GB; 8 x ~2GB models + a 5GB
    # newest blob overflowed). Float tensors are summed in fp32 then cast back;
    # non-float tensors are taken from the newest ckpt (last in `files`).
    tot_w = sum(weights)
    acc = None            # {key: fp32 accumulator} for float tensors
    keys = None           # canonical key order from ckpt 0
    ref_dtype = None      # {key: source dtype} to cast back
    newest_blob = None
    for i, f in enumerate(files):
        blob = torch.load(f, map_location="cpu", weights_only=False)
        if "model" not in blob:
            raise KeyError(f"{f}: no 'model' key; keys = {list(blob.keys())}")
        sd = blob["model"]
        if acc is None:
            keys = list(sd.keys())
            float_keys = [k for k in keys
                          if torch.is_tensor(sd[k]) and sd[k].dtype.is_floating_point]
            acc = {k: torch.zeros_like(sd[k], dtype=torch.float32) for k in float_keys}
            ref_dtype = {k: sd[k].dtype for k in float_keys}
            print(f"averaging {len(keys)} tensors ({len(float_keys)} float) incrementally ...")
        elif list(sd.keys()) != keys:
            missing = set(keys) ^ set(sd.keys())
            raise ValueError(f"ckpt {i} has mismatched keys vs ckpt 0: {list(missing)[:5]}...")
        for k in acc:
            acc[k].add_(sd[k].to(torch.float32), alpha=weights[i] / tot_w)
        if i == len(files) - 1:
            newest_blob = blob          # keep newest full blob (non-float model + opt + meta)
        else:
            del blob, sd

    newest_sd = newest_blob["model"]
    avg_sd = {k: (acc[k].to(ref_dtype[k]) if k in acc else newest_sd[k]) for k in keys}

    out_blob = dict(newest_blob)                # preserve step/tokens/cfg/opt
    out_blob["model"] = avg_sd
    out_blob["pma_meta"] = {
        "n_averaged": len(files),
        "source_ckpts": [os.path.basename(f) for f in files],
        "ema_decay": args.ema_decay,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.save(out_blob, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
