"""TIES / DARE-TIES model merging for the 500M v3 post-pretrain checkpoints.

Combines multiple state_dicts of the SAME architecture by:
  1. Computing per-parameter "task vectors": delta_i = weights_i - base
  2. (DARE)  Random-dropping delta entries with prob p, rescaling by 1/(1-p)
  3. (TIES)  Keeping only top-k% of |delta| per parameter, then majority-sign
             vote across models, averaging the surviving deltas
  4. Adding merged delta back to base: merged = base + sum(alpha_i * delta_i)

Reference: TIES-Merging (Yadav et al., 2024), DARE (Yu et al., 2024).
Standard 2026 finding: DARE-TIES (DARE then TIES) outperforms either alone.

  # Merge sparsity + SFT + SimPO ckpts against the pretrain base
  python scripts/merge_ties_dare.py \\
      --base    models/checkpoints/500m_v3/step_XXXXXX.pt \\
      --models  sparsity_refine_500m_v3 sft_500m_full_v3 dpo_500m_full_v3 \\
      --out     models/checkpoints/merged_500m_v3/blend.pt \\
      --method  dare-ties --density 0.9 --topk 0.7

The output ckpt has the same schema as any train_gpu save (model + cfg keys),
so downstream tools (inference, further finetuning) load it unchanged.
"""
import os, sys, glob, argparse, torch


ROOT = "/home/glenn/projects/neuro"


def resolve_ckpt(name_or_path):
    """Accept a full path OR a run name (globs latest step_*.pt in that dir)."""
    if "/" in name_or_path or name_or_path.endswith(".pt"):
        return name_or_path
    cks = sorted(glob.glob(f"{ROOT}/models/checkpoints/{name_or_path}/step_*.pt"))
    if not cks:
        raise FileNotFoundError(f"no step_*.pt in run '{name_or_path}'")
    return cks[-1]


def load_sd(path):
    """Load state_dict from any of our checkpoint formats."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    # Standard formats: {"model": ..., "cfg": ...} or bare state_dict
    sd = blob.get("model", blob) if isinstance(blob, dict) else blob
    return sd, blob


def dare_drop(delta, density):
    """DARE: randomly zero (1-density) of delta entries; rescale surviving by 1/density.
    density=1.0 means no drop (identity); density=0.5 means drop half, rescale x2.
    Uses ONE mask across the whole tensor; matches paper's per-parameter random drop."""
    if density >= 1.0:
        return delta
    mask = (torch.rand_like(delta.float()) < density).to(delta.dtype)
    return (delta * mask) / density


def ties_merge(deltas, topk):
    """TIES: trim per-parameter to |top-k| by magnitude, majority-sign vote,
    average surviving deltas.
    deltas: list of tensors, all same shape.
    Returns: merged delta of same shape."""
    if not deltas:
        return None
    stacked = torch.stack(deltas, dim=0)  # (N, *shape)

    # 1. Trim: for each model's delta, zero out entries below top-k% by |value|.
    if topk < 1.0:
        flat = stacked.view(stacked.shape[0], -1)
        k = max(1, int(topk * flat.shape[1]))
        # Per-model threshold from top-k magnitudes
        thresh = flat.abs().topk(k, dim=1).values[:, -1:]  # (N, 1)
        trim_mask = (flat.abs() >= thresh).to(flat.dtype)
        flat = flat * trim_mask
        stacked = flat.view_as(stacked)

    # 2. Elect sign: for each parameter, majority sign of NONZERO deltas across models.
    #    (Sum of signs; if positive, keep positive deltas; if negative, keep negatives; zero: skip.)
    sign_sum = stacked.sum(dim=0).sign()  # (*shape); 0 if all cancel

    # 3. Keep only deltas whose sign matches the elected majority.
    sign_stacked = stacked.sign()
    keep_mask = (sign_stacked == sign_sum.unsqueeze(0)) & (stacked != 0)  # (N, *shape)

    # 4. Average surviving deltas per parameter.
    counts = keep_mask.sum(dim=0).clamp_min(1).to(stacked.dtype)
    merged = (stacked * keep_mask.to(stacked.dtype)).sum(dim=0) / counts
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",   required=True,
                     help="base ckpt path or run_name (the shared starting point)")
    ap.add_argument("--models", nargs="+", required=True,
                     help="task ckpts to merge (paths or run_names)")
    ap.add_argument("--out",    required=True, help="output merged .pt path")
    ap.add_argument("--method", choices=["ties", "dare", "dare-ties"],
                     default="dare-ties")
    ap.add_argument("--density", type=float, default=0.9,
                     help="DARE: fraction of delta entries KEPT (1.0 = no drop)")
    ap.add_argument("--topk",    type=float, default=0.7,
                     help="TIES: per-parameter fraction of |delta| KEPT before sign vote")
    ap.add_argument("--alpha",   type=float, nargs="+", default=None,
                     help="per-model scaling factors for the merged delta "
                          "(defaults to 1/N each)")
    ap.add_argument("--seed",    type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    base_path = resolve_ckpt(args.base)
    base_sd, base_blob = load_sd(base_path)
    print(f"[merge] base:   {os.path.basename(base_path)}   ({len(base_sd)} tensors)")

    model_paths = [resolve_ckpt(m) for m in args.models]
    model_sds = []
    for p in model_paths:
        sd, _ = load_sd(p)
        model_sds.append(sd)
        print(f"[merge] task:   {os.path.basename(p)}   ({len(sd)} tensors)")

    N = len(model_sds)
    alphas = args.alpha if args.alpha is not None else [1.0 / N] * N
    if len(alphas) != N:
        raise ValueError(f"--alpha must have {N} values (one per model), got {len(alphas)}")
    print(f"[merge] method: {args.method} | density={args.density} topk={args.topk} alphas={alphas}")

    merged = {}
    skipped = 0
    for name, base_w in base_sd.items():
        # Collect per-model deltas for this parameter; skip parameters missing from any task
        deltas = []
        for sd in model_sds:
            if name not in sd or sd[name].shape != base_w.shape:
                deltas = None
                break
            deltas.append(sd[name].to(torch.float32) - base_w.to(torch.float32))
        if deltas is None:
            merged[name] = base_w.clone()   # keep base if any task lacks this param
            skipped += 1
            continue

        if args.method == "dare":
            # DARE alone: drop+rescale each delta, weighted-average, add to base
            new_deltas = [dare_drop(d, args.density) for d in deltas]
            merged_delta = sum(a * d for a, d in zip(alphas, new_deltas))
        elif args.method == "ties":
            # TIES alone: trim+sign-vote+avg across models, then apply weighted alpha as a scalar
            m = ties_merge(deltas, args.topk)
            merged_delta = sum(alphas) * m
        else:  # dare-ties
            new_deltas = [dare_drop(d, args.density) for d in deltas]
            m = ties_merge(new_deltas, args.topk)
            merged_delta = sum(alphas) * m

        merged[name] = (base_w.to(torch.float32) + merged_delta).to(base_w.dtype)

    print(f"[merge] merged {len(merged) - skipped} tensors; kept {skipped} at base "
          f"(missing/shape-mismatch in a task)")

    # Preserve config from the base blob so downstream loaders see a normal ckpt
    cfg = base_blob.get("cfg") if isinstance(base_blob, dict) else None
    out = {
        "model": merged,
        "cfg": cfg,
        "merged_from": {
            "base": base_path,
            "models": model_paths,
            "method": args.method,
            "density": args.density,
            "topk": args.topk,
            "alpha": alphas,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out, args.out)
    print(f"[OK] wrote {args.out}")


if __name__ == "__main__":
    main()
