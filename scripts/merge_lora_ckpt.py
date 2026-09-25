"""Bake a LoRA adapter into its base checkpoint -> a plain (merged) checkpoint.

Folds (alpha/r) * B@A into each wrapped base weight and unwraps back to plain
nn.Linear, so the output has NO adapter keys and NO inference overhead. Use it to
chain stages (LoRA-SFT -> merge -> LoRA-DPO -> merge -> demo) or to produce a
deployable single-file model for the brain-mod demo.

  python scripts/merge_lora_ckpt.py \
      --base   models/checkpoints/<sft_base_run>/step_XXXXXXX.pt \
      --adapter models/checkpoints/<lora_run>/adapter_step_XXXXXXX.pt \
      --out    models/checkpoints/<lora_run>/merged.pt

If --base is omitted, the base path recorded inside the adapter file is used
(resolved to the latest step_*.pt of that run_name).
"""
import os, sys, glob, argparse
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train_sft import build_model
import lora as _lora

ROOT = "/home/glenn/projects/neuro"


def resolve_ckpt(path_or_run):
    if path_or_run and "/" not in path_or_run and not path_or_run.endswith(".pt"):
        cks = sorted(glob.glob(f"{ROOT}/models/checkpoints/{path_or_run}/step_*.pt"))
        if not cks:
            raise FileNotFoundError(f"no step_*.pt for run '{path_or_run}'")
        return cks[-1]
    return path_or_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="adapter_step_*.pt file")
    ap.add_argument("--base", default=None,
                    help="base ckpt path/run (default: the base recorded in the adapter)")
    ap.add_argument("--out", required=True, help="output merged .pt path")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ad = torch.load(args.adapter, map_location=args.device, weights_only=False)
    base_path = resolve_ckpt(args.base or ad.get("base"))
    if not base_path or not os.path.exists(base_path):
        raise FileNotFoundError(f"base checkpoint not found: {base_path}")

    base_blob = torch.load(base_path, map_location=args.device, weights_only=False)
    cfg = base_blob["cfg"]
    model = build_model(cfg).to(args.device)
    miss, unexp = model.load_state_dict(base_blob["model"], strict=False)
    if unexp:
        print(f"  [info] ignoring {len(unexp)} base keys (e.g. {unexp[0]})")

    r, alpha = ad.get("lora_r", 8), ad.get("lora_alpha", 16)
    targets = tuple(ad.get("lora_targets", _lora.DEFAULT_TARGETS))
    n_w, _, names = _lora.inject_lora(model, r=r, alpha=alpha, targets=targets)
    _lora.load_lora_state_dict(model, ad["lora"])
    n_merged = _lora.merge_lora(model)
    assert n_merged == n_w, (n_merged, n_w)

    # Sanity: merged model has no adapter keys and still runs a forward pass.
    assert not _lora.lora_state_dict(model), "merged model still has adapter keys"
    with torch.no_grad():
        _ = model(torch.zeros(1, 8, dtype=torch.long, device=args.device))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg,
                "step": ad.get("step", 0), "tokens": ad.get("tokens", 0),
                "merged_from": {"base": base_path, "adapter": args.adapter,
                                "r": r, "alpha": alpha}}, args.out)
    print(f"[OK] merged {n_w} adapters (r={r}, alpha={alpha}) into base "
          f"{os.path.basename(base_path)} -> {args.out}")


if __name__ == "__main__":
    main()
