"""Measure val_ppl of a specific checkpoint file, reusing the TRAINER's exact
evaluate() + build_model() + StreamText so the number is directly comparable to
the [eval] val_ppl lines in the training logs (same EVAL_SEED, same val split,
same 100-iter protocol, same ablation).

  python scripts/ppl_eval.py --config configs/sparsity_refine_500m.yaml \
      --ckpt models/checkpoints/sparsity_refine_500m/step_0328307.pt

Used for the isolated PMA test: point --ckpt at the raw-final pretrain ckpt vs a
PMA soup built from the same run, holding --config fixed (arch must match).
"""
import argparse, sys, os, yaml
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train_gpu import build_model, evaluate, BIN
from data_stream import StreamText


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="config for arch + data + eval knobs")
    ap.add_argument("--ckpt", required=True, help="explicit checkpoint .pt to load")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    with open(args.config) as f:
        c = yaml.safe_load(f)
    c.setdefault("eval_batch", c["batch_size"])
    device = args.device
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    data = StreamText(c.get("data_bin", BIN), vocab=c["vocab"],
                      orig_vocab=c.get("data_orig_vocab", 128256))

    model = build_model(c).to(device)
    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    miss, unexp = model.load_state_dict(blob["model"], strict=False)
    dropped = [k for k in unexp if "lif." not in k]
    if dropped:
        print(f"[WARN] dropped {len(dropped)} keys (arch mismatch?): {dropped[:3]}")
    if miss:
        print(f"[WARN] {len(miss)} missing keys: {miss[:3]}")

    full, ablated, spike = evaluate(model, data, c, device, iters=args.iters)
    tok = blob.get("tokens", 0) / 1e6
    print(f"\nckpt: {os.path.basename(args.ckpt)}  (step={blob.get('step','?')}, {tok:.0f}M tok)")
    print(f"  val_ppl      = {full:.4f}")
    if ablated is not None:
        print(f"  ablated_ppl  = {ablated:.4f}  (memory Delta = {ablated-full:+.2f})")
        print(f"  spike_rate   = {spike:.4f}")


if __name__ == "__main__":
    main()
