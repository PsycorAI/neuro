"""prune_ffn.py -- 2026-10-25 (Neuro 78, Glenn: build + test a pruned copy). Removes FFN units of the v4 attention blocks
that can never fire (SparseAct fires when a > theta; temp/n78_dead_scan.py measured each unit's maximum margin over
1M+ tokens, with and without the brains). A unit that never fires outputs exactly 0, so removing its W_in row and W_out
column changes nothing for those inputs, and every forward / decode step reads fewer weights.

A pruned "copy" is a run directory holding a symlink to the source checkpoint + PRUNE_KEEP.pt ({block index: kept unit
indices}); scripts/demo/engine.py applies it after loading (the source config still builds the full model). Brains:
each pruned Linear records the kept indices (_prune_rows on W_in, _prune_cols on W_out) so a LoRA delta is sliced the
same way (BrainSlot). Never edits src/.
  python scripts/demo/prune_ffn.py --src sft_v4_chat_w8 --dst sft_v4_chat_w8_pruned --margins logs/n78_dead_bound.pt"""
import os, sys, argparse
import torch
import torch.nn as nn


def apply_prune(model, keep):
    """keep: {block index: LongTensor of kept hidden units}. In place; returns the number of parameters removed."""
    from hybrid_attn import SparseAct
    removed = 0
    for i, idx in keep.items():
        b = model.blocks[int(i)]
        f = b.ffn
        assert isinstance(f, nn.Sequential) and isinstance(f[1], SparseAct), f"block {i}: not a SparseAct FFN"
        idx = torch.as_tensor(idx, dtype=torch.long)
        w_in, w_out, sa = f[0], f[2], f[1]
        dev, dt = w_in.weight.device, w_in.weight.dtype
        k = len(idx); di = idx.to(dev)
        new_in = nn.Linear(w_in.in_features, k, bias=w_in.bias is not None, device=dev, dtype=dt)
        new_out = nn.Linear(k, w_out.out_features, bias=w_out.bias is not None, device=dev, dtype=dt)
        with torch.no_grad():
            new_in.weight.copy_(w_in.weight[di])
            if w_in.bias is not None:
                new_in.bias.copy_(w_in.bias[di])
            new_out.weight.copy_(w_out.weight[:, di])
            if w_out.bias is not None:
                new_out.bias.copy_(w_out.bias)
        removed += (w_in.weight.numel() - new_in.weight.numel()) + (w_out.weight.numel() - new_out.weight.numel())
        new_in._prune_rows, new_out._prune_cols = di, di
        f[0], f[2] = new_in, new_out
        sa.hidden = k
        if sa.homeo == "unit":
            sa.theta = sa.theta[di].clone()
            sa.rate_acc = sa.rate_acc[di].clone()
        if "down" in getattr(b, "sc", {}):                    # spike code on the hidden activations: per-unit theta
            sc = b.sc["down"]
            sc.theta = nn.Parameter(sc.theta.detach()[di].clone(), requires_grad=sc.theta.requires_grad)
    return removed


def lora_delta(lin, delta):
    """A LoRA delta (out x in) for `lin`, sliced like the pruned Linear (unchanged for an unpruned one)."""
    if getattr(lin, "_prune_rows", None) is not None:
        delta = delta[lin._prune_rows.to(delta.device)]
    if getattr(lin, "_prune_cols", None) is not None:
        delta = delta[:, lin._prune_cols.to(delta.device)]
    return delta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--margins", required=True); ap.add_argument("--delta", type=float, default=0.05)
    a = ap.parse_args()
    CK = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "checkpoints")
    src, dst = os.path.join(CK, a.src), os.path.join(CK, a.dst)
    ck = sorted(f for f in os.listdir(src) if f.startswith("step_") and f.endswith(".pt"))[-1]
    os.makedirs(dst, exist_ok=True)
    if not os.path.exists(os.path.join(dst, ck)):
        os.symlink(os.path.join(src, ck), os.path.join(dst, ck))
    mg = torch.load(a.margins)
    # a bool mask per block = PROVABLY dead units (temp/n78_dead_bound.py: the input-bound proof, base + every brain);
    # float = measured max margins (temp/n78_dead_scan.py), units below -delta removed
    keep = {int(i): (torch.nonzero(~v).flatten() if v.dtype == torch.bool else torch.nonzero(v >= -a.delta).flatten())
            for i, v in mg.items()}
    torch.save(keep, os.path.join(dst, "PRUNE_KEEP.pt"))
    qf = os.path.join(src, "QUANTIZED_FROM")                  # brains of the source run stay loadable (BrainSlot)
    open(os.path.join(dst, "QUANTIZED_FROM"), "w").write(open(qf).read().strip() if os.path.exists(qf) else a.src)
    open(os.path.join(dst, "PRUNED_FROM"), "w").write(f"{a.src} delta {a.delta} margins {a.margins}\n")
    tot = sum(len(v) for v in mg.values()); kept = sum(len(v) for v in keep.values())
    print(f"[prune_ffn] {a.dst}: kept {kept} of {tot} FFN units ({100 * kept / tot:.1f}%); per block "
          + " ".join(f"b{i}:{len(v)}" for i, v in keep.items()))
