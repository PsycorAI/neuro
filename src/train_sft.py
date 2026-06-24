"""SFT (supervised fine-tuning) trainer for PsycorNeuro.

Loads a pretrained base checkpoint, trains on instruction-response pairs
(Alpaca-format), with loss masked to response tokens only. Saves to a new
run_name. Resume from the SFT ckpt works too (auto-resume from latest step_*).

  python src/train_sft.py --config configs/sft_30m.yaml

Config keys (in addition to standard arch/optim):
  sft_init:     base ckpt path or run_name (mandatory)
  sft_dataset:  HF dataset id (default: yahma/alpaca-cleaned)
  sft_split:    dataset split (default: train)
  sft_max_len:  max sequence length for SFT (default: block_size)
"""
import os, sys, time, math, argparse, glob
import numpy as np
import torch
import torch.nn.functional as F
import yaml
sys.path.insert(0, os.path.dirname(__file__))
from model import SpikingHebbianLM
from sft_data import load_alpaca, build_sft_examples, SFTLoader


def build_model(c):
    return SpikingHebbianLM(c["vocab"], d=c["d"], n_neurons=c["n_neurons"],
                            d_mem=c["d_mem"], recurrent=c.get("recurrent", False),
                            rec_density=c.get("rec_density", 0.05),
                            compile_safe=c.get("compile", False),
                            tie_weights=c.get("tie_weights", False),
                            n_layers=c.get("n_layers", 1),
                            use_fpt=c.get("use_fpt", False),
                            fpt_K=c.get("fpt_K", 10),
                            beta=c.get("beta", 0.9), lam=c.get("lam", 0.98),
                            eta=c.get("eta", 1.0),
                            learnable_decay=c.get("learnable_decay", False),
                            write_gate=c.get("write_gate", False),
                            delta_rule=c.get("delta_rule", False),
                            beta_floor=c.get("beta_floor", 0.0),
                            decay_gate=c.get("decay_gate", False),
                            titans=c.get("titans", False),
                            local_attn=c.get("local_attn", False),
                            local_window=c.get("local_window", 64),
                            n_heads=c.get("n_heads", 1),
                            pre_conv=c.get("pre_conv", False),
                            pre_conv_kernel=c.get("pre_conv_kernel", 4),
                            vector_beta=c.get("vector_beta", False))


def build_optimizers(model, c, device):
    kind = c.get("optimizer", "adamw").lower()
    if kind == "muon":
        from muon import build_muon_adamw
        muon, adamw, n_mat, n_other = build_muon_adamw(
            model, muon_lr=c.get("muon_lr", 0.001),
            adamw_lr=c["lr"], adamw_fused=(device == "cuda"))
        print(f"optimizer: Muon ({n_mat} matrices, lr={c.get('muon_lr', 0.001)}) "
              f"+ AdamW ({n_other} other, lr={c['lr']})")
        return [o for o in (muon, adamw) if o is not None]
    fused_kw = {"fused": True} if device == "cuda" else {}
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], **fused_kw)
    print(f"optimizer: AdamW{' (fused)' if fused_kw else ''} (lr={c['lr']})")
    return [opt]


def load_lut_encode(vocab):
    """Returns text→list[int] encoder using the project LUT + Llama-3 tokenizer."""
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")
    lut_path = f"/home/glenn/projects/neuro/data/vocab_map_{vocab}.npy"
    lut = np.load(lut_path)
    def encode(text):
        ids = tk.encode(text, add_special_tokens=False)
        return [int(lut[t]) for t in ids if lut[t] != 0]
    return encode


def save_ckpt(ckdir, model, opts, step, tokens, c):
    p = f"{ckdir}/step_{step:07d}.pt"
    torch.save({"model": model.state_dict(),
                "opt_states": [o.state_dict() for o in opts],
                "step": step, "tokens": tokens, "cfg": c}, p)
    for o in sorted(glob.glob(f"{ckdir}/step_*.pt"))[:-c.get("keep_last", 2)]:
        os.remove(o)
    print(f"  saved {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    with open(args.config) as f:
        c = yaml.safe_load(f)
    device = args.device
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(c.get("seed", 0))

    # Build model
    model = build_model(c).to(device)
    nparams = sum(p.numel() for p in model.parameters()) / 1e6

    # Load BASE pretrained weights (sft_init) — model-only, optimizer fresh.
    sft_init = c["sft_init"]
    if "/" not in sft_init:
        cks = sorted(glob.glob(
            f"/home/glenn/projects/neuro/models/checkpoints/{sft_init}/step_*.pt"))
        if not cks:
            raise FileNotFoundError(f"no base ckpt found for sft_init={sft_init}")
        sft_init = cks[-1]
    blob = torch.load(sft_init, map_location=device, weights_only=False)
    miss, unexp = model.load_state_dict(blob["model"], strict=False)
    if unexp: print(f"  [info] ignoring {len(unexp)} ckpt keys (e.g. {unexp[0]})")
    if miss:  print(f"  [warn] {len(miss)} keys missing in ckpt: {miss[:3]}")
    print(f"loaded BASE: {sft_init} ({blob.get('tokens',0)/1e9:.2f}B tok)")

    opts = build_optimizers(model, c, device)

    # Data
    encode = load_lut_encode(c["vocab"])
    max_len = c.get("sft_max_len", c["block_size"])
    print(f"loading SFT dataset {c.get('sft_dataset','yahma/alpaca-cleaned')}...")
    raw = load_alpaca(c.get("sft_dataset", "yahma/alpaca-cleaned"),
                       c.get("sft_split", "train"))
    print(f"  raw: {len(raw)} examples")
    examples, skipped = build_sft_examples(raw, encode, max_len)
    print(f"  encoded: {len(examples)} (skipped {skipped})")

    loader = SFTLoader(examples, c["batch_size"], seed=c.get("seed", 0))

    # Resume from SFT ckpt if present
    ckdir = f"/home/glenn/projects/neuro/models/checkpoints/{c['run_name']}"
    os.makedirs(ckdir, exist_ok=True)
    step, tokens = 0, 0
    cks = sorted(glob.glob(f"{ckdir}/step_*.pt"))
    if cks:
        s = torch.load(cks[-1], map_location=device, weights_only=False)
        model.load_state_dict(s["model"], strict=False)
        if "opt_states" in s and len(s["opt_states"]) == len(opts):
            for o, st in zip(opts, s["opt_states"]): o.load_state_dict(st)
        step, tokens = s["step"], s["tokens"]
        print(f"resumed SFT ckpt @ step {step}, {tokens/1e6:.1f}M tok")

    per_step_tok = c["batch_size"] * c["grad_accum"] * max_len
    total_steps = max(1, c["max_tokens"] // per_step_tok)
    warmup_steps = int(c.get("lr_warmup_frac", 0.02) * total_steps)
    lr_min_frac = c.get("lr_min_frac", 0.1)
    grad_clip = c.get("grad_clip", 1.0)

    def lr_scale(s):
        if s < warmup_steps: return s / max(1, warmup_steps)
        p = (s - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr_min_frac + (1 - lr_min_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    base_lrs = [[g["lr"] for g in o.param_groups] for o in opts]
    print(f"arch={c['arch']} params={nparams:.2f}M device={device} | "
          f"per_step_tok={per_step_tok} | total_steps≈{total_steps} | warmup={warmup_steps}\n")

    t0 = time.time(); win = t0
    amp = c["amp"] and device == "cuda"
    while tokens < c["max_tokens"]:
        model.train()
        scale = lr_scale(step)
        for o, lrs in zip(opts, base_lrs):
            for g, base in zip(o.param_groups, lrs): g["lr"] = base * scale
        for o in opts: o.zero_grad(set_to_none=True)

        for _ in range(c["grad_accum"]):
            ids, lbl, msk = next(loader)
            ids = ids.to(device); lbl = lbl.to(device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
                out = model(ids)
                loss = F.cross_entropy(out.reshape(-1, c["vocab"]),
                                        lbl.reshape(-1), ignore_index=-100)
            (loss / c["grad_accum"]).backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        for o in opts: o.step()
        step += 1; tokens += per_step_tok

        if step % c["log_every"] == 0:
            dt = time.time() - win; win = time.time()
            vram = (f" | vram {torch.cuda.memory_allocated()/1024**3:.1f}/"
                    f"{torch.cuda.memory_reserved()/1024**3:.1f} GB"
                    if device == "cuda" else "")
            print(f"step {step} | {tokens/1e6:.1f}M tok | loss {loss.item():.3f} "
                  f"| {c['log_every']*per_step_tok/dt:,.0f} tok/s{vram}", flush=True)
        if step % c["ckpt_every"] == 0:
            save_ckpt(ckdir, model, opts, step, tokens, c)

    save_ckpt(ckdir, model, opts, step, tokens, c)
    print(f"done: {tokens/1e6:.1f}M tok in {(time.time()-t0)/3600:.2f}h")


if __name__ == "__main__":
    main()
