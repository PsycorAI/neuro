"""Eval the latest phase3_5090 checkpoint.

Reports val ppl, memory-ablation Δ, spike rate, energy per token.
Saves a brain file for later persistent-state demos.

    python scripts/eval_phase3.py
    python scripts/eval_phase3.py --iters 100
"""
import os, sys, glob, math, time, argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import SpikingHebbianLM
from data_stream import StreamText
import energy

ROOT = os.path.join(os.path.dirname(__file__), "..")
BIN = "/home/glenn/projects/bdh/data/tokenized/train.bin"
CK_DIR = os.path.join(ROOT, "models", "checkpoints", "phase3_5090")


def latest(d):
    cks = sorted(glob.glob(os.path.join(d, "step_*.pt")))
    if not cks:
        raise FileNotFoundError(f"no checkpoints in {d}")
    return cks[-1]


def build(c):
    return SpikingHebbianLM(c["vocab"], d=c["d"], n_neurons=c["n_neurons"],
                            d_mem=c["d_mem"],
                            recurrent=c.get("recurrent", False),
                            rec_density=c.get("rec_density", 0.05),
                            compile_safe=c.get("compile", False),
                            tie_weights=c.get("tie_weights", False),
                            n_layers=c.get("n_layers", 1),
                            use_fpt=c.get("use_fpt", False),
                            fpt_K=c.get("fpt_K", 10),
                                lam=c.get("lam", 0.98),
                                learnable_decay=c.get("learnable_decay", False),
                                write_gate=c.get("write_gate", False),
                                delta_rule=c.get("delta_rule", False),
                                beta_floor=c.get("beta_floor", 0.0),
                                decay_gate=c.get("decay_gate", False))


@torch.no_grad()
def val_ppl(model, data, c, device, iters, ablate=False):
    total = 0.0
    for _ in range(iters):
        x, y = data.batch(c["eval_batch"], c["block_size"], "val", device)
        out = model(x, ablate_memory=ablate)
        total += F.cross_entropy(out.reshape(-1, c["vocab"]), y.reshape(-1)).item()
    return math.exp(total / iters)


@torch.no_grad()
def spike_rate(model, data, c, device):
    x, _ = data.batch(c["eval_batch"], c["block_size"], "val", device)
    _, sr = model(x, return_stats=True)
    return sr.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    device = args.device

    ckpt = latest(CK_DIR)
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    c = blob["cfg"]
    print(f"checkpoint: {ckpt}")
    print(f"  step {blob.get('step')}, tokens {blob.get('tokens', 0)/1e9:.3f}B")
    print(f"  arch={c['arch']} n_layers={c.get('n_layers',1)} use_fpt={c.get('use_fpt',False)}")

    m = build(c).to(device)
    m.load_state_dict(blob["model"], strict=False)
    m.eval()
    nparams = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"  params: {nparams:.1f}M")

    data = StreamText(BIN, vocab=c["vocab"])

    t0 = time.time()
    full = val_ppl(m, data, c, device, args.iters, ablate=False)
    abl = val_ppl(m, data, c, device, args.iters, ablate=True)
    sr = spike_rate(m, data, c, device)
    print(f"\n=== eval (iters={args.iters}, T={c['block_size']}) ===")
    print(f"  val_ppl       {full:.2f}")
    print(f"  ablated_ppl   {abl:.2f}   (memory Δ = {abl-full:+.2f})")
    print(f"  spike_rate    {sr:.3f}   ({(1-sr)*100:.1f}% silent)")
    print(f"  eval time:    {time.time()-t0:.1f}s")

    # Energy comparison vs transformer baseline (n_layer=4, d=1024).
    print(f"\n=== theoretical inference energy (45nm AC/MAC) ===")
    energy.compare(m, sr, baseline_d=c["d"], baseline_layers=c.get("n_layers", 4),
                   vocab=c["vocab"], seq_lens=[c["block_size"], 512, 2048, 4096])

    # Save a brain snapshot for persistent-state demos.
    brain_path = os.path.join(ROOT, "models", "brains", f"phase3_5090_step{blob.get('step')}.brain")
    os.makedirs(os.path.dirname(brain_path), exist_ok=True)
    x, _ = data.batch(1, c["block_size"], "val", device)
    _, state = m(x, return_final_state=True)
    m.save_brain(state, brain_path)
    sz = os.path.getsize(brain_path) / 1024
    print(f"\nsaved brain ({sz:.1f} KB) -> {brain_path}")


if __name__ == "__main__":
    main()
