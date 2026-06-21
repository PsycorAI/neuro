"""Phase 1a of the brain-mod product goal: does slow-decay exposure produce
a brain that captures multi-million-token patterns, or does M saturate / become
noise?

Mechanism: monkey-patch each block's `lam` (M decay) before exposure. Default
is 0.98 → M effectively holds the last ~50 tokens. We test 0.98, 0.999, 1.0.

Procedure:
  1. Load the phase3_5090 92M checkpoint (frozen).
  2. For each λ in [0.98, 0.999, 1.0]:
       a. Override block λ.
       b. Stream N exposure tokens through the model (no gradients).
       c. Save the resulting brain.
       d. Measure: brain L1 / L2 / max norm (saturation check), spike rate,
          and held-out val perplexity WITH the brain loaded vs M=0.
  3. Report: does any λ produce a non-saturated brain that meaningfully helps
     perplexity on held-out data from the same distribution?

A positive result (perplexity drop > 5%, no saturation) means brain-mod
exposure is a viable specialization mechanism. A negative result kills the
brain-mod-as-product story and pushes us toward full-size CPT specialists.

Usage:
    python scripts/brain_slow_decay_test.py
    python scripts/brain_slow_decay_test.py --exposure_tokens 10_000_000
"""
import os, sys, glob, math, argparse, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import SpikingHebbianLM
from data_stream import StreamText

ROOT = os.path.join(os.path.dirname(__file__), "..")
CK_DIR = os.path.join(ROOT, "models", "checkpoints", "phase3_5090")
BRAIN_DIR = os.path.join(ROOT, "models", "brains", "slow_decay_test")
BIN = "/home/glenn/projects/bdh/data/tokenized/train.bin"


def latest(d):
    return sorted(glob.glob(os.path.join(d, "step_*.pt")))[-1]


def build_from_ckpt(device):
    ckpt = latest(CK_DIR)
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    c = blob["cfg"]
    m = SpikingHebbianLM(c["vocab"], d=c["d"], n_neurons=c["n_neurons"],
                         d_mem=c["d_mem"],
                         recurrent=c.get("recurrent", False),
                         rec_density=c.get("rec_density", 0.05),
                         compile_safe=c.get("compile", False),
                         tie_weights=c.get("tie_weights", False),
                         n_layers=c.get("n_layers", 1),
                         use_fpt=c.get("use_fpt", False),
                         fpt_K=c.get("fpt_K", 10)).to(device)
    m.load_state_dict(blob["model"], strict=False)
    m.eval()
    return m, c, ckpt


def override_lam(model, new_lam):
    """Monkey-patch lam on every block AND the LM itself."""
    model.lam = new_lam
    for b in model.blocks:
        b.lam = new_lam


@torch.no_grad()
def expose(model, data, c, device, total_tokens, block_size=128):
    """Stream `total_tokens` through the model, return the final brain state."""
    state = None
    consumed = 0
    t0 = time.time()
    while consumed < total_tokens:
        x, _ = data.batch(1, block_size, "train", device)
        out = model(x, initial_state=state, return_final_state=True)
        _, state = out
        consumed += block_size
        if consumed % (1_000_000) < block_size:
            elapsed = time.time() - t0
            rate = consumed / max(1e-6, elapsed)
            print(f"    exposed {consumed/1e6:.1f}M tokens "
                  f"({rate:.0f} tok/s)")
    return state


def brain_stats(state):
    """L1, L2, max for each layer's M. Tells us if M saturated."""
    stats = []
    for i, lay in enumerate(state["layers"]):
        M = lay["M"].float()
        if M.dim() == 3:
            M = M[0]
        stats.append({
            "layer": i,
            "l1": M.abs().sum().item(),
            "l2": M.norm().item(),
            "max": M.abs().max().item(),
        })
    return stats


@torch.no_grad()
def val_ppl_with_brain(model, data, c, device, brain, iters=20):
    """Held-out perplexity with this brain loaded as initial state."""
    block = c.get("block_size", 128)
    total = 0.0
    # Normalize brain tensors to no-batch shapes once (M=2D, mem/prev_spk=1D)
    norm_brain = {"layers": []}
    for lay in brain["layers"]:
        ll = {}
        for k, v in lay.items():
            while v.dim() > (2 if k == "M" else 1):
                v = v.squeeze(0)
            ll[k] = v
        norm_brain["layers"].append(ll)
    for _ in range(iters):
        x, y = data.batch(8, block, "val", device)
        B = x.shape[0]
        bb = {"layers": [
            {k: (v.unsqueeze(0).expand(B, *v.shape).contiguous())
             for k, v in lay.items()}
            for lay in norm_brain["layers"]
        ]}
        out = model(x, initial_state=bb)
        total += F.cross_entropy(out.reshape(-1, c["vocab"]),
                                  y.reshape(-1)).item()
    return math.exp(total / iters)


def zero_brain(model, device):
    return {"layers": [
        {"M": torch.zeros(model.d_mem, model.d_mem, device=device),
         "mem": torch.zeros(model.n_neurons, device=device),
         "prev_spk": torch.zeros(model.n_neurons, device=device)}
        for _ in range(model.n_layers)
    ]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exposure_tokens", type=int, default=2_000_000,
                    help="how many tokens of exposure per λ setting "
                         "(default 2M — enough to see saturation if it happens)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--block_size", type=int, default=128)
    args = ap.parse_args()
    device = args.device
    os.makedirs(BRAIN_DIR, exist_ok=True)

    m, c, ckpt = build_from_ckpt(device)
    print(f"checkpoint: {ckpt}")
    print(f"trained-time λ = {c.get('lam', 0.98)} ; d_mem={c['d_mem']} ; "
          f"n_layers={c.get('n_layers',1)}\n")
    data = StreamText(BIN, vocab=c["vocab"])

    # Baseline: no exposure (M = 0)
    base_brain = zero_brain(m, device)
    base_ppl = val_ppl_with_brain(m, data, c, device, base_brain, iters=20)
    print(f"[baseline]  M=0 val_ppl = {base_ppl:.2f}\n")

    results = []
    for lam in [0.98, 0.999, 1.0]:
        print(f"\n=== λ = {lam} ===")
        override_lam(m, lam)
        state = expose(m, data, c, device, args.exposure_tokens, args.block_size)
        stats = brain_stats(state)
        # Save brain
        bpath = os.path.join(BRAIN_DIR, f"lam_{lam:.4f}.brain")
        # Need original λ to save_brain — restore briefly so format_version is right
        m.lam = 0.98
        m.save_brain(state, bpath)
        m.lam = lam
        sz = os.path.getsize(bpath) / (1024*1024)
        print(f"  saved brain ({sz:.1f} MB) -> {bpath}")
        print(f"  brain stats per layer:")
        for s in stats:
            print(f"    layer {s['layer']}: L1={s['l1']:.1e}  L2={s['l2']:.1e}  "
                  f"max|M|={s['max']:.3f}")
        # held-out ppl with brain loaded
        ppl = val_ppl_with_brain(m, data, c, device, state, iters=20)
        delta = base_ppl - ppl
        delta_pct = 100 * delta / base_ppl
        print(f"  val_ppl with brain = {ppl:.2f}  (Δ vs M=0: {delta:+.2f}, "
              f"{delta_pct:+.2f}%)")
        results.append({"lam": lam, "ppl": ppl, "delta_pct": delta_pct,
                        "max_norm": max(s['max'] for s in stats)})

    # Restore default λ
    override_lam(m, 0.98)
    print("\n" + "=" * 70)
    print("SUMMARY (lower ppl + non-saturated brain = brain-mod product viable)")
    print("=" * 70)
    print(f"{'λ':>8} | {'val_ppl':>9} | {'Δ%':>8} | {'max|M|':>10} | {'verdict':>20}")
    print("-" * 70)
    print(f"{'M=0':>8} | {base_ppl:>9.2f} | {0:>8.2f} | {0:>10.2f} | {'baseline':>20}")
    for r in results:
        verdict = ("SATURATED" if r["max_norm"] > 1000 else
                   "no help" if r["delta_pct"] < 1 else
                   "promising" if r["delta_pct"] < 5 else
                   "STRONG signal")
        print(f"{r['lam']:>8} | {r['ppl']:>9.2f} | {r['delta_pct']:>+8.2f} | "
              f"{r['max_norm']:>10.2f} | {verdict:>20}")


if __name__ == "__main__":
    main()
