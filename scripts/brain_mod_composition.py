"""Brain-mod composition test (Blog #5).

Claim: two specialized brain mods M_A and M_B can be composed at INFERENCE TIME
by element-wise addition, M_AB = M_A + M_B, and the composed brain retrieves
cues from BOTH sets correctly. This is something a transformer can't do — its
"memory" is static weights that don't have an additive group structure.

Procedure:
  1. Load the trained phase3_5090 model (weights are FIXED — we never train).
  2. Build two disjoint concept sets:
        Set A: 20 random (cue, target) pairs, cues drawn from vocab range [1000, 2000)
        Set B: 20 random (cue, target) pairs, cues drawn from vocab range [3000, 4000)
       Targets are random tokens NOT used as cues.
  3. Expose the model to set A's pairs as a single concatenated sequence.
     Save the final state -> brain_A.
  4. Reset state, expose to set B -> brain_B.
  5. Compose: brain_AB[layer]["M"] = brain_A[layer]["M"] + brain_B[layer]["M"].
     (Keep brain_A's mem/prev_spk for the carry-state; the M is the load-bearing piece.)
  6. For each brain in {A, B, AB, base (zero M), random_M}, measure recall:
        - For each cue c, prime model with c, get logits, check if argmax == target
        - Report accuracy on set A cues and on set B cues
  7. Pass criteria:
        - brain_A: high on A, chance on B
        - brain_B: chance on A, high on B
        - brain_AB: high on BOTH (the headline result)
        - base / random: chance on both

Usage:
    python scripts/brain_mod_composition.py
"""
import os, sys, glob, math, argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import SpikingHebbianLM

ROOT = os.path.join(os.path.dirname(__file__), "..")
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
                            fpt_K=c.get("fpt_K", 10))


def build_concept_set(n_pairs, cue_range, target_range, seed):
    """Returns (cues: list[int], targets: list[int]). All unique."""
    rng = torch.Generator().manual_seed(seed)
    cues = torch.randperm(cue_range[1] - cue_range[0], generator=rng)[:n_pairs]
    cues = (cues + cue_range[0]).tolist()
    targets = torch.randperm(target_range[1] - target_range[0], generator=rng)[:n_pairs]
    targets = (targets + target_range[0]).tolist()
    return cues, targets


def expose(model, cues, targets, device, repeats=8):
    """Build a sequence of repeated (cue, target) pairs, run model, return final state."""
    seq = []
    for _ in range(repeats):
        for c, t in zip(cues, targets):
            seq.append(c); seq.append(t)
    x = torch.tensor(seq, device=device).unsqueeze(0)         # (1, T)
    with torch.no_grad():
        _, state = model(x, return_final_state=True)
    return state


def compose_brains(brain_A, brain_B):
    """M_AB = M_A + M_B per layer. Use A's mem/prev_spk as carry state."""
    out = {"layers": []}
    for la, lb in zip(brain_A["layers"], brain_B["layers"]):
        out["layers"].append({
            "M": la["M"] + lb["M"],
            "mem": la["mem"],
            "prev_spk": la["prev_spk"],
        })
    return out


def zero_brain(model, device):
    L = model.n_layers
    return {"layers": [
        {"M": torch.zeros(model.d_mem, model.d_mem, device=device),
         "mem": torch.zeros(model.n_neurons, device=device),
         "prev_spk": torch.zeros(model.n_neurons, device=device)}
        for _ in range(L)
    ]}


def random_brain(model, device, scale=0.01, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    L = model.n_layers
    return {"layers": [
        {"M": scale * torch.randn(model.d_mem, model.d_mem, device=device, generator=g),
         "mem": torch.zeros(model.n_neurons, device=device),
         "prev_spk": torch.zeros(model.n_neurons, device=device)}
        for _ in range(L)
    ]}


@torch.no_grad()
def recall_signal(model, brain, baseline_brain, cues, targets, device):
    """Average logit boost on the target token when using `brain` vs `baseline_brain`.

    A model with no working memory of (c->t) gives delta ~= 0.
    A model with strong binding gives delta > 0 (target logit specifically boosted).
    A random brain gives delta ~ noise (centered at 0 with high variance).
    """
    deltas, ranks_post, ranks_base = [], [], 0
    rank_sum_post, rank_sum_base = 0, 0
    for c, t in zip(cues, targets):
        x = torch.tensor([[c]], device=device)
        l_base = model(x, initial_state=baseline_brain)[0, -1]
        l_post = model(x, initial_state=brain)[0, -1]
        deltas.append((l_post[t] - l_base[t]).item())
        # rank of target token (1 = best prediction; lower = better)
        r_post = (l_post > l_post[t]).sum().item() + 1
        r_base = (l_base > l_base[t]).sum().item() + 1
        rank_sum_post += r_post
        rank_sum_base += r_base
    n = len(cues)
    return sum(deltas) / n, rank_sum_post / n, rank_sum_base / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_pairs", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=8,
                    help="how many times to cycle through the concept set during exposure")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device

    ckpt = latest(CK_DIR)
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    c = blob["cfg"]
    print(f"checkpoint: {ckpt}")
    m = build(c).to(device)
    m.load_state_dict(blob["model"], strict=False)
    m.eval()

    # Disjoint concept sets. Target tokens chosen from a third range to avoid overlap.
    A_cues, A_targets = build_concept_set(args.n_pairs,
                                          cue_range=(1000, 2000),
                                          target_range=(8000, 9000), seed=0)
    B_cues, B_targets = build_concept_set(args.n_pairs,
                                          cue_range=(3000, 4000),
                                          target_range=(10000, 11000), seed=1)

    print(f"\nset A: {args.n_pairs} pairs, cues in [1000,2000), targets in [8000,9000)")
    print(f"set B: {args.n_pairs} pairs, cues in [3000,4000), targets in [10000,11000)")
    print(f"exposure: {args.repeats}x cycles through each set\n")

    brain_A = expose(m, A_cues, A_targets, device, args.repeats)
    brain_B = expose(m, B_cues, B_targets, device, args.repeats)
    brain_AB = compose_brains(brain_A, brain_B)
    brain_base = zero_brain(m, device)
    brain_rand = random_brain(m, device, scale=0.01)

    print(f"{'brain':>12} | {'A: Δlogit':>9}  {'rank (vs base)':>20}  | "
          f"{'B: Δlogit':>9}  {'rank (vs base)':>20}")
    print("-" * 86)
    for name, brain in [("random M", brain_rand),
                         ("A only", brain_A),
                         ("B only", brain_B),
                         ("A+B (sum)", brain_AB)]:
        a_d, a_rp, a_rb = recall_signal(m, brain, brain_base, A_cues, A_targets, device)
        b_d, b_rp, b_rb = recall_signal(m, brain, brain_base, B_cues, B_targets, device)
        print(f"{name:>12} | {a_d:>+8.3f}   {a_rp:>7.0f} (vs {a_rb:>5.0f})    | "
              f"{b_d:>+8.3f}   {b_rp:>7.0f} (vs {b_rb:>5.0f})")

    print("\nReading the table:")
    print("  Δlogit = (logit with this brain) - (logit with zero brain), for target token.")
    print("  rank   = mean rank of target token in next-token predictions (1 = best,")
    print("           lower rank = stronger recall). Shown vs the baseline (M=0) rank.")
    print(f"  vocab size = {c['vocab']}, so chance rank = {c['vocab']/2:.0f}.")
    print("  Expected:")
    print("    A only:    A target rank drops (better),       B target rank stays/rises.")
    print("    B only:    A target rank stays/rises,           B target rank drops.")
    print("    A+B sum:   BOTH ranks drop (the composition headline).")


if __name__ == "__main__":
    main()
