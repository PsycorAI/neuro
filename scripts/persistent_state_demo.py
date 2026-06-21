"""Persistent synaptic state demo at 92M scale.

Blog #2 hero. Story:
  1. Prime the model with a (cue, target) sequence.
  2. Save the resulting M (the "brain") to disk -> brain_primed.brain
  3. Tear down the model entirely. Re-instantiate from the checkpoint.
  4. Run recall WITHOUT loading the brain  -> "naive" baseline.
  5. Load the brain from disk, run recall  -> "resumed" recall.
  6. Show: the resumed model retains the binding the original primed model had.

Same logit-delta methodology as brain_mod_composition.py.

Usage:
    python scripts/persistent_state_demo.py
"""
import os, sys, glob, argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import SpikingHebbianLM

ROOT = os.path.join(os.path.dirname(__file__), "..")
CK_DIR = os.path.join(ROOT, "models", "checkpoints", "phase3_5090")
BRAIN_DIR = os.path.join(ROOT, "models", "brains")
BRAIN_PATH = os.path.join(BRAIN_DIR, "persistent_demo.brain")


def latest(d):
    cks = sorted(glob.glob(os.path.join(d, "step_*.pt")))
    if not cks:
        raise FileNotFoundError(f"no checkpoints in {d}")
    return cks[-1]


def build_from_ckpt(device):
    """Instantiate a fresh model from the latest phase3_5090 checkpoint."""
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


def build_concept_set(n_pairs, seed=0):
    g = torch.Generator().manual_seed(seed)
    cues = (torch.randperm(1000, generator=g)[:n_pairs] + 1000).tolist()
    targets = (torch.randperm(1000, generator=g)[:n_pairs] + 8000).tolist()
    return cues, targets


@torch.no_grad()
def prime_and_save(model, cues, targets, repeats, device, brain_path):
    seq = []
    for _ in range(repeats):
        for c, t in zip(cues, targets):
            seq.append(c); seq.append(t)
    x = torch.tensor(seq, device=device).unsqueeze(0)
    _, state = model(x, return_final_state=True)
    model.save_brain(state, brain_path)
    sz = os.path.getsize(brain_path) / 1024
    return sz


def zero_brain(model, device):
    L = model.n_layers
    return {"layers": [
        {"M": torch.zeros(model.d_mem, model.d_mem, device=device),
         "mem": torch.zeros(model.n_neurons, device=device),
         "prev_spk": torch.zeros(model.n_neurons, device=device)}
        for _ in range(L)
    ]}


@torch.no_grad()
def recall_stats(model, brain, baseline_brain, cues, targets, device):
    """Mean target Δlogit and mean target rank."""
    deltas, ranks_post, ranks_base = [], 0, 0
    for c, t in zip(cues, targets):
        x = torch.tensor([[c]], device=device)
        l_base = model(x, initial_state=baseline_brain)[0, -1]
        l_post = model(x, initial_state=brain)[0, -1]
        deltas.append((l_post[t] - l_base[t]).item())
        ranks_post += (l_post > l_post[t]).sum().item() + 1
        ranks_base += (l_base > l_base[t]).sum().item() + 1
    n = len(cues)
    return sum(deltas) / n, ranks_post / n, ranks_base / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_pairs", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device
    os.makedirs(BRAIN_DIR, exist_ok=True)

    cues, targets = build_concept_set(args.n_pairs)

    print("=" * 70)
    print("STEP 1: instantiate model from checkpoint, prime, save brain to disk")
    print("=" * 70)
    m, c, ckpt = build_from_ckpt(device)
    print(f"  loaded:   {ckpt}")
    nparams = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"  params:   {nparams:.1f}M")
    sz = prime_and_save(m, cues, targets, args.repeats, device, BRAIN_PATH)
    print(f"  primed with {args.repeats}x cycle of {args.n_pairs} pairs")
    print(f"  brain saved to disk: {BRAIN_PATH} ({sz:.1f} KB)")

    # Same model instance, with-brain vs without-brain.
    base = zero_brain(m, device)
    in_session_brain = SpikingHebbianLM.load_brain(BRAIN_PATH)
    in_session_brain = {"layers": [
        {k: v.to(device) for k, v in lay.items()} for lay in in_session_brain["layers"]
    ]}

    d_in, r_in, _ = recall_stats(m, in_session_brain, base, cues, targets, device)
    print(f"\n  in-session recall: Δlogit={d_in:+.3f}  target rank={r_in:.0f}")

    print("\n" + "=" * 70)
    print("STEP 2: tear down model entirely, build a FRESH instance")
    print("=" * 70)
    del m
    torch.cuda.empty_cache() if device == "cuda" else None
    m2, _, _ = build_from_ckpt(device)
    print(f"  fresh model instance: params={sum(p.numel() for p in m2.parameters())/1e6:.1f}M")

    print("\nSTEP 3a: recall WITHOUT loading the brain (naive baseline)")
    d_naive, r_naive, r_chance = recall_stats(m2, base, base, cues, targets, device)
    print(f"  naive recall: Δlogit={d_naive:+.3f}  target rank={r_naive:.0f}  (chance={r_chance:.0f})")

    print("\nSTEP 3b: load brain from disk, recall WITH loaded brain")
    loaded = SpikingHebbianLM.load_brain(BRAIN_PATH)
    loaded = {"layers": [
        {k: v.to(device) for k, v in lay.items()} for lay in loaded["layers"]
    ]}
    d_loaded, r_loaded, _ = recall_stats(m2, loaded, base, cues, targets, device)
    print(f"  resumed recall: Δlogit={d_loaded:+.3f}  target rank={r_loaded:.0f}")

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"  same-session  primed:    Δlogit={d_in:+.3f}  rank={r_in:.0f}")
    print(f"  fresh model   no brain:  Δlogit={d_naive:+.3f}  rank={r_naive:.0f}")
    print(f"  fresh model   loaded:    Δlogit={d_loaded:+.3f}  rank={r_loaded:.0f}")
    rank_recovery = (r_naive - r_loaded) / max(1, (r_naive - r_in)) * 100
    print(f"\n  rank recovery from brain load: {rank_recovery:.1f}%")
    print(f"  (100% = brain load fully restores the in-session binding)")
    print(f"  brain file size: {sz:.1f} KB")


if __name__ == "__main__":
    main()
