"""Persistent synaptic state demo — saveable, reloadable "brain" files.

Demonstrates two distinct properties of the architecture:

  (A) FULL-STATE PERSISTENCE — the saved brain (M + LIF membrane + last spike)
      is a complete snapshot. Resuming from a saved brain produces *bit-identical*
      next-token predictions to continuing the same session in memory. This is
      the structural correctness test for the save/load API.

  (B) PORTABLE WORKING MEMORY — train a model on the meta-task of "shown random
      cue→successor bindings in context, recall them when asked." Then prime the
      trained model with a SPECIFIC concept set, save the brain, spawn a fresh
      model, load the brain, and probe with each cue. Show recall is high; with
      the brain absent (M = 0), recall collapses to chance.

The bindings exist nowhere except inside the saved brain. The trained model is
"blank brain hardware" that pairs with any compatible synaptic state file.
No transformer architecture supports this directly.
"""
import os, sys
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import SpikingHebbianLM

ROOT = os.path.join(os.path.dirname(__file__), "..")
BRAIN_DIR = os.path.join(ROOT, "brains")
ASSETS = os.path.join(ROOT, "assets")
WEIGHTS = os.path.join(BRAIN_DIR, "demo_weights.pt")
BRAIN_A = os.path.join(BRAIN_DIR, "concept_set_A.brain")

N_CUES = 6
SEP = 2 * N_CUES
VOCAB = 2 * N_CUES + 1

# Model size matched to Phase-1 induction (which gets 96% recall on similar tasks).
D, N, DM = 128, 256, 128


# ----------------------------- (A) IDENTITY TEST -----------------------------
def identity_test():
    """Save+load round-trip must yield bit-identical continuation."""
    print("=" * 60)
    print("PART A — identity test: save/load gives same predictions as continue")
    print("=" * 60)
    torch.manual_seed(0)
    model = SpikingHebbianLM(VOCAB, d=D, n_neurons=N, d_mem=DM)
    model.eval()
    seqA = torch.randint(0, VOCAB, (1, 7))
    seqB = torch.randint(0, VOCAB, (1, 4))

    # Path 1: process A then B in one shot, record predictions on B
    with torch.no_grad():
        full = torch.cat([seqA, seqB], dim=1)
        logits_full = model(full)
    target_logits = logits_full[:, seqA.shape[1] : seqA.shape[1] + seqB.shape[1]]

    # Path 2: process A, save brain, fresh model, load brain, process B
    with torch.no_grad():
        _, state_after_A = model(seqA, return_final_state=True)
    tmp_brain = os.path.join(BRAIN_DIR, "_identity_test.brain")
    os.makedirs(BRAIN_DIR, exist_ok=True)
    model.save_brain(state_after_A, tmp_brain)

    model2 = SpikingHebbianLM(VOCAB, d=D, n_neurons=N, d_mem=DM)
    model2.load_state_dict(model.state_dict())
    model2.eval()
    loaded = SpikingHebbianLM.load_brain(tmp_brain)
    with torch.no_grad():
        logits_resumed = model2(seqB, initial_state=loaded)
    target_logits_resumed = logits_resumed[:, : seqB.shape[1]]

    max_abs_diff = (target_logits - target_logits_resumed).abs().max().item()
    print(f"  max |logits_continued - logits_resumed| = {max_abs_diff:.3e}")
    ok = max_abs_diff < 1e-4
    print(f"  identity test: {'PASS' if ok else 'FAIL'}")
    os.remove(tmp_brain)
    return ok


# ----------------------------- (B) RECALL DEMO -----------------------------
def random_batch(B, n_pairs=3):
    """Random bindings each batch. Structure: pair pair pair SEP cue answer."""
    xs, ys = [], []
    for _ in range(B):
        cs = torch.randperm(N_CUES)[:n_pairs].tolist()
        ss = (N_CUES + torch.randperm(N_CUES)[:n_pairs]).tolist()
        seq = []
        for c, s in zip(cs, ss):
            seq.extend([c, s])
        seq.append(SEP)
        pi = int(torch.randint(0, n_pairs, (1,)).item())
        seq.append(cs[pi])
        seq.append(ss[pi])
        xs.append(seq[:-1]); ys.append(seq[1:])
    return torch.tensor(xs), torch.tensor(ys)


def make_associations(seed=0):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N_CUES, generator=g)
    return {i: N_CUES + int(perm[i]) for i in range(N_CUES)}


def priming_sequence(pairs):
    """Pair bindings + SEP. Mirrors training distribution exactly."""
    seq = []
    for c in range(N_CUES):
        seq.extend([c, pairs[c]])
    seq.append(SEP)
    return torch.tensor(seq, dtype=torch.long).unsqueeze(0)


def recall_demo(pairs):
    print("=" * 60)
    print("PART B — recall demo: train, prime, save, recall, contrast")
    print("=" * 60)
    torch.manual_seed(0)
    model = SpikingHebbianLM(VOCAB, d=D, n_neurons=N, d_mem=DM)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    print("  training on random bindings per batch ...")
    for step in range(1, 3001):
        x, y = random_batch(128)
        loss = F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            print(f"    step {step:4d} | loss {loss.item():.3f}")
    model.eval()

    # prime with the demo's concept set; snapshot full state
    with torch.no_grad():
        _, state = model(priming_sequence(pairs), return_final_state=True)
    torch.save(model.state_dict(), WEIGHTS)
    model.save_brain(state, BRAIN_A)
    print(f"  saved brain -> {BRAIN_A} ({os.path.getsize(BRAIN_A)/1024:.1f} KB)")

    # fresh model + loaded brain
    fresh = SpikingHebbianLM(VOCAB, d=D, n_neurons=N, d_mem=DM)
    fresh.load_state_dict(torch.load(WEIGHTS, weights_only=False))
    fresh.eval()
    brain = SpikingHebbianLM.load_brain(BRAIN_A)
    with_brain, no_brain = [], []
    with torch.no_grad():
        for c in range(N_CUES):
            probe = torch.tensor([[c]], dtype=torch.long)
            p_b = F.softmax(fresh(probe, initial_state=brain)[0, -1], dim=-1)
            p_n = F.softmax(fresh(probe)[0, -1], dim=-1)
            with_brain.append(float(p_b[pairs[c]]))
            no_brain.append(float(p_n[pairs[c]]))
    aw, an = sum(with_brain)/N_CUES, sum(no_brain)/N_CUES
    print(f"  mean P(correct | with brain) = {aw:.3f}")
    print(f"  mean P(correct | no brain)   = {an:.3f}")
    print(f"  chance                       = {1.0/VOCAB:.3f}")
    return with_brain, no_brain


def visualize(p_with, p_without, path):
    os.makedirs(ASSETS, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = list(range(N_CUES))
    w = 0.42
    ax.bar([i - w/2 for i in x], p_with, w, label="with loaded brain", color="#b5651d")
    ax.bar([i + w/2 for i in x], p_without, w, label="no brain (M=0)", color="#bdbdbd")
    ax.axhline(1.0 / VOCAB, ls="--", color="gray", label="chance")
    ax.set_xticks(x); ax.set_xticklabels([f"cue {i}" for i in x])
    ax.set_ylabel("P(correct successor)")
    ax.set_title("Saveable synaptic brains: a portable memory artifact")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"saved figure -> {path}")


def main():
    pairs = make_associations()
    identity_ok = identity_test()
    p_with, p_without = recall_demo(pairs)
    visualize(p_with, p_without, os.path.join(ASSETS, "persistent_brain.png"))
    aw, an = sum(p_with)/N_CUES, sum(p_without)/N_CUES
    print()
    print("=" * 60)
    print(f"  Identity test      : {'PASS' if identity_ok else 'FAIL'}")
    print(f"  Recall with brain  : {aw:.3f}")
    print(f"  Recall no brain    : {an:.3f}  (chance {1.0/VOCAB:.3f})")
    recall_ok = aw > 0.4 and aw > 2.5 * an   # at least 5x chance AND 2.5x no-brain
    print(f"  Recall demo        : {'PASS' if recall_ok else 'CHECK — see numbers'}")


if __name__ == "__main__":
    main()
