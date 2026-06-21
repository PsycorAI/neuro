"""Brain-mod domain-transfer test.

Question: does a brain built from chunk-A of the corpus help on chunk-A more
than on chunk-B, and vice versa? If yes → brain is capturing local patterns
specifically and could carry domain specialization. If brain helps both chunks
equally → brain captures generic in-distribution structure, not specialization.

Procedure (frozen 92M model, no gradients anywhere):
  1. Split the validation tail into chunk A (first half) and chunk B (second half).
  2. Build brain_A by exposing the model to chunk A (with state carried across
     blocks).
  3. Build brain_B the same way from chunk B.
  4. Measure val_ppl on a fresh slice from chunk A and a fresh slice from chunk B,
     each with: (i) no brain, (ii) brain_A loaded, (iii) brain_B loaded.
  5. Specificity = (brain_A's improvement on A) − (brain_A's improvement on B).
     Higher = stronger specialization.

This is the proxy for the real domain test (math vs code vs prose). We use
two halves of our own data because we don't have domain labels in train.bin.
A positive specificity result here lets us justify the more expensive
multi-corpus test later.

Usage:
    python scripts/brain_domain_transfer.py
    python scripts/brain_domain_transfer.py --exposure_tokens 200000
"""
import os, sys, glob, math, argparse, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import SpikingHebbianLM
from data_stream import StreamText

ROOT = os.path.join(os.path.dirname(__file__), "..")
CK_DIR = os.path.join(ROOT, "models", "checkpoints", "phase3_5090")
BRAIN_DIR = os.path.join(ROOT, "models", "brains", "domain_transfer")
BIN = "/home/glenn/projects/bdh/data/tokenized/train.bin"


def latest(d):
    return sorted(glob.glob(os.path.join(d, "step_*.pt")))[-1]


def build_from_ckpt(device):
    blob = torch.load(latest(CK_DIR), map_location=device, weights_only=False)
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
    return m, c


class ChunkedStream:
    """Stream a contiguous (lo, hi) window of the binary file, no random sampling.
    Used to expose the model to a SPECIFIC region of data (a 'domain' chunk)."""

    def __init__(self, bin_path, vocab, lo, hi):
        self.data = np.memmap(bin_path, dtype=np.uint32, mode="r")
        self.lo = int(lo)
        self.hi = int(hi)
        self.vocab = vocab
        lut_path = f"/home/glenn/projects/neuro/data/vocab_map_{vocab}.npy"
        self.lut = np.load(lut_path)
        self.cur = self.lo

    def reset(self):
        self.cur = self.lo

    def next_chunk(self, T, device):
        """Return (x, y) for the next T+1 tokens as a single sequence (B=1).
        Wraps around if we exhaust the window. Returns None when window done."""
        if self.cur + T + 1 > self.hi:
            return None
        block = np.asarray(self.data[self.cur:self.cur + T + 1], dtype=np.int64)
        block = self.lut[block]
        self.cur += T
        x = torch.from_numpy(block[None, :-1]).to(device)
        y = torch.from_numpy(block[None, 1:]).to(device)
        return x, y


@torch.no_grad()
def build_brain(model, stream, T, total_tokens, device):
    """Stream `total_tokens` from `stream` through the model, return final state."""
    state = None
    consumed = 0
    stream.reset()
    t0 = time.time()
    while consumed < total_tokens:
        out = stream.next_chunk(T, device)
        if out is None:
            stream.reset()
            continue
        x, _ = out
        _, state = model(x, initial_state=state, return_final_state=True)
        consumed += T
        if consumed % 50000 < T:
            print(f"      exposed {consumed/1000:.0f}K tokens "
                  f"({consumed/max(1e-6, time.time()-t0):.0f} tok/s)")
    return state


def zero_brain(model, device):
    return {"layers": [
        {"M": torch.zeros(model.d_mem, model.d_mem, device=device),
         "mem": torch.zeros(model.n_neurons, device=device),
         "prev_spk": torch.zeros(model.n_neurons, device=device)}
        for _ in range(model.n_layers)
    ]}


def _normalize_brain(brain):
    """Strip leading batch dims so M is (d_mem, d_mem) and mem/prev_spk are (n_neurons,)."""
    out = {"layers": []}
    for lay in brain["layers"]:
        ll = {}
        for k, v in lay.items():
            target_dim = 2 if k == "M" else 1
            while v.dim() > target_dim:
                v = v.squeeze(0)
            ll[k] = v
        out["layers"].append(ll)
    return out


@torch.no_grad()
def measure_ppl(model, stream, T, brain, n_samples, device):
    """Compute ppl on `n_samples` fresh sequences from `stream`, with `brain` as init state."""
    nb = _normalize_brain(brain)
    total = 0.0
    used = 0
    stream.reset()
    while used < n_samples:
        out = stream.next_chunk(T, device)
        if out is None:
            break
        x, y = out
        # Broadcast brain to B=1 (or B=8 if we batch later)
        bb = {"layers": [
            {k: v.unsqueeze(0).contiguous() for k, v in lay.items()}
            for lay in nb["layers"]
        ]}
        out2 = model(x, initial_state=bb)
        total += F.cross_entropy(out2.reshape(-1, model.vocab),
                                  y.reshape(-1)).item()
        used += 1
    return math.exp(total / max(1, used))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exposure_tokens", type=int, default=200000,
                    help="tokens per brain build")
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--n_eval", type=int, default=40,
                    help="number of held-out sequences per (chunk, brain) cell")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device
    os.makedirs(BRAIN_DIR, exist_ok=True)

    m, c = build_from_ckpt(device)
    block = args.block_size

    # Split the file into four regions:
    #   A_train: 80%-85% (exposure for brain_A)
    #   A_test:  85%-87% (held-out chunk-A for measurement)
    #   B_train: 87%-92% (exposure for brain_B)
    #   B_test:  92%-94% (held-out chunk-B for measurement)
    data_len = len(np.memmap(BIN, dtype=np.uint32, mode="r"))
    A_train = (int(0.80 * data_len), int(0.85 * data_len))
    A_test  = (int(0.85 * data_len), int(0.87 * data_len))
    B_train = (int(0.87 * data_len), int(0.92 * data_len))
    B_test  = (int(0.92 * data_len), int(0.94 * data_len))
    print(f"data_len={data_len:,}  block={block}  exposure_tokens={args.exposure_tokens:,}")
    print(f"  A_train range: {A_train[0]:,}..{A_train[1]:,}")
    print(f"  A_test  range: {A_test[0]:,}..{A_test[1]:,}")
    print(f"  B_train range: {B_train[0]:,}..{B_train[1]:,}")
    print(f"  B_test  range: {B_test[0]:,}..{B_test[1]:,}\n")

    print("Building brain_A (exposure to chunk A)...")
    s = ChunkedStream(BIN, c["vocab"], *A_train)
    brain_A = build_brain(m, s, block, args.exposure_tokens, device)
    bp = os.path.join(BRAIN_DIR, "brain_A.brain")
    m.save_brain(brain_A, bp)
    print(f"  saved {bp}\n")

    print("Building brain_B (exposure to chunk B)...")
    s = ChunkedStream(BIN, c["vocab"], *B_train)
    brain_B = build_brain(m, s, block, args.exposure_tokens, device)
    bp = os.path.join(BRAIN_DIR, "brain_B.brain")
    m.save_brain(brain_B, bp)
    print(f"  saved {bp}\n")

    base_brain = zero_brain(m, device)

    print("Measuring perplexity on each test chunk with each brain:")
    results = []
    for chunk_name, chunk_range in [("A_test", A_test), ("B_test", B_test)]:
        s = ChunkedStream(BIN, c["vocab"], *chunk_range)
        row = {"chunk": chunk_name}
        for brain_name, brain in [("none", base_brain), ("brain_A", brain_A),
                                    ("brain_B", brain_B)]:
            ppl = measure_ppl(m, s, block, brain, args.n_eval, device)
            row[brain_name] = ppl
        results.append(row)

    # Pretty-print
    print(f"\n{'chunk':>10} | {'none':>9} | {'brain_A':>10} | {'brain_B':>10} | "
          f"{'A help':>8} | {'B help':>8}")
    print("-" * 72)
    for r in results:
        a_help = (r["none"] - r["brain_A"]) / r["none"] * 100
        b_help = (r["none"] - r["brain_B"]) / r["none"] * 100
        print(f"{r['chunk']:>10} | {r['none']:>9.2f} | {r['brain_A']:>10.2f} | "
              f"{r['brain_B']:>10.2f} | {a_help:>+7.2f}% | {b_help:>+7.2f}%")

    # Specificity metric
    a_help_on_A = (results[0]["none"] - results[0]["brain_A"]) / results[0]["none"] * 100
    a_help_on_B = (results[1]["none"] - results[1]["brain_A"]) / results[1]["none"] * 100
    b_help_on_A = (results[0]["none"] - results[0]["brain_B"]) / results[0]["none"] * 100
    b_help_on_B = (results[1]["none"] - results[1]["brain_B"]) / results[1]["none"] * 100
    print(f"\nSpecificity (positive = brain helps its OWN domain more):")
    print(f"  brain_A:  +{a_help_on_A-a_help_on_B:.2f}% (on A vs on B)")
    print(f"  brain_B:  +{b_help_on_B-b_help_on_A:.2f}% (on B vs on A)")
    if a_help_on_A - a_help_on_B > 0.5 and b_help_on_B - b_help_on_A > 0.5:
        verdict = "POSITIVE: brain captures local domain patterns"
    elif abs(a_help_on_A - a_help_on_B) < 0.3 and abs(b_help_on_B - b_help_on_A) < 0.3:
        verdict = "NEUTRAL: brain helps generically, not domain-specifically"
    else:
        verdict = "UNCLEAR or NEGATIVE"
    print(f"  verdict: {verdict}")


if __name__ == "__main__":
    main()
