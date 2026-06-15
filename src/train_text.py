"""Phase-1 language sanity + Goal-2 energy comparison.

Trains the spiking Hebbian LM and a matched tiny transformer on a slice of the
real Llama-3-tokenized corpus, and reports:
  - G1: both beat a bigram baseline (perplexity)
  - G6: per-token INFERENCE energy, spiking (AC-heavy) vs transformer (MAC), by seq_len
"""
import os, sys, time, math
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from model import SpikingHebbianLM
from baseline import TinyTransformer
from data import TextData
import energy

BIN = "/home/glenn/projects/bdh/data/tokenized/train.bin"
DEVICE = "cpu"
VOCAB, T, B, STEPS = 4096, 48, 32, 800


def val_ce(model, data, iters=20):
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for _ in range(iters):
            x, y = data.batch(B, T, split="val", device=DEVICE)
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            tot += F.cross_entropy(logits.reshape(-1, data.vocab), y.reshape(-1)).item()
    return tot / iters


def train(model, data, tag):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.time()
    for step in range(1, STEPS + 1):
        model.train()
        x, y = data.batch(B, T, split="train", device=DEVICE)
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        loss = F.cross_entropy(logits.reshape(-1, data.vocab), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"  trained {tag} ({sum(p.numel() for p in model.parameters()):,} params) in {time.time()-t0:.0f}s")


def main():
    torch.manual_seed(0)
    print("loading text slice from bdh train.bin ...")
    data = TextData(BIN, n_tokens=1_500_000, vocab=VOCAB)
    bigram = data.bigram_ce()

    spk = SpikingHebbianLM(VOCAB, d=128, n_neurons=256, d_mem=128).to(DEVICE)
    tfm = TinyTransformer(VOCAB, d=128, n_head=2, n_layer=2, max_T=T).to(DEVICE)
    train(spk, data, "spiking")
    train(tfm, data, "transformer")

    ce_spk, ce_tfm = val_ce(spk, data), val_ce(tfm, data)
    # measured spike rate for the energy estimate
    x, y = data.batch(B, T, split="val", device=DEVICE)
    with torch.no_grad():
        _, sr = spk(x, return_stats=True)
    sr = sr.item()

    print("\n=== G1: language vs bigram (val perplexity, lower=better) ===")
    print(f"  bigram      : ppl {math.exp(bigram):8.2f}")
    print(f"  transformer : ppl {math.exp(ce_tfm):8.2f}")
    print(f"  spiking-BDH : ppl {math.exp(ce_spk):8.2f}")
    g1 = ce_spk < bigram
    print(f"  G1 spiking beats bigram: {'PASS' if g1 else 'FAIL'}")

    print("\n=== G6: inference energy per token (45nm model) ===")
    energy.compare(spk, sr, baseline_d=128, baseline_layers=2,
                   vocab=VOCAB, seq_lens=[48, 512, 4096])


if __name__ == "__main__":
    main()
