"""Muon + tie_weights sweep at Run A scale (~9M params), short budgets.

Trains a fresh model for each (optimizer, muon_lr, tie_weights) combination
on a short token budget, records final val_ppl + spike_rate, and prints a
comparison table.

Designed so a 6-config sweep finishes in well under an hour on a 5080.
"""
import os, sys, time, math, json
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import SpikingHebbianLM
from data_stream import StreamText
from muon import build_muon_adamw

BIN = "/home/glenn/projects/bdh/data/tokenized/train.bin"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Architecture matches phase25_a.yaml.
VOCAB = 16384
D, N, DM = 256, 512, 256
BLOCK = 128
BATCH = 128
GRAD_ACCUM = 1
LR = 1.5e-3
MAX_TOK = 30_000_000     # 30M tokens per config; ~9-10 min each at ~55k tok/s
EVAL_ITERS = 30
EVAL_BATCH = 64


def build_model(tie):
    return SpikingHebbianLM(VOCAB, d=D, n_neurons=N, d_mem=DM,
                            recurrent=True, rec_density=0.02,
                            compile_safe=True, tie_weights=tie).to(DEVICE)


def build_opts(model, optimizer, muon_lr, lr=LR):
    if optimizer == "adamw":
        return [torch.optim.AdamW(model.parameters(), lr=lr, fused=True)]
    muon, adamw, n_mat, n_other = build_muon_adamw(
        model, muon_lr=muon_lr, adamw_lr=lr, adamw_fused=True)
    return [o for o in (muon, adamw) if o is not None]


@torch.no_grad()
def evaluate(model, data, iters=EVAL_ITERS):
    model.eval()
    losses = []
    sr_sum = 0.0
    for _ in range(iters):
        x, y = data.batch(EVAL_BATCH, BLOCK, "val", DEVICE)
        out, sr = model(x, return_stats=True)
        losses.append(F.cross_entropy(out.reshape(-1, VOCAB), y.reshape(-1)).item())
        sr_sum += sr.item()
    return math.exp(sum(losses) / len(losses)), sr_sum / iters


def run_one(label, optimizer, muon_lr, tie, data):
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    model = build_model(tie)
    nparams = sum(p.numel() for p in model.parameters())
    opts = build_opts(model, optimizer, muon_lr)

    per_step = BATCH * GRAD_ACCUM * BLOCK
    tokens = 0
    step = 0
    t0 = time.time()
    last_loss = float("nan")
    nan_seen = False
    while tokens < MAX_TOK:
        model.train()
        for o in opts: o.zero_grad(set_to_none=True)
        for _ in range(GRAD_ACCUM):
            x, y = data.batch(BATCH, BLOCK, "train", DEVICE)
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                out = model(x)
                loss = F.cross_entropy(out.reshape(-1, VOCAB), y.reshape(-1)) / GRAD_ACCUM
            loss.backward()
        for o in opts: o.step()
        tokens += per_step
        step += 1
        last_loss = loss.item() * GRAD_ACCUM
        if math.isnan(last_loss):
            nan_seen = True
            break

    elapsed = time.time() - t0
    ppl, spike = evaluate(model, data)
    return {
        "label": label,
        "optimizer": optimizer,
        "muon_lr": muon_lr,
        "tie_weights": tie,
        "params": nparams,
        "tokens": tokens,
        "elapsed_s": elapsed,
        "tok_per_s": tokens / elapsed,
        "final_loss": last_loss,
        "val_ppl": ppl,
        "spike_rate": spike,
        "nan": nan_seen,
    }


CONFIGS = [
    # (label, optimizer, muon_lr, tie_weights)
    ("AdamW",                "adamw", None,   False),
    ("AdamW + tie",          "adamw", None,   True),
    ("Muon 0.02 + tie",      "muon",  0.02,   True),
    ("Muon 0.005 + tie",     "muon",  0.005,  True),
    ("Muon 0.002 + tie",     "muon",  0.002,  True),
    ("Muon 0.005 (no tie)",  "muon",  0.005,  False),
]


def main():
    print(f"device: {DEVICE}, vocab: {VOCAB}, max_tokens: {MAX_TOK/1e6:.0f}M, "
          f"steps/M tok: {1e6 / (BATCH*BLOCK):.0f}")
    print(f"shared arch: d={D}, n={N}, d_mem={DM}, block={BLOCK}, batch={BATCH}")
    print()
    data = StreamText(BIN, vocab=VOCAB)
    results = []
    for label, opt, mlr, tie in CONFIGS:
        print(f"--- running: {label} ---")
        try:
            r = run_one(label, opt, mlr, tie, data)
        except Exception as e:
            print(f"  ERROR: {e.__class__.__name__}: {e}")
            results.append({"label": label, "error": str(e)})
            continue
        results.append(r)
        nan_tag = "  NaN" if r["nan"] else ""
        print(f"  params={r['params']/1e6:.2f}M  ppl={r['val_ppl']:.1f}  "
              f"spike={r['spike_rate']:.3f}  loss={r['final_loss']:.3f}  "
              f"{r['tok_per_s']:,.0f} tok/s  ({r['elapsed_s']:.0f}s){nan_tag}")
        print()

    print("=" * 78)
    print(f"{'config':<22} | {'params':>7} | {'ppl':>8} | {'spike':>6} | "
          f"{'tok/s':>8} | {'sec':>4}")
    print("-" * 78)
    for r in results:
        if "error" in r:
            print(f"{r['label']:<22} | ERROR {r['error']}"); continue
        nan = " *NaN*" if r["nan"] else ""
        print(f"{r['label']:<22} | {r['params']/1e6:>6.2f}M | {r['val_ppl']:>8.1f}{nan} | "
              f"{r['spike_rate']:>6.3f} | {r['tok_per_s']:>8,.0f} | {r['elapsed_s']:>4.0f}")
    out_path = os.path.join(os.path.dirname(__file__), "..", "temp", "sweep_muon_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nfull results -> {out_path}")


if __name__ == "__main__":
    main()
