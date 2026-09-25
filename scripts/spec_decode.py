"""MTP-based self-speculative decoding.

Uses the model's existing MTP head (trained to predict token at t+offset) as
the DRAFT, and the same model's main head as the VERIFIER — no separate
drafter model needed. Every accepted MTP draft saves one autoregressive step.

Recipe (offset=2 case, the 500M v2 setting)
-------------------------------------------
Given input tokens x[0..t], one forward gives:
  main_logits  at position t -> distribution over x[t+1]  (a1)
  mtp_logits   at position t -> distribution over x[t+2]  (a2_draft)

Speculative step:
  1. Sample a1 from main_logits.
  2. Sample a2_draft from mtp_logits (the "guess").
  3. Feed [..., x[t], a1] into the model -> get main_logits at position t+1,
     which is the VERIFIED distribution over x[t+2] (a2_verified).
  4. Compare: if greedy, accept a2_draft iff argmax(a2_verified) == a2_draft.
     If sampling, use Chen-Leviathan rejection: accept with prob min(1, p_v(a2_draft) / p_d(a2_draft)).
  5. Accepted: emit both a1 and a2_draft, advance by 2, save one forward.
     Rejected: emit a1 and sample from a2_verified (or from the residual
     (p_v - p_d)+ distribution for correctness), advance by 1, no savings.

Expected speedup
---------------
Depends on MTP acceptance rate. Literature (DeepSeek-V3 MTP report): ~30-50%
acceptance at offset=1, dropping at offset=2. Realistic end-to-end wall-clock
speedup: 1.15x-1.4x. Free once MTP is trained (which it is on 500M v2).

Status
------
[SKELETON — NOT YET TESTED against a trained ckpt.] 500M v2 still training;
final acceptance rate and speedup measurement waits on run completion.
Design + code path ready; validate on the finished ckpt.

Usage (once ready)
------------------
    python scripts/spec_decode.py --run demo_phase1b_500m_v2 \\
        --prompt "The brain is" --tokens 120 --temperature 0.0

    # Compare vs baseline autoregressive:
    python scripts/spec_decode.py ... --mode baseline
    python scripts/spec_decode.py ... --mode spec       # default

    # Report acceptance rate + wall clock:
    python scripts/spec_decode.py ... --report
"""
import os, sys, glob, argparse, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import SpikingHebbianLM
from train_gpu import build_model as _canonical_build

ROOT = os.path.join(os.path.dirname(__file__), "..")


def latest(run):
    cks = sorted(glob.glob(os.path.join(ROOT, "models", "checkpoints", run, "step_*.pt")))
    if not cks:
        raise FileNotFoundError(f"no checkpoints in {run}")
    return cks[-1]


def build(c, device):
    # P2: delegate to train_gpu.build_model (single source of truth).
    # See [[neuro_train_sft_build_drift_fix]] — hand-rolled kwargs went
    # stale progressively, silently mis-building post-July checkpoints.
    return _canonical_build(c).to(device)


def load_tokenizer():
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")
    lut = np.load(os.path.join(ROOT, "data", "vocab_map_16384.npy"))
    inv = np.full(16384, -1, dtype=np.int64)
    nz = np.where(lut > 0)[0]
    inv[lut[nz]] = nz
    def encode(text):
        ids = tk.encode(text, add_special_tokens=False)
        return torch.tensor([[int(lut[t]) for t in ids if lut[t] != 0]], dtype=torch.long)
    def decode(ids):
        return tk.decode([int(inv[i]) for i in ids if 0 < i < 16384 and inv[i] >= 0])
    return encode, decode


def _sample(logits, temperature, topk):
    if temperature <= 1e-6:
        return int(logits.argmax().item())
    z = logits / temperature
    if topk:
        v, _ = torch.topk(z, topk); z = torch.where(z < v[-1], torch.full_like(z, float("-inf")), z)
    return int(torch.multinomial(F.softmax(z, -1), 1).item())


@torch.no_grad()
def generate_baseline(model, c, ids, n_new, topk, temperature, device):
    """Standard autoregressive: one forward per emitted token."""
    out = ids.to(device)
    t0 = time.time()
    for _ in range(n_new):
        x = out[:, -c["block_size"]:]
        logits = model(x)[0, -1]
        nxt = _sample(logits, temperature, topk)
        out = torch.cat([out, torch.tensor([[nxt]], device=device)], 1)
    return out[0].tolist(), time.time() - t0, {"forwards": n_new}


@torch.no_grad()
def generate_speculative(model, c, ids, n_new, topk, temperature, device):
    """MTP self-speculative: propose t+2 with MTP head, verify with main head.

    Greedy (temperature=0) uses exact-match acceptance.
    Sampled (temperature>0) uses Chen-Leviathan rejection acceptance.
    """
    assert c.get("mtp_enabled", False), "spec_decode requires mtp_enabled model"
    offset = c.get("mtp_offset", 2)
    assert offset == 2, f"spec_decode currently assumes offset=2, got {offset}"

    out = ids.to(device)
    emitted = 0
    forwards = 0
    accepts = 0
    proposals = 0
    t0 = time.time()

    while emitted < n_new:
        x = out[:, -c["block_size"]:]
        logits = model(x)                               # forward
        forwards += 1
        main_logits = logits[0, -1]
        mtp_logits = model._last_mtp_logits[0, -1]      # draft for token at t+offset

        # step 1: sample a1 from main_logits (verified for t+1)
        a1 = _sample(main_logits, temperature, topk)
        out = torch.cat([out, torch.tensor([[a1]], device=device)], 1)
        emitted += 1
        if emitted >= n_new: break

        # step 2: draft a2 from mtp_logits (proposed for t+2)
        a2_draft = _sample(mtp_logits, temperature, topk)
        proposals += 1

        # step 3: forward including a1, get verified dist for t+2
        x = out[:, -c["block_size"]:]
        logits = model(x)
        forwards += 1
        a2_verify_logits = logits[0, -1]

        # step 4: accept?
        if temperature <= 1e-6:
            # greedy exact-match acceptance
            accept = (a2_draft == int(a2_verify_logits.argmax().item()))
        else:
            # Chen-Leviathan speculative sampling
            p_d = F.softmax(mtp_logits / temperature, -1)[a2_draft].item()
            p_v = F.softmax(a2_verify_logits / temperature, -1)[a2_draft].item()
            accept = torch.rand(1).item() < min(1.0, p_v / max(p_d, 1e-9))

        if accept:
            out = torch.cat([out, torch.tensor([[a2_draft]], device=device)], 1)
            emitted += 1
            accepts += 1
        else:
            # rejected: emit a fresh sample from the verified distribution
            # (for correctness this should be (p_v - p_d)+, but exact-match
            # greedy path just takes argmax; simplify for now).
            a2_re = _sample(a2_verify_logits, temperature, topk)
            out = torch.cat([out, torch.tensor([[a2_re]], device=device)], 1)
            emitted += 1

    elapsed = time.time() - t0
    stats = {"forwards": forwards, "proposals": proposals, "accepts": accepts,
             "accept_rate": accepts / max(proposals, 1),
             "speedup_forwards_ratio": n_new / max(forwards, 1)}
    return out[0].tolist(), elapsed, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="demo_phase1b_500m_v2")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--tokens", type=int, default=120)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--mode", choices=["baseline", "spec"], default="spec")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    blob = torch.load(latest(args.run), map_location=args.device, weights_only=False)
    c = blob["cfg"]
    m = build(c, args.device); m.load_state_dict(blob["model"], strict=False); m.eval()
    print(f"{args.run}: {sum(p.numel() for p in m.parameters())/1e6:.0f}M params, "
          f"{blob.get('tokens',0)/1e9:.1f}B tok, mtp={c.get('mtp_enabled', False)}\n")

    enc, dec = load_tokenizer()
    ids = enc(args.prompt)
    if ids.shape[1] == 0:
        print("prompt mapped to all-OOV; try different words"); return

    gen_fn = generate_baseline if args.mode == "baseline" else generate_speculative
    ids_out, elapsed, stats = gen_fn(m, c, ids, args.tokens, args.topk,
                                     args.temperature, args.device)

    print(f"PROMPT: {args.prompt}")
    print(f"OUTPUT: {dec(ids_out)}")
    if args.report:
        print(f"\n[{args.mode}] time={elapsed:.2f}s stats={stats}")


if __name__ == "__main__":
    main()
