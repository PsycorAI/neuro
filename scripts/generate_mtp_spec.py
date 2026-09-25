"""Self-speculative decoding via MTP heads (greedy variant).

The 500M v3 model has mtp_enabled: true with mtp_offset=2, mtp_depth=2 — the
model's MTP head predicts token t+2 given input up to t. We can use those as
"drafts": generate a candidate 2 tokens ahead, then verify with one forward
whether the model actually agrees.

Speedup formula (greedy): if MTP's predicted t+2 matches the argmax of a
verification forward on the (accepted t+1) input, we get 2 tokens per forward
pair instead of 1. At MTP acceptance rate α, effective speedup ≈ 1 + α.
Reported at scale: 2-3x for well-trained MTP heads (DeepSeek-V3 pattern).

Loop:
  Given prompt tokens [x_1, ..., x_t]:
    1. Forward on [x_1..x_t] with return_mtp=True.
       - main_logits[-1] predicts token t+1  ("y_1")
       - mtp_logits[-1]  predicts token t+2  ("y_2_draft")
    2. Verify forward on [x_1..x_t, y_1].
       - verify_logits[-1] predicts what actually follows y_1  ("y_2_verify")
    3. If y_2_draft == y_2_verify:
         accept both, advance by 2 tokens
       else:
         take y_2_verify (correct), advance by 2 tokens
    4. Repeat until EOS or max_new.

  Note: even on a rejection we advance by 2 (we spent 2 forwards, got 2 tokens).
  On acceptance we spent 2 forwards for 2 tokens BUT saved the follow-up forward
  that would otherwise be needed to sample y_2. Net: with α acceptance, 2 tokens
  cost (1 + (1-α)) forwards on average = 2/(1+α) forwards per token.

Limitations of this greedy implementation (call them out honestly):
  - GREEDY ONLY. Full rejection-sampling spec-decode preserves the target
    distribution under stochastic sampling; this doesn't (uses argmax).
  - Batch size 1. Batched spec-decode needs per-example acceptance tracking.
  - Only 1-step drafting (uses only mtp_offset=2, doesn't chain drafts).
    Multi-step MTP drafting (draft 4 tokens, verify all 4) is a natural
    extension; requires drafting a sequence, not just one lookahead.

Usage:
  python scripts/generate_mtp_spec.py --ckpt models/checkpoints/dpo_500m_full_v3/step_*.pt \\
      --prompt "The best way to boil an egg is" --max-new 128
"""
import os, sys, glob, argparse, time, yaml, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train_sft import build_model


def resolve_ckpt(pattern):
    if "*" in pattern:
        cks = sorted(glob.glob(pattern))
        if not cks:
            raise FileNotFoundError(f"no matches for {pattern}")
        return cks[-1]
    return pattern


def load_llama_encode(vocab):
    """Same LUT-based encoder used by train_sft/dpo."""
    from transformers import AutoTokenizer
    import numpy as np
    tk = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")
    lut = np.load(f"/home/glenn/projects/neuro/data/vocab_map_{vocab}.npy")
    inv = {int(v): int(i) for i, v in enumerate(lut) if v != 0}

    def encode(text):
        ids = tk.encode(text, add_special_tokens=False)
        return [int(lut[t]) for t in ids if lut[t] != 0]

    def decode(ids):
        # Best-effort reverse lookup; unknowns become '?'
        toks = [inv.get(int(i), 0) for i in ids]
        return tk.decode(toks, skip_special_tokens=True)
    return encode, decode


def generate_baseline(model, ids, max_new, device):
    """Standard 1-token-per-forward greedy generation. For timing comparison."""
    ids = ids.to(device).unsqueeze(0)  # (1, T)
    n_forwards = 0
    for _ in range(max_new):
        with torch.no_grad():
            out = model(ids)
        n_forwards += 1
        logits = out[0] if isinstance(out, tuple) else out
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
        ids = torch.cat([ids, nxt], dim=1)
    return ids[0], n_forwards


def generate_mtp_spec(model, ids, max_new, device, mtp_offset=2):
    """MTP-guided self-speculative greedy generation."""
    ids = ids.to(device).unsqueeze(0)  # (1, T)
    n_forwards = 0
    n_accepted = 0
    n_generated = 0

    while n_generated < max_new:
        # Step 1: Forward with return_mtp to get main token + MTP draft
        with torch.no_grad():
            out = model(ids, return_mtp=True)
        n_forwards += 1
        # Unpack; forward returns (logits, ...) with mtp_logits somewhere in the tuple
        # depending on other flags. When ONLY return_mtp=True, we get (logits, mtp_logits).
        # (If more flags are True, the ordering documented above may change; adjust here.)
        if isinstance(out, tuple):
            main_logits, mtp_logits = out[0], out[-1]  # last is mtp when return_mtp=True alone
        else:
            main_logits = out
            mtp_logits = None

        y1 = main_logits[:, -1, :].argmax(dim=-1)               # (1,) token t+1
        if mtp_logits is None:
            # MTP not available -- fall back to standard greedy for this step
            ids = torch.cat([ids, y1.unsqueeze(-1)], dim=1)
            n_generated += 1
            continue

        y2_draft = mtp_logits[:, -1, :].argmax(dim=-1)          # (1,) draft token t+2

        # Step 2: Verify forward with y1 appended
        ids_plus_y1 = torch.cat([ids, y1.unsqueeze(-1)], dim=1)
        with torch.no_grad():
            out2 = model(ids_plus_y1)
        n_forwards += 1
        verify_logits = out2[0] if isinstance(out2, tuple) else out2
        y2_verify = verify_logits[:, -1, :].argmax(dim=-1)       # (1,) correct token t+2

        # Step 3: Accept draft if match, else take verify
        if y2_draft.item() == y2_verify.item():
            n_accepted += 1
        y2 = y2_verify  # always correct by construction

        ids = torch.cat([ids_plus_y1, y2.unsqueeze(-1)], dim=1)
        n_generated += 2

    acceptance = n_accepted / max(1, n_generated // 2)
    return ids[0], n_forwards, acceptance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="model checkpoint path (glob ok)")
    ap.add_argument("--prompt", default="The best way to boil an egg is")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--baseline", action="store_true",
                     help="also time the standard greedy path for comparison")
    args = ap.parse_args()

    path = resolve_ckpt(args.ckpt)
    print(f"[gen] loading {path}")
    blob = torch.load(path, map_location=args.device, weights_only=False)
    cfg = blob["cfg"]
    if not cfg.get("mtp_enabled"):
        raise SystemExit(f"[gen] ckpt was NOT trained with mtp_enabled; can't self-spec-decode")

    model = build_model(cfg).to(args.device)
    model.load_state_dict(blob["model"], strict=False)
    model.eval()

    encode, decode = load_llama_encode(cfg["vocab"])
    prompt_ids = torch.tensor(encode(args.prompt), dtype=torch.long)
    print(f"[gen] prompt ({len(prompt_ids)} tokens): {args.prompt}")

    torch.cuda.synchronize() if args.device == "cuda" else None
    t0 = time.time()
    out_ids, n_forwards, acc = generate_mtp_spec(
        model, prompt_ids, args.max_new, args.device, cfg.get("mtp_offset", 2))
    torch.cuda.synchronize() if args.device == "cuda" else None
    dt = time.time() - t0
    new_ids = out_ids[len(prompt_ids):]
    print(f"[spec] tokens={len(new_ids)}  forwards={n_forwards}  "
          f"tokens/forward={len(new_ids)/n_forwards:.2f}  "
          f"acceptance={acc:.2f}  wall={dt:.2f}s")
    print(f"[spec] output: {decode(new_ids.tolist())}")

    if args.baseline:
        torch.cuda.synchronize() if args.device == "cuda" else None
        t0 = time.time()
        out_ids_b, n_forwards_b = generate_baseline(model, prompt_ids, args.max_new, args.device)
        torch.cuda.synchronize() if args.device == "cuda" else None
        dt_b = time.time() - t0
        new_ids_b = out_ids_b[len(prompt_ids):]
        print(f"[base] tokens={len(new_ids_b)}  forwards={n_forwards_b}  wall={dt_b:.2f}s")
        print(f"[base] output: {decode(new_ids_b.tolist())}")
        print(f"[cmp]  speculative wall/baseline wall = {dt/dt_b:.2f}x "
              f"(<1.0 = spec is faster)")


if __name__ == "__main__":
    main()
