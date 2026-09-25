"""Prompt a trained PsycorNeuro checkpoint and read its continuation.

  python scripts/generate.py --run phase3_350M --prompt "The brain is"
  python scripts/generate.py --run phase3_350M --prompt "Once upon a time" --tokens 120 --temperature 0.8
  python scripts/generate.py --run sparsity_refine_500m --prompt "The brain is" --tokens 120

Decodes via the checkpoint's OWN tokenizer + vocab LUT, picked automatically
from cfg["vocab"]: 16384 -> Llama-3 tokenizer + data/vocab_map_16384.npy (the
30M/90M/350M screen recipe); 32064 -> Phi-3.5 tokenizer + data/vocab_map_32064.npy
(the 500M v2 recipe, KD'd from microsoft/Phi-3.5-mini-instruct). Needs HF access
to the matching tokenizer (both ungated).
"""
import os, sys, glob, argparse
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


def load_tokenizer(vocab):
    from transformers import AutoTokenizer
    if vocab == 32064:
        tk_name, lut_path = "microsoft/Phi-3.5-mini-instruct", os.path.join(ROOT, "data", "vocab_map_32064.npy")
    else:
        tk_name, lut_path = "NousResearch/Meta-Llama-3-8B", os.path.join(ROOT, "data", f"vocab_map_{vocab}.npy")
    tk = AutoTokenizer.from_pretrained(tk_name)
    lut = np.load(lut_path)
    inv = np.full(vocab, -1, dtype=np.int64)        # student id -> original tokenizer id
    nz = np.where(lut > 0)[0]
    inv[lut[nz]] = nz
    def encode(text):
        # 2026-08-07 fix: keep id 0 (unknown/rare) in the sequence.
        # Training-time data_stream.py:37 KEEPS id 0 in the token stream
        # (~4% of training tokens map to it — Llama-3 tokens outside the
        # top-32k student vocab). Filtering id 0 here made inference
        # sequences ~4% shorter than training, changing the distribution
        # the model was trained on. Preserving id 0 matches training.
        ids = tk.encode(text, add_special_tokens=False)
        return torch.tensor([[int(lut[t]) for t in ids]], dtype=torch.long)
    def decode(ids):
        # 2026-08-07 fix: emit visible <unk> for id 0 / unmapped ids
        # instead of silent drop. Prevents the "the of the" / "of theity"
        # output patterns where content words vanished without trace.
        parts = []
        for i in ids:
            i = int(i)
            if 0 < i < vocab and inv[i] >= 0:
                parts.append(tk.decode([int(inv[i])]))
            else:
                parts.append("<unk>")
        return "".join(parts)
    return encode, decode


@torch.no_grad()
def generate(model, c, ids, n_new, topk, temperature, device):
    out = ids.to(device)
    for _ in range(n_new):
        x = out[:, -c["block_size"]:]
        logits = model(x)[0, -1] / max(temperature, 1e-6)
        if topk:
            v, _ = torch.topk(logits, topk); logits[logits < v[-1]] = float("-inf")
        nxt = torch.multinomial(F.softmax(logits, -1), 1)
        out = torch.cat([out, nxt.view(1, 1)], 1)
    return out[0].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="phase3_350M")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--tokens", type=int, default=80)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    blob = torch.load(latest(args.run), map_location=args.device, weights_only=False)
    c = blob["cfg"]
    m = build(c, args.device); m.load_state_dict(blob["model"], strict=False); m.eval()
    print(f"{args.run}: {sum(p.numel() for p in m.parameters())/1e6:.0f}M params, "
          f"{blob.get('tokens',0)/1e9:.1f}B tok\n")

    enc, dec = load_tokenizer(c["vocab"])
    ids = enc(args.prompt)
    if ids.shape[1] == 0:
        print("prompt mapped to all-OOV (rare tokens); try different words"); return
    gen = generate(m, c, ids, args.tokens, args.topk, args.temperature, args.device)
    print(f"PROMPT: {args.prompt}")
    print(f"OUTPUT: {dec(gen)}")


if __name__ == "__main__":
    main()
