"""Qualitative sampling from the trained 20M models.

Generates short continuations from both the spiking and transformer checkpoints
on a few prompts, decoded back to text via the Llama-3 tokenizer + the vocab
LUT (`data/vocab_map_16384.npy`) used for training.

Usage:
  python scripts/sample_phase25.py
  python scripts/sample_phase25.py --tokens 80 --topk 5
"""
import os, sys, glob, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import SpikingHebbianLM
from baseline import TinyTransformer

ROOT = os.path.join(os.path.dirname(__file__), "..")
SPK_DIR = os.path.join(ROOT, "models", "checkpoints", "phase25_b")
TF_DIR  = os.path.join(ROOT, "models", "checkpoints", "phase25_b_tf")
LUT_PATH = os.path.join(ROOT, "data", "vocab_map_16384.npy")
PROMPTS = [
    "The quick brown fox",
    "Once upon a time,",
    "In the beginning was",
    "The capital of France is",
]


def latest_ckpt(d):
    cks = sorted(glob.glob(os.path.join(d, "step_*.pt")))
    if not cks: raise FileNotFoundError(f"no checkpoints in {d}")
    return cks[-1]


def build_from_cfg(c):
    if c["arch"] == "spiking":
        return SpikingHebbianLM(c["vocab"], d=c["d"], n_neurons=c["n_neurons"],
                                d_mem=c["d_mem"],
                                recurrent=c.get("recurrent", False),
                                rec_density=c.get("rec_density", 0.05),
                                compile_safe=c.get("compile", False),
                                tie_weights=c.get("tie_weights", False),
                                n_layers=c.get("n_layers", 1),
                                use_fpt=c.get("use_fpt", False),
                                fpt_K=c.get("fpt_K", 10),
                                lam=c.get("lam", 0.98),
                                learnable_decay=c.get("learnable_decay", False),
                                write_gate=c.get("write_gate", False),
                                delta_rule=c.get("delta_rule", False),
                                beta_floor=c.get("beta_floor", 0.0),
                                decay_gate=c.get("decay_gate", False))
    return TinyTransformer(c["vocab"], d=c["d"], n_head=c["n_head"],
                           n_layer=c["n_layer"], max_T=c["block_size"])


def load(ckpt_path, device):
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = blob["cfg"]
    m = build_from_cfg(c).to(device)
    m.load_state_dict(blob["model"])
    m.eval()
    return m, c


def load_tokenizer():
    """Return (encode_fn, decode_fn) for Llama-3 → vocab-16384 space.
    Returns None if the tokenizer can't be loaded (no internet / no HF token)."""
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")
    except Exception as e:
        print(f"  [warn] tokenizer unavailable ({e.__class__.__name__}: {e})")
        return None, None
    lut = np.load(LUT_PATH)                      # lut[orig_token] = new_token (0=unk)
    inv = np.full(16384, -1, dtype=np.int64)
    nz = np.where(lut > 0)[0]
    inv[lut[nz]] = nz
    def encode(text):
        ids = tk.encode(text, add_special_tokens=False)
        mapped = [int(lut[t]) for t in ids if lut[t] != 0]   # drop OOV
        return torch.tensor([mapped], dtype=torch.long)
    def decode(ids):
        orig = [int(inv[i]) for i in ids if 0 < i < 16384 and inv[i] >= 0]
        return tk.decode(orig)
    return encode, decode


@torch.no_grad()
def generate(model, c, prompt_ids, n_new, topk, temperature, device):
    out = prompt_ids.to(device)
    for _ in range(n_new):
        # respect block_size by sliding window
        x = out[:, -c["block_size"]:]
        logits = model(x)[0, -1] / max(temperature, 1e-6)
        if topk and topk < logits.size(0):
            v, _ = torch.topk(logits, topk)
            logits[logits < v[-1]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        out = torch.cat([out, next_id.view(1, 1)], dim=1)
    return out[0].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--tokens", type=int, default=60, help="tokens to generate per prompt")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()
    device = args.device
    torch.manual_seed(0)

    spk_ck, tf_ck = latest_ckpt(SPK_DIR), latest_ckpt(TF_DIR)
    spk, sc = load(spk_ck, device)
    tf,  tc = load(tf_ck,  device)

    encode, decode = load_tokenizer()
    if encode is None:
        print("  Falling back to random token-ID prompts (no detokenizable output).")

    for i, prompt in enumerate(PROMPTS):
        print("=" * 72)
        print(f"Prompt {i+1}: \"{prompt}\"")
        print("=" * 72)
        if encode is None:
            ids = torch.randint(1, sc["vocab"], (1, 4))
            print(f"  (prompt id-only: {ids[0].tolist()})")
        else:
            ids = encode(prompt)
            if ids.shape[1] == 0:
                print("  prompt mapped to all-OOV; skipping")
                continue

        for name, model, c in (("SPIKING ", spk, sc), ("TRANSFORM", tf, tc)):
            gen_ids = generate(model, c, ids, args.tokens, args.topk, args.temperature, device)
            if decode:
                text = decode(gen_ids)
                print(f"  [{name}] {text}")
            else:
                print(f"  [{name}] {gen_ids}")


if __name__ == "__main__":
    main()
