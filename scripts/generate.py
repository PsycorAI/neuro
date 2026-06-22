"""Prompt a trained PsycorNeuro checkpoint and read its continuation.

  python scripts/generate.py --run phase3_350M --prompt "The brain is"
  python scripts/generate.py --run phase3_350M --prompt "Once upon a time" --tokens 120 --temperature 0.8

Decodes via the Llama-3 tokenizer + the training vocab LUT (data/vocab_map_16384.npy).
Needs HF access to a Llama-3 tokenizer (NousResearch/Meta-Llama-3-8B, ungated).
"""
import os, sys, glob, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import SpikingHebbianLM

ROOT = os.path.join(os.path.dirname(__file__), "..")
LUT_PATH = os.path.join(ROOT, "data", "vocab_map_16384.npy")


def latest(run):
    cks = sorted(glob.glob(os.path.join(ROOT, "models", "checkpoints", run, "step_*.pt")))
    if not cks:
        raise FileNotFoundError(f"no checkpoints in {run}")
    return cks[-1]


def build(c, device):
    return SpikingHebbianLM(c["vocab"], d=c["d"], n_neurons=c["n_neurons"], d_mem=c["d_mem"],
                            recurrent=c.get("recurrent", False), rec_density=c.get("rec_density", 0.05),
                            compile_safe=c.get("compile", False), tie_weights=c.get("tie_weights", False),
                            n_layers=c.get("n_layers", 1), use_fpt=c.get("use_fpt", False),
                            fpt_K=c.get("fpt_K", 10), lam=c.get("lam", 0.98),
                            learnable_decay=c.get("learnable_decay", False),
                            write_gate=c.get("write_gate", False),
                            delta_rule=c.get("delta_rule", False)).to(device)


def load_tokenizer():
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")
    lut = np.load(LUT_PATH)
    inv = np.full(16384, -1, dtype=np.int64)        # student id -> original Llama-3 id
    nz = np.where(lut > 0)[0]
    inv[lut[nz]] = nz
    def encode(text):
        ids = tk.encode(text, add_special_tokens=False)
        return torch.tensor([[int(lut[t]) for t in ids if lut[t] != 0]], dtype=torch.long)
    def decode(ids):
        return tk.decode([int(inv[i]) for i in ids if 0 < i < 16384 and inv[i] >= 0])
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

    enc, dec = load_tokenizer()
    ids = enc(args.prompt)
    if ids.shape[1] == 0:
        print("prompt mapped to all-OOV (rare tokens); try different words"); return
    gen = generate(m, c, ids, args.tokens, args.topk, args.temperature, args.device)
    print(f"PROMPT: {args.prompt}")
    print(f"OUTPUT: {dec(gen)}")


if __name__ == "__main__":
    main()
