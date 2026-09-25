"""Bake a domain brain onto a trained checkpoint.

Given a checkpoint dir and a corpus .bin file, feeds the corpus through the
model in chunks with state carried across chunks, then saves the final
Hebbian state as a .brain file. Loadable via SpikingHebbianLM.load_brain(path)
and passable as initial_state= to model.forward().

Usage:
  python scripts/bake_brain.py \\
      --run sft_500m_tulu3 \\
      --corpus data/brain_corpora/knowledge_500m.bin \\
      --out models/brains/sft_500m_tulu3/knowledge.brain \\
      --tokens 10000000   # optional; default = whole corpus

Brains are checkpoint-specific: baking on model A does NOT transfer to
model B. Rebake after any weight update (new SFT run, DPO, etc).
"""
import os, sys, glob, argparse, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train_gpu import build_model  # canonical builder -- do not re-derive

ROOT = os.path.join(os.path.dirname(__file__), "..")


def latest_ckpt(run):
    cks = sorted(glob.glob(os.path.join(ROOT, "models", "checkpoints", run, "step_*.pt")))
    if not cks:
        raise FileNotFoundError(f"no checkpoints in {run}")
    return cks[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="checkpoint dir name under models/checkpoints/")
    ap.add_argument("--corpus", required=True, help="path to .bin token file (uint16)")
    ap.add_argument("--out", required=True, help="output .brain path")
    ap.add_argument("--tokens", type=int, default=None,
                    help="how many tokens to feed (default: whole corpus)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"[loading] {args.run}", flush=True)
    ckpt_path = latest_ckpt(args.run)
    blob = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    c = blob["cfg"]
    m = build_model(c).to(args.device)
    m.load_state_dict(blob["model"], strict=False)
    m.eval()
    n_params = sum(p.numel() for p in m.parameters())
    block_size = c["block_size"]
    print(f"[loaded ] {ckpt_path.split('/')[-1]}  {n_params/1e6:.1f}M params  "
          f"block_size={block_size}", flush=True)

    # Corpus stores raw Llama-3 128k IDs as uint32 (same on-disk format as
    # /home/glenn/projects/bdh/data/tokenized/train.bin; verified via
    # temp/diag on data/brain_corpora/*.bin 2026-07-31). The model was
    # trained on LUT-projected IDs in c["vocab"] range, so we project here
    # via the standard vocab_map_{vocab}.npy — same LUT the training data
    # loader (src/data_stream.py) uses.
    tokens_raw = np.memmap(args.corpus, dtype=np.uint32, mode="r")
    total_in_corpus = len(tokens_raw)
    n_use = args.tokens if args.tokens else total_in_corpus
    n_use = min(n_use, total_in_corpus)
    lut_path = os.path.join(ROOT, "data", f"vocab_map_{c['vocab']}.npy")
    lut = np.load(lut_path).astype(np.int64)
    max_raw = int(tokens_raw[:100000].max())
    print(f"[corpus ] {args.corpus} -- {total_in_corpus/1e6:.1f}M tokens (raw Llama-3, "
          f"sample_max={max_raw}), using {n_use/1e6:.1f}M", flush=True)
    print(f"[lut    ] {lut_path} -- projecting to {c['vocab']} vocab", flush=True)

    state = None
    t0 = time.time()
    n_chunks = (n_use + block_size - 1) // block_size
    reported = 0
    with torch.no_grad():
        for i in range(n_chunks):
            s = i * block_size
            e = min(s + block_size, n_use)
            if e - s < 8:
                break
            ids_raw = tokens_raw[s:e].astype(np.int64)
            ids_proj = lut[ids_raw]                 # LUT project raw -> vocab
            x = torch.from_numpy(ids_proj).unsqueeze(0).to(args.device)
            _, state = m(x, initial_state=state, return_final_state=True)
            pct = 100.0 * (i + 1) / n_chunks
            if pct >= reported + 10:
                dt = time.time() - t0
                print(f"  {pct:5.1f}%  ({(i+1)*block_size/1e6:.1f}M tok  {dt:.0f}s  "
                      f"{(i+1)*block_size/max(dt,1e-3):.0f} tok/s)", flush=True)
                reported += 10

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    m.save_brain(state, args.out)
    dt = time.time() - t0
    print(f"[saved  ] {args.out}  ({n_use/1e6:.1f}M tok in {dt:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
