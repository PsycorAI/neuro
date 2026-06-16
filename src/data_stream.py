"""Streaming text over the full Llama-3-tokenized corpus (bdh's train.bin).

memmaps the whole file (no in-RAM cap) and remaps tokens to the top-`vocab` most
frequent ids via a LUT built once from a sample and cached to disk.
"""
import os
import numpy as np
import torch


class StreamText:
    def __init__(self, bin_path, vocab=16384, sample=50_000_000,
                 map_path=None, orig_vocab=128256):
        self.data = np.memmap(bin_path, dtype=np.uint32, mode="r")
        self.vocab = vocab
        if map_path is None:
            map_path = f"/home/glenn/projects/neuro/data/vocab_map_{vocab}.npy"
        if os.path.exists(map_path):
            self.lut = np.load(map_path)
        else:
            s = np.asarray(self.data[:sample], dtype=np.int64)
            uniq, cnt = np.unique(s, return_counts=True)
            keep = uniq[np.argsort(-cnt)[: vocab - 1]]
            lut = np.zeros(orig_vocab, dtype=np.int64)
            lut[keep] = np.arange(1, len(keep) + 1)
            os.makedirs(os.path.dirname(map_path), exist_ok=True)
            np.save(map_path, lut)
            self.lut = lut
        self.split = int(len(self.data) * 0.999)   # tiny val tail

    def batch(self, B, T, split="train", device="cpu"):
        lo, hi = (0, self.split) if split == "train" else (self.split, len(self.data) - T - 1)
        # one sequential read per batch (~B*T*4 tokens), then B random sub-windows
        region = B * (T + 1) * 4
        start = np.random.randint(lo, max(lo + 1, hi - region - 1))
        block = np.asarray(self.data[start:start + region], dtype=np.int64)
        block = self.lut[block]
        starts = np.random.randint(0, region - T - 1, size=B)
        xb = np.stack([block[s:s + T + 1] for s in starts])
        x = torch.from_numpy(xb[:, :-1]).to(device)
        y = torch.from_numpy(xb[:, 1:]).to(device)
        return x, y
