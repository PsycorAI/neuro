"""Streaming text over the full Llama-3-tokenized corpus (bdh's train.bin).

memmaps the whole file (no in-RAM cap) and remaps tokens to the top-`vocab` most
frequent ids via a LUT built once from a sample and cached to disk.
"""
import os
import numpy as np
import torch


class StreamText:
    def __init__(self, bin_path, vocab=16384, sample=50_000_000,
                 map_path=None, orig_vocab=128256, independent_windows=False):
        self.data = np.memmap(bin_path, dtype=np.uint32, mode="r")
        self.vocab = vocab
        # independent_windows (2026-09-25, default off = byte-identical): the
        # default sampler draws all B windows from ONE ~4*B*(T+1)-token region,
        # so a microbatch is a few correlated documents (and windows overlap).
        # True draws each train window from its own random start. Train split
        # only -- val keeps the old sampler so eval windows match across arms.
        self.independent_windows = independent_windows
        # Random-access view for the independent-windows sampler (2026-09-30). The device read-ahead is 8 MB, so every
        # cold 2 KB window read pulled 8 MB (~37 ms; ~1.2 s per 32-window batch -> the 30M screens were ~2x data-bound).
        # MADV_RANDOM on a SEPARATE mapping turns read-ahead off for these reads only (~17 ms/batch, temp/bench_madvise.py);
        # the bytes read are identical, and the region sampler / val keep read-ahead on self.data.
        self._rand_data = self.data
        if independent_windows:
            import mmap
            self._rand_data = np.memmap(bin_path, dtype=np.uint32, mode="r")
            if getattr(self._rand_data, "_mmap", None) is not None and hasattr(mmap, "MADV_RANDOM"):
                self._rand_data._mmap.madvise(mmap.MADV_RANDOM)
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

    def batch(self, B, T, split="train", device="cpu", return_raw=False):
        lo, hi = (0, self.split) if split == "train" else (self.split, len(self.data) - T - 1)
        if self.independent_windows and split == "train":
            starts = np.random.randint(lo, hi - T - 1, size=B)
            rb = np.stack([np.asarray(self._rand_data[s:s + T + 1], dtype=np.int64) for s in starts])
            xb = self.lut[rb]
            x = torch.from_numpy(xb[:, :-1]).to(device)
            y = torch.from_numpy(xb[:, 1:]).to(device)
            if not return_raw:
                return x, y
            return x, y, torch.from_numpy(rb[:, :-1]).to(device)
        # one sequential read per batch (~B*T*4 tokens), then B random sub-windows
        region = B * (T + 1) * 4
        start = np.random.randint(lo, max(lo + 1, hi - region - 1))
        raw_block = np.asarray(self.data[start:start + region], dtype=np.int64)
        block = self.lut[raw_block]
        starts = np.random.randint(0, region - T - 1, size=B)
        xb = np.stack([block[s:s + T + 1] for s in starts])
        x = torch.from_numpy(xb[:, :-1]).to(device)
        y = torch.from_numpy(xb[:, 1:]).to(device)
        if not return_raw:
            return x, y
        # raw Llama-3 IDs for the SAME windows (input only — needed for KD teacher)
        rb = np.stack([raw_block[s:s + T + 1] for s in starts])
        raw = torch.from_numpy(rb[:, :-1]).to(device)
        return x, y, raw


class SequentialStream:
    """Contiguous-document streaming for STATEFUL training (TBPTT).

    Unlike StreamText (which samples random windows), this maintains B parallel
    read cursors at distinct random start positions and returns the NEXT
    contiguous block of tokens for each cursor on every call. Consecutive calls
    continue where the last left off, so the model's carried state (M, LIF
    membrane, prev_spk) follows a real document stream instead of resetting at
    every random window.

    The corpus (~14.5B tokens) is enormous relative to what a short run consumes,
    so cursors won't collide or wrap in practice. If a cursor would run past the
    train region it is re-seeded to a new random spot and its reset flag is set,
    so the training loop can zero that batch element's state at the seam.
    """

    def __init__(self, bin_path, vocab, batch_size, map_path=None,
                 orig_vocab=128256, seed=0, margin_tokens=20_000_000):
        self.data = np.memmap(bin_path, dtype=np.uint32, mode="r")
        self.vocab = vocab
        self.B = batch_size
        self.margin = margin_tokens
        if map_path is None:
            map_path = f"/home/glenn/projects/neuro/data/vocab_map_{vocab}.npy"
        self.lut = np.load(map_path)
        self.train_hi = int(len(self.data) * 0.999)
        self.rng = np.random.default_rng(seed)
        # B distinct random start cursors, each with room to run without wrapping
        self.cursors = self.rng.integers(0, self.train_hi - margin_tokens,
                                         size=batch_size)

    def next(self, T, device):
        """Return (x, y, reset_mask). reset_mask[i]=True if cursor i was re-seeded
        this step (its carried state should be zeroed by the caller)."""
        xb = np.empty((self.B, T), dtype=np.int64)
        yb = np.empty((self.B, T), dtype=np.int64)
        reset = np.zeros(self.B, dtype=bool)
        for i in range(self.B):
            c = int(self.cursors[i])
            if c + T + 1 > self.train_hi:               # would run past region
                c = int(self.rng.integers(0, self.train_hi - self.margin))
                self.cursors[i] = c
                reset[i] = True
            block = self.lut[np.asarray(self.data[c:c + T + 1], dtype=np.int64)]
            xb[i] = block[:-1]
            yb[i] = block[1:]
            self.cursors[i] += T
        x = torch.from_numpy(xb).to(device)
        y = torch.from_numpy(yb).to(device)
        return x, y, torch.from_numpy(reset).to(device)

    def skip(self, n_calls, T):
        """Advance the cursors as if next(T) had been called n_calls times, without reading data (2026-09-29).
        Used on checkpoint resume so a stateful run does not replay its stream from the start. Same wrap rule
        (and rng draws) as next(), so the cursors match an uninterrupted run exactly."""
        for _ in range(n_calls):
            for i in range(self.B):
                if int(self.cursors[i]) + T + 1 > self.train_hi:
                    self.cursors[i] = int(self.rng.integers(0, self.train_hi - self.margin))
                self.cursors[i] += T
