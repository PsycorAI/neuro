"""Curriculum warmup: easy-to-hard document ordering for the first fraction
of pretraining (arXiv:2506.11300). Data-side only, zero arch risk.

The corpus (bdh's train.bin) is a flat pre-tokenized stream with no document
boundaries, so difficulty is scored per fixed-size block of raw token ids
using gzip compression ratio as a compression-ratio proxy (Lampinen et al.'s
difficulty signal family): a block whose token-id bytes compress well is
repetitive/predictable ("easy"); a block near the entropy ceiling of gzip
is not ("hard"). Scored once over a sample region and cached to disk, same
pattern as StreamText's vocab_map cache.
"""
import os
import zlib

import numpy as np
import torch


def _score_difficulty(raw_block):
    """Lower = more compressible = easier. raw_block: (block_tokens,) uint32."""
    raw_bytes = np.ascontiguousarray(raw_block, dtype=np.uint32).tobytes()
    comp = zlib.compress(raw_bytes, level=6)
    return len(comp) / len(raw_bytes)


def build_curriculum_order(bin_path, block_tokens=4096, sample_tokens=50_000_000,
                            cache_path=None):
    """Returns raw-stream start offsets of non-overlapping blocks, sorted
    ascending by difficulty (easiest first). Cached to disk after first build."""
    if cache_path is None:
        cache_path = f"/home/glenn/projects/neuro/data/curriculum_order_{block_tokens}.npy"
    if os.path.exists(cache_path):
        return np.load(cache_path)
    data = np.memmap(bin_path, dtype=np.uint32, mode="r")
    n_blocks = min(sample_tokens, len(data)) // block_tokens
    starts = np.arange(n_blocks, dtype=np.int64) * block_tokens
    scores = np.empty(n_blocks, dtype=np.float64)
    for i, s in enumerate(starts):
        scores[i] = _score_difficulty(data[s:s + block_tokens])
    order = starts[np.argsort(scores, kind="stable")]
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, order)
    return order


class CurriculumStream:
    """Drop-in wrapper over StreamText: same .batch(B, T, split, device)
    interface. For the first warmup_calls train-split calls, draws windows
    from the easy-to-hard block order (a random T+1 sub-window within each
    block, so repeated block reuse after wraparound isn't byte-identical);
    every val-split call and every train call past warmup_calls delegates
    straight to the wrapped StreamText's normal uniform-random sampling."""

    def __init__(self, base, bin_path, warmup_calls, block_tokens=4096,
                 sample_tokens=50_000_000, cache_path=None):
        self.base = base
        self.warmup_calls = warmup_calls
        self.block_tokens = block_tokens
        self._train_calls = 0
        self._oi = 0
        self.order = build_curriculum_order(bin_path, block_tokens, sample_tokens,
                                             cache_path)

    def batch(self, B, T, split="train", device="cpu", **kwargs):
        if split != "train" or self._train_calls >= self.warmup_calls:
            if split == "train":
                self._train_calls += 1
            return self.base.batch(B, T, split, device, **kwargs)
        assert not kwargs.get("return_raw", False), (
            "CurriculumStream doesn't support return_raw during the warmup "
            "phase (no KD teacher screens combine with curriculum yet)")
        assert T + 1 <= self.block_tokens, (
            f"block_size+1 ({T + 1}) must fit in curriculum_block_tokens "
            f"({self.block_tokens})")
        self._train_calls += 1
        idx = (self._oi + np.arange(B)) % len(self.order)
        self._oi = (self._oi + B) % len(self.order)
        starts = self.order[idx]
        xb = np.empty((B, T), dtype=np.int64)
        yb = np.empty((B, T), dtype=np.int64)
        for i, s in enumerate(starts):
            off = np.random.randint(0, self.block_tokens - T)
            raw = np.asarray(self.base.data[s + off:s + off + T + 1], dtype=np.int64)
            mapped = self.base.lut[raw]
            xb[i] = mapped[:-1]
            yb[i] = mapped[1:]
        x = torch.from_numpy(xb).to(device)
        y = torch.from_numpy(yb).to(device)
        return x, y
