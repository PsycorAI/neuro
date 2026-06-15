"""Phase-1 data.

RepeatTask  - synthetic induction/copy task. Only solvable by storing
              token->successor bindings in the fast-weight memory; ablating the
              memory must collapse it to chance. Used to PROVE the mechanism.

TextData    - a slice of the real Llama-3-tokenized corpus (bdh's train.bin),
              remapped to the top-`vocab` most frequent tokens so a tiny model
              trains fast on CPU. Used for a language sanity check vs a bigram.
"""
import numpy as np
import torch


class RepeatTask:
    def __init__(self, seq_len=8, n_symbols=20):
        self.L = seq_len
        self.S = n_symbols
        self.sep = n_symbols
        self.vocab = n_symbols + 1

    def batch(self, B, device="cpu"):
        s = torch.randint(0, self.S, (B, self.L))
        sep = torch.full((B, 1), self.sep)
        full = torch.cat([s, sep, s], dim=1).to(device)      # (B, 2L+1)
        x = full[:, :-1]
        y = full[:, 1:]
        # second-copy target positions (exclude its first token: it has no prior binding)
        second = torch.arange(self.L + 1, 2 * self.L, device=device)
        return x, y, second


class TextData:
    def __init__(self, bin_path, n_tokens=1_500_000, vocab=4096,
                 orig_vocab=128256, val_frac=0.1):
        raw = np.memmap(bin_path, dtype=np.uint32, mode="r")
        toks = np.asarray(raw[:n_tokens], dtype=np.int64)
        # remap top-(vocab-1) frequent ids -> 1..vocab-1 ; everything else -> 0 (unk)
        uniq, counts = np.unique(toks, return_counts=True)
        keep = uniq[np.argsort(-counts)[: vocab - 1]]
        lut = np.zeros(orig_vocab, dtype=np.int64)
        lut[keep] = np.arange(1, len(keep) + 1)
        mapped = torch.from_numpy(lut[toks])
        n_val = int(len(mapped) * val_frac)
        self.train = mapped[:-n_val]
        self.val = mapped[-n_val:]
        self.vocab = vocab

    def batch(self, B, T, split="train", device="cpu"):
        data = self.train if split == "train" else self.val
        ix = torch.randint(0, len(data) - T - 1, (B,))
        x = torch.stack([data[i:i + T] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + T + 1] for i in ix]).to(device)
        return x, y

    def bigram_ce(self):
        """Add-1-smoothed bigram cross-entropy (nats) on the val split."""
        V = self.vocab
        counts = torch.zeros(V, V)
        t = self.train
        counts.view(-1).index_add_(0, t[:-1] * V + t[1:], torch.ones(len(t) - 1))
        probs = (counts + 1.0) / (counts.sum(1, keepdim=True) + V)
        v = self.val
        return (-probs[v[:-1], v[1:]].log().mean()).item()
