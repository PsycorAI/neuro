"""Tiny causal transformer baseline, for the quality + energy comparison (Goal 2)."""
import torch
import torch.nn as nn


class TinyTransformer(nn.Module):
    def __init__(self, vocab, d=128, n_head=2, n_layer=2, max_T=128):
        super().__init__()
        self.d = d
        self.n_layer = n_layer
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_T, d)
        layer = nn.TransformerEncoderLayer(
            d, n_head, dim_feedforward=4 * d, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layer)
        self.head = nn.Linear(d, vocab)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        h = self.embed(idx) + self.pos(pos)[None]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=idx.device), 1)
        h = self.blocks(h, mask=mask)
        return self.head(h)
