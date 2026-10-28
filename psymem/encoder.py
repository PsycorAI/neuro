# psymem/encoder.py -- 2026-10-27 (Neuro 85): the Psymem spiking front end (docs/n85_psymem_encoder_design_2026_10_27.md).
# Replaces the dense bge-small transformer. Token embeddings -> blocks of [depthwise token conv + FFN], where the input of
# every conv / linear is a signed integer spike count (src/spike_code.py SpikeCode, as in v4: accumulate-only on event-driven
# hardware) -> mean over tokens (accumulate) -> spike code -> a 384-d vector in the teacher's space. No attention, no softmax.
# Stage 1 trains it to mimic bge-small (psymem/enc_train.py); Psymem's expansion layer (FlyHash / Hebbian) and synapses are
# unchanged, so this makes the whole recall path spiking. Tokenizer = bge's WordPiece vocabulary (a lookup, not a network).
import sys, functools
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '/home/glenn/projects/neuro/src')
from spike_code import SpikeCode

TOK = 'BAAI/bge-small-en-v1.5'


class Block(nn.Module):
    def __init__(self, d, k, dil, levels):
        super().__init__()
        self.n1, self.c1 = nn.RMSNorm(d), SpikeCode(d, levels)
        self.conv = nn.Conv1d(d, d, k, padding=dil * (k // 2), dilation=dil, groups=d)  # token mixing, per channel
        self.n2, self.c2, self.up = nn.RMSNorm(d), SpikeCode(d, levels), nn.Linear(d, 4 * d)
        self.c3, self.down = SpikeCode(4 * d, levels, signed=False), nn.Linear(4 * d, d)

    def forward(self, x, mask):
        h = self.c1(self.n1(x)) * mask[..., None]
        x = x + self.conv(h.transpose(1, 2)).transpose(1, 2)
        return x + self.down(self.c3(F.relu(self.up(self.c2(self.n2(x)))) ** 2))


class SpikeEnc(nn.Module):
    def __init__(self, vocab=30522, d=384, layers=4, k=5, levels=4, out=384):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d, k, 2 ** i, levels) for i in range(layers))
        self.norm, self.code, self.out = nn.RMSNorm(d), SpikeCode(d, levels), nn.Linear(d, out, bias=False)

    def forward(self, ids, mask):
        x, m = self.emb(ids), mask.to(self.emb.weight.dtype)
        for b in self.blocks:
            x = b(x, m)
        pooled = (self.norm(x) * m[..., None]).sum(1) / m.sum(1, keepdim=True)
        return F.normalize(self.out(self.code(pooled)).float(), dim=-1)


class EncEmbedder:
    """Same interface as bench.Embedder: texts -> unit vectors (np.float32). Loads models/psymem/enc_<name>.pt."""

    def __init__(self, name, dev='cuda'):
        from transformers import AutoTokenizer
        ck = torch.load(f'/home/glenn/projects/neuro/models/psymem/enc_{name}.pt', map_location='cpu')
        self.m = SpikeEnc(**ck['cfg']).to(dev).eval(); self.m.load_state_dict(ck['model'])
        self.tok, self.dev, self.cache = AutoTokenizer.from_pretrained(TOK), dev, {}

    @torch.no_grad()
    def __call__(self, texts, bs=256):
        need = sorted({t for t in texts if t not in self.cache}, key=len)
        for i in range(0, len(need), bs):
            b = self.tok(need[i:i + bs], padding=True, truncation=True, max_length=256, return_tensors='pt').to(self.dev)
            self.cache.update(zip(need[i:i + bs], self.m(b['input_ids'], b['attention_mask']).cpu().numpy()))
        return np.stack([self.cache[t] for t in texts])


@functools.lru_cache(maxsize=2)
def load(name):
    return EncEmbedder(name)
