# CPU smoke (p26h, 2026-10-06): phrase-MQAR batch -> pad to 2C -> 7L model forward + CE at the query -> backward.
import sys, yaml, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model, mqar_phrase_batch, StreamText, BIN
c = yaml.safe_load(open("/home/glenn/projects/neuro/configs/screen_p26h_7L_xphrase25.yaml"))
c.update(compile=False, amp=False, lif_exact=False, lif_signed=False, lif_levels=1, lif_learn_beta=False)
m = build_model(c)
data = StreamText(c.get("data_bin", BIN), vocab=c["vocab"], orig_vocab=128256, val_tokens=c.get("data_val_tokens"))
rng = np.random.RandomState(1)
C = c["attn_chunk"]
for b in m.blocks:
    if hasattr(b, "qkv"):
        b.attn_chunk = C
x, tg = mqar_phrase_batch(2, 16, C, 300, data, rng, "cpu")
L = x.shape[1]
x = F.pad(x, (0, 2 * C - L))
loss = F.cross_entropy(m(x)[:, L - 1, :], tg)
loss.backward()
print(f"[OK] phrase MQAR through 7L model on CPU: len {L} padded {x.shape[1]}, loss {loss.item():.3f}, grads finite "
      f"{all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)}")
