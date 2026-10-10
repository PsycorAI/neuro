# CPU check of mqar_phrase_batch (p26h, 2026-10-06): length, query bigram unique and placed, target = its value.
import sys, time, numpy as np, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import mqar_phrase_batch, StreamText, BIN
data = StreamText(BIN, vocab=32768, orig_vocab=128256, val_tokens=14527194)
rng = np.random.RandomState(0)
C, V = 512, 32768
t0 = time.time()
for N, o in ((4, 0), (32, 511), (17, 100)):
    x, tg = mqar_phrase_batch(4, N, C, o, data, rng, "cpu")
    oo = min(o, C - 2)
    assert x.shape == (4, C + oo + 2) and x.shape[1] <= 2 * C, x.shape
    for i in range(4):
        a, b = x[i, -2].item(), x[i, -1].item()
        codes = (x[i, :-3] * V + x[i, 1:-2]).tolist()
        hits = [p for p, cde in enumerate(codes) if cde == a * V + b]
        assert len(hits) == 1 and hits[0] < C and hits[0] % 3 == 0, hits
        assert x[i, hits[0] + 2].item() == tg[i].item()
print(f"[OK] mqar_phrase_batch: lengths, unique query bigram, targets correct ({time.time() - t0:.2f}s for 12 rows)")
print("sample key/value:", x[0, -2:].tolist(), "->", tg[0].item())
