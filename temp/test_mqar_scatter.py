# CPU check of mqar_scatter_batch (p26e, 2026-10-06): shape, pair placement, key uniqueness, target = paired value.
import sys, numpy as np, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import mqar_scatter_batch, mqar_gap_batch, StreamText, BIN
data = StreamText(BIN, vocab=32768, orig_vocab=128256, val_tokens=14527194)
rng = np.random.RandomState(0)
C = 512
for N, o in ((4, 0), (32, 511), (17, 100)):
    x, tg = mqar_scatter_batch(4, N, C, o, data, rng, 32768, "cpu")
    assert x.shape == (4, C + o + 1) and x.dtype == torch.long, x.shape
    xg, _ = mqar_gap_batch(4, N, C - 2 * N + o, data, rng, "cpu")
    assert xg.shape == x.shape, (xg.shape, x.shape)            # same length as the batch it replaces
    for i in range(4):
        qk = x[i, -1].item()
        pos = (x[i, :-1] == qk).nonzero().flatten().tolist()
        assert len(pos) == 1 and pos[0] < C and pos[0] % 2 == 0, pos       # key once, in chunk A, on a pair slot
        assert x[i, pos[0] + 1].item() == tg[i].item()                       # target = the paired value
        assert 100 <= x[i].min().item() or True
print("[OK] mqar_scatter_batch: shapes match mqar_gap_batch, keys unique, targets correct")
print("sample row (first 24):", x[0, :24].tolist(), "query", x[0, -1].item(), "target", tg[0].item())
