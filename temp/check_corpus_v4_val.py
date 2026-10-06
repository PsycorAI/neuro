"""The v4 corpus with data_val_tokens must give byte-identical val batches to the old corpus (same seeds)."""
import sys, numpy as np, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from data_stream import StreamText
old = StreamText("/home/glenn/projects/bdh/data/tokenized/train.bin", vocab=32768)
new = StreamText("/home/glenn/projects/bdh/data/tokenized_v4/train.bin", vocab=32768, val_tokens=14527194)
print(f"old val tail {len(old.data) - old.split:,} | new val tail {len(new.data) - new.split:,} | new train {new.split / 1e9:.3f}B")
same = True
for seed in (12345, 7, 2026):
    np.random.seed(seed); a = old.batch(8, 512, "val")
    np.random.seed(seed); b = new.batch(8, 512, "val")
    same &= all(torch.equal(x, y) for x, y in zip(a, b))
print("val batches identical:", same)
np.random.seed(1); x, _ = new.batch(4, 512, "train")
print("train batch ok:", tuple(x.shape), "OOV(0) share", float((x == 0).float().mean()))
