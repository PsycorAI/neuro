"""test_chunk_stream.py -- 2026-10-05. ChunkStreamer (scripts/demo/chunk_stream.py) vs one chunked forward over the
whole history: last-position logits for several history lengths. Also shows the memory-off gap. Any chunked-attention
model; for non-chunked ckpts pass --chunk.
  python temp/test_chunk_stream.py [--chunk 512] ckpt.pt"""
import sys, argparse, torch
sys.path.insert(0, "/home/glenn/projects/neuro/src"); sys.path.insert(0, "/home/glenn/projects/neuro/scripts/demo")
from train_gpu import build_model, StreamText, BIN
from chunk_stream import ChunkStreamer
ap = argparse.ArgumentParser(); ap.add_argument("--chunk", type=int, default=0); ap.add_argument("ckpt")
a = ap.parse_args()
s = torch.load(a.ckpt, map_location="cpu", weights_only=False); c = dict(s["cfg"])
m = build_model(c).cuda(); m.load_state_dict(s["model"], strict=False); m.eval()
C = a.chunk or int(c["attn_chunk"])
st = ChunkStreamer(m, C)
data = StreamText(c.get("data_bin", BIN), vocab=c["vocab"], orig_vocab=c.get("data_orig_vocab", 128256),
                  val_tokens=c.get("data_val_tokens"))
import numpy as np
np.random.seed(3)
x, _ = data.batch(1, 4 * C, "val", "cpu")
ids = x[0].tolist()
with torch.no_grad():
    for L in (C // 2, C + 100, 2 * C + 1, 3 * C + 300, 4 * C):
        full = m(torch.tensor([ids[:L]], device="cuda"))[0, -1].float()
        sl, start = st.logits(ids[:L])
        sl = sl.float()
        p, q = torch.softmax(full, -1), torch.softmax(sl, -1)
        print(f"L={L:5d} window starts {start:5d} | max|dlogit| {(full - sl).abs().max().item():.3f} | "
              f"top1 agree {int(full.argmax() == sl.argmax())} | TV {0.5 * (p - q).abs().sum().item():.4f} | "
              f"cached chunks {len(st.cache)}")
print("[OK] stream test done")
