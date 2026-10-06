"""Isolate the recall_eval OOM: one model, one N, with memory checkpoints
after each stage, to find exactly which call allocates the huge tensor.
"""
import sys, os, glob, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from recall_eval import build, latest, make_batch

ROOT = os.path.join(os.path.dirname(__file__), "..")
device = "cuda"


def mb():
    return torch.cuda.memory_allocated() / 1024**2


torch.cuda.reset_peak_memory_stats()
blob = torch.load(latest("x1_fdecay_adamw"), map_location=device, weights_only=False)
c = blob["cfg"]
c["gdn2_chunk_size"] = 64
m = build(c, device)
m.load_state_dict(blob["model"], strict=False)
m.eval()
print(f"after load: {mb():.0f} MB")

N = 64
B = 256
x, target, cand = make_batch(B, N, c["vocab"], device, seed=0)
print(f"batch shape {tuple(x.shape)}, pass1 shape {tuple(x[:, :-1].shape)}")

torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    out = m(x[:, :-1], return_final_state=True)
print(f"after pass1 (build M): alloc={mb():.0f} MB  peak={torch.cuda.max_memory_allocated()/1024**2:.0f} MB")
state = out[-1]

torch.cuda.reset_peak_memory_stats()
try:
    with torch.no_grad():
        logits = m(x[:, -1:], initial_state=state)[:, -1, :]
    print(f"after pass2 (query): alloc={mb():.0f} MB  peak={torch.cuda.max_memory_allocated()/1024**2:.0f} MB")
    print("NO OOM at N=64, B=256, chunk=64")
except torch.OutOfMemoryError as e:
    print(f"OOM during pass2. peak before failure: {torch.cuda.max_memory_allocated()/1024**2:.0f} MB")
    raise
