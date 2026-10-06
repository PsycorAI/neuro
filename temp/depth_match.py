"""Find param-matched deeper arms for the 30M depth-vs-width screen.
Control = configs/screen_bs512_s0_30m.yaml (d1024, L2, N2048 = 2.0xd).
Deeper arms hold N/d = 2.0 and shrink d to match total params."""
import sys, yaml
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model

base = yaml.safe_load(open("/home/glenn/projects/neuro/configs/screen_bs512_s0_30m.yaml"))
def count(d, L):
    c = dict(base); c.update(d=d, d_mem=d, n_neurons=2 * d, n_layers=L)
    return sum(p.numel() for p in build_model(c).parameters())
print(f"control d1024 L2: {count(1024, 2)/1e6:.2f}M")
for L in (3, 4):
    for d in range(640, 1025, 64):
        print(f"L{L} d{d} N{2*d}: {count(d, L)/1e6:.2f}M")
