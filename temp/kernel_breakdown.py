# kernel_breakdown.py -- 2026-09-30 (fused-neuron scoping). COUNTS ONLY (shared GPU). Eager FORWARD kernels of ONE
# v4-draft spiking block (+ lif_exact_ml signed L4), attributed to its direct child modules and to the neuron call.
import sys, yaml, torch
from collections import Counter
from torch.profiler import profile, ProfilerActivity, record_function
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model
import lif_exact as LE

c = yaml.safe_load(open("/home/glenn/projects/neuro/configs/flagship_v4_363m.draft.yaml"))
c.update({"lif_exact": True, "lif_levels": 4, "lif_signed": True, "compile": False})
m = build_model(c).cuda().train()
blk = next(b for b in m.blocks if not hasattr(b, "qkv"))
ctx = {}
for name, mod in blk.named_children():
    def pre(_m, _a, name=name):
        ctx[name] = record_function(f"MOD::{name}"); ctx[name].__enter__()
    def post(_m, _a, _o, name=name):
        ctx[name].__exit__(None, None, None)
    mod.register_forward_pre_hook(pre); mod.register_forward_hook(post)
_orig = LE.lif_exact_ml
def wrapped(*a, **k):
    with record_function("MOD::NEURON"):
        return _orig(*a, **k)
LE.lif_exact_ml = wrapped
x = torch.randint(0, c["vocab"], (1, 512), device="cuda")
def f():
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        m(x)
f(); torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
    f(); torch.cuda.synchronize()

def sub_kernels(e):
    return len(e.kernels) + sum(sub_kernels(ch) for ch in e.cpu_children)
blk_names = set()
tot = Counter()
# total forward kernels in the first spiking block = everything under its forward_sequence; approximate with sum of labels
for e in p.events():
    if e.name.startswith("MOD::"):
        tot[e.name[5:]] += sub_kernels(e)
all_k = sum(1 for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA)
print(f"whole-model FORWARD kernels (B=1,T=512): {all_k}; spiking blocks {sum(1 for b in m.blocks if not hasattr(b, 'qkv'))}")
print("per-block-0 child modules (kernels summed over ALL spiking blocks sharing that child name):")
for k, v in tot.most_common(25):
    print(f"  {k:28s} {v}")
top = Counter(e.name[:70] for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA)
print("top kernel names (whole forward):")
for k, v in top.most_common(15):
    print(f"  {v:6d}  {k}")
