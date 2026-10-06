# census_compiled.py -- 2026-09-30 (CUDA-graph scoping, docs/cuda_graphs_v4_design.md). COUNTS ONLY (GPU 1 is shared).
# The 1B probe's STRUCTURE (temp/probe_1b_b8.yaml: 9L, chunk 256, engram, mtp, gate_rank...) at reduced WIDTH so it fits
# beside a live screen; kernel/sync/graph counts depend on structure, not width. Compiled (mode=default, dynamic=False),
# one main fwd+bwd at B=1, T=512, after 3 warm-up iterations. Variants: FPT neuron (probe as-is) vs lif_exact_ml signed L4.
import sys, yaml, torch, re
from collections import Counter
from torch.profiler import profile, ProfilerActivity
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model
import torch._dynamo
from torch._dynamo.utils import counters

def cat(n):
    if n.startswith("triton_") or "triton_poi" in n or "triton_red" in n or "triton_per" in n: return "inductor-fused"
    if "_lif" in n: return "lif kernel"
    if re.search(r"gemm|gemv|cutlass|sm\d+_xmma|ampere|cublas|Kernel2", n): return "matmul"
    if "Memset" in n or "Memcpy" in n: return "memset/copy"
    return "eager aten (" + n.split("<")[0].replace("void ", "")[:40] + ")"

base = yaml.safe_load(open("/home/glenn/projects/neuro/temp/probe_1b_b8.yaml"))
for name, over in [("probe structure, FPT (as-is)", {}),
                   ("probe structure, lif_exact_ml signed L4", {"lif_exact": True, "lif_levels": 4, "lif_signed": True})]:
    c = dict(base); c.update(over)
    c.update({"d": 512, "d_mem": 512, "n_neurons": 2432})          # width / 4: same structure
    torch._dynamo.reset(); counters.clear()
    m = build_model(c).cuda().train()
    cm = torch.compile(m, mode="default", dynamic=False)
    x = torch.randint(0, c["vocab"], (1, 512), device="cuda")
    def f():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = cm(x); out = out[0] if isinstance(out, tuple) else out
        out.float().mean().backward()
    for _ in range(3): f()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        f(); torch.cuda.synchronize()
    ev = p.events()
    ks = [e.name for e in ev if e.device_type == torch.autograd.DeviceType.CUDA]
    syncs = sum(1 for e in ev if e.name in ("cudaStreamSynchronize", "cudaDeviceSynchronize")) - 1
    d2h = sum(1 for e in ev if "Memcpy DtoH" in e.name)
    launches = sum(1 for e in ev if e.name in ("cudaLaunchKernel", "cuLaunchKernel", "cuLaunchKernelEx", "cudaLaunchKernelExC"))
    print(f"== {name}: fwd+bwd kernels {len(ks)} | launch API calls {launches} | host syncs {syncs} (DtoH {d2h}) | "
          f"dynamo unique graphs {counters['stats'].get('unique_graphs', '?')}", flush=True)
    for k, v in Counter(cat(n) for n in ks).most_common(10):
        print(f"   {v:6d}  {k}")
    del m, cm; torch.cuda.empty_cache()
