# count_neuron_kernels.py -- 2026-09-30 (fused-neuron scoping, docs/fused_neuron_kernel_design.md). COUNTS ONLY, no timing
# (GPU 1 is shared with a live screen, so timings would be bogus; kernel/sync counts are load-independent).
# Eager fwd+bwd, B=1, T=512, bf16 autocast. Reports CUDA kernels + host syncs for (1) one neuron call in isolation and
# (2) one whole-model fwd+bwd, for the FPT neuron (lif_parallel) vs the fused exact kernel (lif_exact_ml).
import sys, yaml, torch
from torch.profiler import profile, ProfilerActivity
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model
import lif_parallel as LP, lif_exact as LE

def counts(fn):
    fn(); torch.cuda.synchronize()                      # warm-up (Triton JIT, allocator)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        fn(); torch.cuda.synchronize()
    ev = p.events()
    k = sum(1 for e in ev if e.device_type == torch.autograd.DeviceType.CUDA and e.name not in ("Memset (Device)",)
            and "Memcpy" not in e.name)
    sync = sum(1 for e in ev if e.name in ("cudaStreamSynchronize", "cudaDeviceSynchronize") or
               (e.name == "cudaMemcpyAsync" and False))
    d2h = sum(1 for e in ev if "Memcpy DtoH" in e.name)
    return k, sync - 1, d2h                              # -1: our own trailing synchronize

B, T = 1, 512
for N in (2048, 9728):
    cur = torch.randn(B, T, N, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    beta = torch.full((N,), 0.9, device="cuda")
    def f_fpt():
        s, m = LP.lif_parallel(cur, 0.9, vth=1.0, K=10); (s.float().sum() + m.float().sum()).backward()
    def f_ml():
        s, m = LE.lif_exact_ml(cur, beta, vth=1.0, levels=4, signed=True); (s.float().sum() + m.float().sum()).backward()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        print(f"neuron only N={N}: FPT K=10 kernels/syncs/DtoH = {counts(f_fpt)} | fused lif_exact_ml = {counts(f_ml)}", flush=True)

C = "/home/glenn/projects/neuro/configs/"
cases = [("30M signed 2L (lif_exact_ml)", C + "screen_r5_signed_s0_30m.yaml", {}),
         ("v4 draft as-is (FPT)", C + "flagship_v4_363m.draft.yaml", {}),
         ("v4 draft + lif_exact_ml signed L4", C + "flagship_v4_363m.draft.yaml",
          {"lif_exact": True, "lif_levels": 4, "lif_signed": True})]
for name, path, over in cases:
    c = yaml.safe_load(open(path)); c.update(over); c["compile"] = False
    m = build_model(c).cuda().train()
    x = torch.randint(0, c["vocab"], (B, T), device="cuda")
    def f():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(x); out = out[0] if isinstance(out, tuple) else out
        out.float().mean().backward()
    n_spk = sum(1 for b in m.blocks if not hasattr(b, "qkv"))
    print(f"{name:36s} layers {len(m.blocks)} (spiking {n_spk}) | whole-model fwd+bwd kernels/syncs/DtoH = {counts(f)}", flush=True)
    del m; torch.cuda.empty_cache()
