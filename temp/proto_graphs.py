"""proto_graphs.py -- 2026-10-02 (Glenn OK): CUDA-graphs Phase-1 prototype (docs/cuda_graphs_v4_design.md).
Standalone (no edits to shared modules). Times the block stack's fwd+bwd (embedding/head eager, no optimizer, no
MQAR) for one config in four modes and checks gradient parity of the graphed mode vs eager:
  eager | eager+graphs | regional compile | regional compile+graphs
Each block is wrapped in an adapter whose only input is x (zero states made inside, residual rule applied), and all
adapters are captured together by torch.cuda.make_graphed_callables (shared pool; fwd in order, bwd reversed).
DSN k is fixed (k_cur None): a graph bakes Python ints in (the k-ramp needs a separate fix).
  python temp/proto_graphs.py --config configs/probe_g4_c0s1.yaml --batch 8 [--modes eager,graph,rc,rcgraph]"""
import argparse, sys, time, yaml
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/glenn/projects/neuro/src")
from train_gpu import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--modes", default="eager,graph,rc,rcgraph")
a = ap.parse_args()
c = yaml.safe_load(open(a.config))
B, T, dev = a.batch, c["block_size"], "cuda"
torch.manual_seed(0)
model = build_model(c).to(dev).train()
for b_ in model.blocks:                     # standard flag (telemetry .item() syncs break capture)
    if hasattr(b_, "telemetry_eval_only"):
        b_.telemetry_eval_only = True


class Adapter(nn.Module):
    def __init__(self, m, blk):
        super().__init__()
        self.blk, self.n, self.res = blk, m.n_neurons, (m.n_layers > 1 and not m._attn_owns_residual(blk))
        self.fpt_K = m.fpt_K
    def forward(self, x):
        Bx = x.shape[0]
        mem = torch.zeros(Bx, self.n, device=x.device)
        M = torch.zeros(Bx * self.blk.n_state_units, self.blk.d_state, self.blk.d_state, device=x.device)
        prev = torch.zeros(Bx, self.n, device=x.device)
        out, _, _, spk, _ = self.blk.forward_sequence(x, mem, M, prev, fpt_K=self.fpt_K)
        return out + x if self.res else out


def make_stack(mode):
    ads = [Adapter(model, b) for b in model.blocks]
    if mode.startswith("rc"):
        ads = [torch.compile(m_, mode="default", dynamic=False) for m_ in ads]
    if mode.endswith("graph"):
        # warm-up on a side stream (compile + autotune + lazy init) before capture
        x0 = torch.randn(B, T, c["d"], device=dev, requires_grad=True)
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
            for _ in range(3):
                h = x0
                for m_ in ads:
                    h = m_(h)
                h.float().mean().backward()
        torch.cuda.current_stream().wait_stream(s)
        model.zero_grad(set_to_none=True)
        # sample inputs: each block's input = previous block's output (bf16 under autocast)
        samples, h = [], torch.randn(B, T, c["d"], device=dev)
        with torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
            for m_ in ads:
                samples.append((h.detach().requires_grad_(),))
                with torch.no_grad():
                    h = m_(h)
        with torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
            # allow_unused_input: blocks carry dead params (W_beta/W_alpha etc., code review 10-01)
            ads = list(torch.cuda.make_graphed_callables(tuple(ads), tuple(samples), allow_unused_input=True))
        model.zero_grad(set_to_none=True)
    return ads


def micro(ads, idx, tgt):
    with torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
        h = model.embed(idx) * model._embed_scale_factor
        if model.engram_v2_enabled:
            h = h + model.engram_v2(idx)
        for m_ in ads:
            h = m_(h)
        if model.out_norm is not None:
            h = model.out_norm(h)
        loss = F.cross_entropy(model.head(h).float().view(-1, c["vocab"]), tgt.view(-1))
    loss.backward()
    return loss


g = torch.Generator(device=dev).manual_seed(1)
idx = torch.randint(0, c["vocab"], (B, T), device=dev, generator=g)
tgt = torch.randint(0, c["vocab"], (B, T), device=dev, generator=g)
grads = {}
for mode in a.modes.split(","):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time(); ads = make_stack(mode); setup = time.time() - t0
    for _ in range(5):                                   # warm-up (compile for rc)
        micro(ads, idx, tgt)
    model.zero_grad(set_to_none=True)
    micro(ads, idx, tgt)                                 # parity: grads of one micro-step
    grads[mode] = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(a.iters):
        micro(ads, idx, tgt)
    torch.cuda.synchronize(); dt = (time.time() - t0) / a.iters
    model.zero_grad(set_to_none=True)
    ref = grads.get("rc" if mode == "rcgraph" else "eager")
    par = ""
    if ref is not None and mode in ("graph", "rcgraph", "eager2"):   # eager2 = eager rerun: the noise floor
        rel = max(((grads[mode][n] - ref[n]).norm() / (ref[n].norm() + 1e-12)).item() for n in ref)
        par = f" | max rel grad diff vs {'rc' if mode == 'rcgraph' else 'eager'} {rel:.2e}"
    print(f"[graphs] {mode:8s} B{B}: {dt*1000:7.1f} ms/micro = {B*T/dt:8,.0f} tok/s | peak reserved "
          f"{torch.cuda.max_memory_reserved()/2**30:.1f} GB | setup {setup:.0f}s{par}", flush=True)
    del ads
