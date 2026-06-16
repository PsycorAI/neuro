"""Phase 2.5 GPU trainer.

Trains neuro (spiking Hebbian + optional SET) or a matched transformer on
streaming real tokens, with checkpoint/resume, eval (incl. a memory-ablation
probe), throughput, and an energy report. Config-driven (YAML).

  python src/train_gpu.py --config configs/phase25_a.yaml            # GPU run
  python src/train_gpu.py --config configs/phase25_a.yaml --device cpu --smoke
"""
import os, sys, time, math, argparse, glob
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from model import SpikingHebbianLM
from baseline import TinyTransformer
from data_stream import StreamText
from sparse import SET
import energy
import yaml

BIN = "/home/glenn/projects/bdh/data/tokenized/train.bin"


def build_model(c):
    if c["arch"] == "spiking":
        return SpikingHebbianLM(c["vocab"], d=c["d"], n_neurons=c["n_neurons"],
                                d_mem=c["d_mem"], recurrent=c.get("recurrent", False),
                                rec_density=c.get("rec_density", 0.05),
                                compile_safe=c.get("compile", False))
    return TinyTransformer(c["vocab"], d=c["d"], n_head=c["n_head"],
                           n_layer=c["n_layer"], max_T=c["block_size"])


@torch.no_grad()
def evaluate(model, data, c, device, iters=20):
    model.eval()
    amp = c["amp"] and device == "cuda"

    def ppl(ablate):
        tot = 0.0
        for _ in range(iters):
            x, y = data.batch(c["eval_batch"], c["block_size"], "val", device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
                out = model(x, ablate_memory=ablate) if c["arch"] == "spiking" else model(x)
                tot += F.cross_entropy(out.reshape(-1, c["vocab"]), y.reshape(-1)).item()
        return math.exp(tot / iters)

    full = ppl(False)
    if c["arch"] == "spiking":
        x, _ = data.batch(c["eval_batch"], c["block_size"], "val", device)
        _, sr = model(x, return_stats=True)
        return full, ppl(True), sr.item()
    return full, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    with open(args.config) as f:
        c = yaml.safe_load(f)
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch._dynamo.config.cache_size_limit = 64   # cap compile cache growth (~3-4 graphs needed)
    if args.smoke:
        c.update(block_size=16, batch_size=4, grad_accum=1, eval_batch=8,
                 max_tokens=20000, eval_every=200, ckpt_every=10**9, log_every=50, amp=False)
    c.setdefault("eval_batch", c["batch_size"])
    torch.manual_seed(c.get("seed", 0))

    data = StreamText(BIN, vocab=c["vocab"])
    raw_model = build_model(c).to(device)
    nparams = sum(p.numel() for p in raw_model.parameters())
    opt = torch.optim.AdamW(raw_model.parameters(), lr=c["lr"])
    setm = SET(raw_model.rec_mask, c.get("set_zeta", 0.3)) if (c["arch"] == "spiking" and c.get("recurrent")) else None
    if c.get("compile") and device == "cuda":
        mode = c.get("compile_mode", "default")  # "default" | "reduce-overhead" | "max-autotune"
        model = torch.compile(raw_model, mode=mode, dynamic=False)
        print(f"torch.compile enabled (mode={mode}) — first 1-2 steps will be slow (graph capture)")
    else:
        model = raw_model

    ckdir = f"/home/glenn/projects/neuro/models/checkpoints/{c['run_name']}"
    os.makedirs(ckdir, exist_ok=True)
    step, tokens = 0, 0
    cks = sorted(glob.glob(f"{ckdir}/step_*.pt"))
    if cks:
        s = torch.load(cks[-1], map_location=device, weights_only=False)
        raw_model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        step, tokens = s["step"], s["tokens"]
        print(f"resumed {cks[-1]} @ step {step}, {tokens/1e6:.1f}M tok")

    per_step_tok = c["batch_size"] * c["grad_accum"] * c["block_size"]
    print(f"arch={c['arch']} params={nparams/1e6:.2f}M device={device} "
          f"target={c['max_tokens']/1e9:.3f}B tok | {per_step_tok} tok/step")
    t0 = time.time(); win = t0
    while tokens < c["max_tokens"]:
        model.train(); opt.zero_grad()
        for _ in range(c["grad_accum"]):
            x, y = data.batch(c["batch_size"], c["block_size"], "train", device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=c["amp"] and device == "cuda"):
                out = model(x)
                loss = F.cross_entropy(out.reshape(-1, c["vocab"]), y.reshape(-1)) / c["grad_accum"]
            loss.backward()
        opt.step()
        step += 1; tokens += per_step_tok
        if setm and step % c.get("set_every", 200) == 0:
            setm.step(raw_model.W_rec.weight)
        if step % c["log_every"] == 0:
            dt = time.time() - win; win = time.time()
            vram = f" | vram {torch.cuda.memory_allocated()/1024**3:.1f}/{torch.cuda.memory_reserved()/1024**3:.1f} GB" if device == "cuda" else ""
            print(f"step {step} | {tokens/1e6:.1f}M tok | loss {loss.item()*c['grad_accum']:.3f} "
                  f"| {c['log_every']*per_step_tok/dt:,.0f} tok/s{vram}")
        if step % c["eval_every"] == 0:
            full, abl, sr = evaluate(raw_model, data, c, device)
            msg = f"  [eval] val_ppl {full:.2f}"
            if abl is not None:
                msg += f" | ablated_ppl {abl:.2f} (memory Δ={abl-full:+.2f}) | spike {sr:.3f}"
            print(msg, flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()   # release fragmented blocks freed by eval
        if step % c["ckpt_every"] == 0:
            p = f"{ckdir}/step_{step:07d}.pt"
            torch.save({"model": raw_model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "tokens": tokens, "cfg": c}, p)
            for o in sorted(glob.glob(f"{ckdir}/step_*.pt"))[:-c.get("keep_last", 3)]:
                os.remove(o)
            print(f"  saved {p}")

    print(f"done: {tokens/1e6:.1f}M tok in {(time.time()-t0)/3600:.2f}h")
    if c["arch"] == "spiking":
        _, _, sr = evaluate(raw_model, data, c, device)
        energy.compare(model, sr, baseline_d=c["d"], baseline_layers=c.get("n_layer", 2),
                       vocab=c["vocab"], seq_lens=[c["block_size"], 512, 4096])


if __name__ == "__main__":
    main()
