"""Parity test for the 2026-09-25 flags (attn_single_residual, telemetry_eval_only,
data_independent_windows). CPU, tiny v3-shaped model (GDN-2 + gate_norm + gate_rank
+ attention + engram_v2 + MTP).
  1. flags OFF: new model.py logits == committed HEAD model.py logits (train+eval),
     and new StreamText batches == HEAD StreamText batches.
  2. telemetry_eval_only: identical logits; no stats in train mode; stats in eval.
  3. attn_single_residual: stream after the attention block == block output (single add).
  4. independent_windows: train windows differ from default, val windows identical.
Run: git show HEAD:src/model.py > temp/_old_model.py (done below), then venv python.
"""
import sys, subprocess, importlib.util, numpy as np, torch
SRC = "/home/glenn/projects/neuro/src"
sys.path.insert(0, SRC)
import model as new_model, data_stream as new_ds

def load_head(name):
    code = subprocess.check_output(["git", "-C", "/home/glenn/projects/neuro", "show", f"HEAD:src/{name}.py"])
    p = f"/home/glenn/projects/neuro/temp/_old_{name}.py"
    open(p, "wb").write(code)
    spec = importlib.util.spec_from_file_location(f"old_{name}", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
old_model, old_ds = load_head("model"), load_head("data_stream")

KW = dict(d=64, n_neurons=192, d_mem=64, n_layers=3, use_fpt=True, fpt_K=10,
          delta_rule=True, decay_gate=True, beta_floor=0.1, gdn2_enabled=True,
          gdn2_chunked=True, gdn2_chunk_size=16, gdn2_gate_norm=True, gdn2_ckpt_chunks=False,
          gate_rank=4, use_rmsnorm=True, embed_scale=True, mtp_enabled=True, mtp_depth=2,
          engram_v2_enabled=True, engram_v2_K=512, engram_v2_heads=2, engram_v2_rank=8,
          attention_layer_positions=[1], attention_heads=4, compile_safe=True)
V = 512
def build(mod, **extra):
    torch.manual_seed(0)
    return mod.SpikingHebbianLM(V, **KW, **extra)
idx = torch.randint(0, V, (2, 40), generator=torch.Generator().manual_seed(1))
ok = True
def check(name, cond):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name)

# 1. flags off == HEAD
for mode in ("train", "eval"):
    a, b = build(old_model), build(new_model)
    getattr(a, mode)(); getattr(b, mode)()
    with torch.no_grad():
        check(f"default logits == HEAD ({mode})", torch.equal(a(idx), b(idx)))
bin_ = "/home/glenn/projects/bdh/data/tokenized/train.bin"
for split in ("train", "val"):
    np.random.seed(7); xo, yo = old_ds.StreamText(bin_, vocab=32768).batch(4, 64, split)
    np.random.seed(7); xn, yn = new_ds.StreamText(bin_, vocab=32768).batch(4, 64, split)
    check(f"default StreamText == HEAD ({split})", torch.equal(xo, xn) and torch.equal(yo, yn))

# 2. telemetry_eval_only
base, tel = build(new_model), build(new_model, telemetry_eval_only=True)
base.train(); tel.train()
with torch.no_grad():
    check("telemetry_eval_only: identical train logits", torch.equal(base(idx), tel(idx)))
spk_blocks = [b for b in tel.blocks if isinstance(b, new_model.SpikingHebbianBlock)]
check("telemetry_eval_only: no stats set in train mode",
      all(getattr(b, "_last_gate_stats", None) is None for b in spk_blocks))
tel.eval()
with torch.no_grad():
    tel(idx)
check("telemetry_eval_only: stats set in eval mode",
      all(b._last_gate_stats is not None and b._last_beta_stats is not None
          and b._last_alpha_stats is not None for b in spk_blocks))

# 3. attn_single_residual
m = build(new_model, attn_single_residual=True).eval()
cap = {}; attn = m.blocks[1]; orig = attn.forward_sequence
def spy(x, *a, **k):
    out = orig(x, *a, **k); cap["x"], cap["o"] = x, out[0]; return out
attn.forward_sequence = spy
cap2 = {}; nxt = m.blocks[2]; orig2 = nxt.forward_sequence
def spy2(x, *a, **k):
    cap2["x"] = x; return orig2(x, *a, **k)
nxt.forward_sequence = spy2
with torch.no_grad():
    m(idx)
check("attn_single_residual: next block input == attn block output (single residual)",
      torch.equal(cap2["x"], cap["o"]))  # default-doubles case: temp/verify_double_residual.py

# 4. independent windows
np.random.seed(7); xd, _ = new_ds.StreamText(bin_, vocab=32768).batch(8, 64, "train")
np.random.seed(7); xi, yi = new_ds.StreamText(bin_, vocab=32768, independent_windows=True).batch(8, 64, "train")
check("independent_windows: train batch differs + right shape", xi.shape == (8, 64) and not torch.equal(xd, xi))
check("independent_windows: y is x shifted", torch.equal(xi[:, 1:], yi[:, :-1]))
np.random.seed(7); vd, _ = new_ds.StreamText(bin_, vocab=32768).batch(8, 64, "val")
np.random.seed(7); vi, _ = new_ds.StreamText(bin_, vocab=32768, independent_windows=True).batch(8, 64, "val")
check("independent_windows: val batch unchanged", torch.equal(vd, vi))
print("ALL PASS" if ok else "SOME FAILED")
