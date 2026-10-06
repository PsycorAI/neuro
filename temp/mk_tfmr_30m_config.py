# mk_tfmr_30m_config.py -- 2026-09-26: 30M preview of the FIXED transformer baseline.
# Control = configs/screen_bs512_s0_30m.yaml (spiking, 59.58M, done). This arm swaps ONLY
# the architecture to the baseline_transformer_363m recipe (all-attention, final_norm,
# depth_scaled_init, no GDN-2/engram/spike sparsity) + attn_single_residual:true (the bug
# fix). Same data, tokens, MQAR aux, optimizer, LR, seed. 2L/d1024 = 58.75M (param+depth match).
import yaml, pathlib
c = yaml.safe_load(open("configs/screen_bs512_s0_30m.yaml"))
c.update(run_name="screen_tfmrfix_s0_30m", n_neurons=1024,
         attention_layer_positions=[0, 1], attention_heads=8, attention_max_seq_len=4096,
         attention_qk_norm=True, attn_single_residual=True, final_norm=True,
         depth_scaled_init=True, delta_rule=False, decay_gate=False, beta_floor=0.0,
         gdn2_enabled=False, gdn2_chunked=False, engram_v2_enabled=False,
         sparsity_lambda=0.0, gate_rank=0)
hdr = ("# screen_tfmrfix_s0_30m -- 30M FIXED transformer baseline preview (2026-09-26).\n"
       "# Control: screen_bs512_s0_30m (spiking). AUTO-GEN temp/mk_tfmr_30m_config.py; see its header.\n")
pathlib.Path("configs/screen_tfmrfix_s0_30m.yaml").write_text(hdr + yaml.safe_dump(c, sort_keys=False))
# verify: builds, param count, and a forward pass on CPU
import sys, torch
sys.path.insert(0, "src")
from train_gpu import build_model
m = build_model(c)
print(f"screen_tfmrfix_s0_30m: {sum(p.numel() for p in m.parameters())/1e6:.2f}M")
print("attn blocks:", sum(1 for b in m.blocks if "Attention" in type(b).__name__), "/", len(m.blocks))
out = m(torch.randint(0, 16384, (2, 64)))
out = out[0] if isinstance(out, tuple) else out
print("forward ok", tuple(out.shape), "finite", bool(torch.isfinite(out).all()))
