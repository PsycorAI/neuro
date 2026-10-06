# size_tfmr_30m.py -- 2026-09-26: param-match an all-attention (fixed single-residual)
# transformer to the 30M spiking control screen_bs512_s0_30m (59.6M). CPU build only.
import sys, yaml, gc
sys.path.insert(0, "src")
from train_gpu import build_model
ctrl = yaml.safe_load(open("configs/screen_bs512_s0_30m.yaml"))
m = build_model(ctrl); print(f"control spiking: {sum(p.numel() for p in m.parameters())/1e6:.2f}M"); del m
tf = yaml.safe_load(open("configs/baseline_transformer_363m.yaml"))
for L in [3, 4, 5, 6]:
    c = dict(tf)
    c.update(vocab=16384, d=1024, d_mem=1024, n_neurons=1024, n_layers=L,
             attention_layer_positions=list(range(L)), attention_heads=8,
             attn_single_residual=True, mtp_enabled=False)
    m = build_model(c)
    print(f"tfmr L{L}: {sum(p.numel() for p in m.parameters())/1e6:.2f}M")
    del m; gc.collect()
