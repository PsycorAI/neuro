# mk_tfmr426_configs.py -- 2026-09-26: v3-matched transformer baseline (Phase 0 #1 candidate).
# Same data/tokenizer/vocab/d/block/MQAR aux/MTP/z-loss as configs/flagship_v3_363m.yaml; arch =
# baseline_transformer_363m (all-attention, final_norm, depth_scaled_init) + attn_single_residual;
# optimizer = AdamW lr 3e-4 wd 0.1 (standard for ~400M GPTs; the 30M preview showed the spiking
# NorMuon recipe cripples attention: ppl 87.8 vs 68.8). n_layers chosen to match v3's 426.5M.
# Writes temp/tfmr426_L<L>.yaml sizes + 5080 tok/s/VRAM probe configs temp/probe_tfmr426_b{8,4}.yaml.
import sys, yaml, gc
sys.path.insert(0, "src")
from train_gpu import build_model
v3 = yaml.safe_load(open("configs/flagship_v3_363m.yaml"))
tf = yaml.safe_load(open("configs/baseline_transformer_363m.yaml"))
best = None
for L in [15, 16, 17, 18]:
    c = dict(tf)
    c.update(vocab=v3["vocab"], d=v3["d"], d_mem=v3["d"], n_neurons=v3["d"], n_layers=L,
             attention_layer_positions=list(range(L)), attention_heads=v3["attention_heads"],
             attention_qk_norm=True, attn_single_residual=True, telemetry_eval_only=True,
             data_independent_windows=False,          # v3 had the correlated sampler; match it
             block_size=v3["block_size"],
             mqar_aug=v3["mqar_aug"], mqar_N_min=v3["mqar_N_min"], mqar_N_max=v3["mqar_N_max"],
             mqar_weight=v3["mqar_weight"],
             mtp_enabled=v3["mtp_enabled"], mtp_offset=v3["mtp_offset"], mtp_weight=v3["mtp_weight"],
             mtp_depth=v3["mtp_depth"], z_loss_weight=v3["z_loss_weight"],
             max_tokens=v3["max_tokens"], data_bin=v3["data_bin"],
             optimizer="adamw", lr=3.0e-4, weight_decay=0.1,
             run_name=f"baseline_tfmr426_L{L}")
    for k in ["normuon", "normuon_beta2", "normuon_eps", "muon_lr", "muon_fix_gate_matrices",
              "muon_route_embeddings_by_type", "muon_precise_gate_filter"]:
        c.pop(k, None)
    m = build_model(c)
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"L{L}: {n:.1f}M (v3 426.5M)")
    if best is None or abs(n - 426.5) < abs(best[1] - 426.5):
        best = (L, n, c)
    del m; gc.collect()
L, n, c = best
print(f"-> chosen L{L} = {n:.1f}M")
yaml.safe_dump(c, open(f"temp/tfmr426_L{L}.yaml", "w"), sort_keys=False)
for B, GA in [(8, 9), (4, 18)]:
    p = dict(c)
    p.update(run_name=f"probe_tfmr426_b{B}", batch_size=B, grad_accum=GA, max_tokens=72 * 512 * 60,
             log_every=10, eval_every=10**9, ckpt_every=10**9, keep_last=1, watchdog_enabled=False)
    yaml.safe_dump(p, open(f"temp/probe_tfmr426_b{B}.yaml", "w"), sort_keys=False)
    print(f"wrote temp/probe_tfmr426_b{B}.yaml")
