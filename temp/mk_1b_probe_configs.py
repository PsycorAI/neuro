#!/usr/bin/env python3
# mk_1b_probe_configs.py -- 2026-09-26: v4 A' (d2048/9L/N9728, ~1000M, gate_rank:16)
# tok/s + VRAM probe configs from the v4 draft. Micro-batch sweep at a constant
# 72 seqs/step (= v3's B18 x GA4), so tok/s is comparable across rows.
import pathlib, yaml
base = yaml.safe_load(open("configs/flagship_v4_363m.draft.yaml"))
for B, GA in [(12, 6), (8, 9), (4, 18)]:
    c = dict(base)
    c.update(run_name=f"probe_1b_b{B}", d=2048, d_mem=2048, n_layers=9,
             attention_layer_positions=[4], attention_heads=16, n_neurons=9728,
             batch_size=B, grad_accum=GA, max_tokens=72 * 512 * 60,   # 60 optimizer steps
             log_every=10, eval_every=10**9, ckpt_every=10**9, keep_last=1,
             watchdog_enabled=False)
    p = pathlib.Path(f"temp/probe_1b_b{B}.yaml")
    p.write_text(f"# AUTO-GEN temp/mk_1b_probe_configs.py -- v4 A' 1B probe, B={B} GA={GA}\n"
                 + yaml.safe_dump(c, sort_keys=False))
    print("wrote", p)
