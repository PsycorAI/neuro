"""mk_g4_configs.py -- 2026-10-02 (Glenn OK): G4 1B layout probe configs (docs/v4_plan_review_2026_10_01.md 6.3-6.4,
docs/speed_plan_v4_2026_10_02.md). Shape S1: d 2048, vocab 32768, block 512.
  probe_g4_c0s1   = c0 layout: H0 (n_neurons 4d) + 12 x [gated top-k attention + DSN 4d (hidden 8192, k 768 ~9.4%)]
  probe_g4_tfmrs1 = fair transformer, same params/FLOPs: 13 x [attention + GELU 6d (12288)] (6d GELU = 12 d^2 = DSN 4d)
Both carry the standard exact speed flags (chunk 256, no chunk ckpt, telemetry_eval_only, batched NS, mqar_pad_to)."""
import yaml
R = "/home/glenn/projects/neuro/configs/"
SPEED = dict(gdn2_chunk_size=256, gdn2_ckpt_chunks=False, telemetry_eval_only=True, muon_batched_ns=True,
             mqar_pad_to=65)
S1 = dict(vocab=32768, d=2048, d_mem=2048, attention_heads=16, engram_v2_K=65536, block_size=512,
          data_bin="/home/glenn/projects/bdh/data/tokenized_v4/train.bin", data_val_tokens=14527194,
          lut_subword_fallback=True, compile_mode="default", compile_dynamic=True)

c = yaml.safe_load(open(R + "screen_r16_c0_s0_30m.yaml"))
c.update(S1, **SPEED)
c.update(n_neurons=8192, n_layers=13, attention_layer_positions=list(range(1, 13)),
         attn_ffn_hidden=8192, attn_ffn_spikeglu_k=768, run_name="probe_g4_c0s1")
t = yaml.safe_load(open(R + "screen_r16_t4_tfmr4_fair_s0_30m.yaml"))
t.update(S1, **SPEED)
t.update(n_neurons=2048, n_layers=13, attention_layer_positions=list(range(13)), attn_ffn_hidden=12288,
         run_name="probe_g4_tfmrs1")
for name, cfg in [("probe_g4_c0s1", c), ("probe_g4_tfmrs1", t)]:
    with open(R + name + ".yaml", "w") as f:
        f.write(f"# {name} -- AUTO-GEN temp/mk_g4_configs.py (2026-10-02 G4 1B layout probe, timing only)\n")
        yaml.safe_dump(cfg, f, sort_keys=False)
    print("[OK]", name)
