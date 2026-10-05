"""mk_g3_configs.py -- 2026-10-03 DRAFT (needs Glenn's OK to launch). G3 mid-scale gate of the v4 plan
(docs/v4_plan_review_2026_10_01.md 6.6): ~150M, 12 layers, 2B tokens on the v4 corpus/vocab, to check 30M -> mid-scale
transfer of the sparse-MLP candidate vs a fair transformer, and to pick memory at depth (haystack at depth).
  g3_cand : H0 + 11 x [A + sparse MLP (relu2 + per-layer homeostatic threshold -> 10%, r19e, 5%)]
  g3_h2   : same, with a 2nd hippocampal (H) layer at position 8 (~2/3 depth) instead of an attention block
  g3_tfmr : fair 12-layer transformer (r16 t4 recipe: full attention, GELU MLP, Engram, gated attention, canon, AdamW)
Shape: d 768, 4d MLP / neurons, 12 heads, vocab 32768 (v4), block 512, B32 GA1. Standard exact speed flags.
NOTE: the transformer's lr (1e-3) was tuned at 30M; a 150M transformer may want ~6-8e-4 (LR-fairness lesson,
memory neuro_transformer_baseline_363m) -- decide before launch."""
import yaml
R = "/home/glenn/projects/neuro/configs/"
V4 = dict(vocab=32768, data_bin="/home/glenn/projects/bdh/data/tokenized_v4/train.bin", data_val_tokens=14527194,
          engram_v2_K=65536, max_tokens=1_000_000_000, eval_every=2000, ckpt_every=2000, keep_last=2,
          d=768, d_mem=768, n_layers=12, attention_heads=12, block_size=512, batch_size=32, grad_accum=1, seed=0)
SPEED = dict(gdn2_chunk_size=256, gdn2_ckpt_chunks=False, telemetry_eval_only=True, muon_batched_ns=True,
             compile=True, compile_dynamic=False, mqar_pad_to=65)


def emit(src, name, note, **upd):
    c = yaml.safe_load(open(R + src + ".yaml")); c.update(upd, run_name=name)
    with open(R + name + ".yaml", "w") as f:
        f.write(f"# {name} -- AUTO-GEN temp/mk_g3_configs.py (2026-10-03, DRAFT) from {src}. {note}\n")
        yaml.safe_dump(c, f, sort_keys=False)
    print("[OK]", name)


emit("screen_r19e_layer05_s0_30m", "g3_cand_150m", "H0 + 11 x [A + sparse MLP r19c].",
     **V4, **SPEED, n_neurons=3072, attention_layer_positions=list(range(1, 12)))
emit("screen_r19e_layer05_s0_30m", "g3_h2_150m", "As g3_cand with a 2nd H layer at position 8.",
     **V4, **SPEED, n_neurons=3072, attention_layer_positions=[p for p in range(1, 12) if p != 8])
emit("screen_r16_t4_tfmr4_fair_s0_30m", "g3_tfmr_150m", "Fair 12-layer transformer (r16 t4 recipe).",
     **V4, n_neurons=768, attention_layer_positions=list(range(0, 12)), compile=True, compile_dynamic=False, lr=0.0008)
