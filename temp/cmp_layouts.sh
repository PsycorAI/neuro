#!/bin/bash
# cmp_layouts.sh -- 2026-10-04: print the arch keys of the 30M vs 150M gap-dose configs.
cd /home/glenn/projects/neuro/configs
for f in screen_r22d_w64x1_s0_30m r21g_w32x1early_150m r21i_x1early_150m; do
  echo "== $f"
  grep -E '^(d|n_neurons|d_mem|n_layers|attention_heads|attn_ffn_hidden|block_size|max_tokens|beta_floor|mqar_gap_[a-z_]+|attn_window_fixed|engram_v2_enabled|gdn2_enabled|delta_rule|sparsity_target|batch_size|lr|muon_lr|recurrent|mem_out_gate|canon_a):' $f.yaml | tr '\n' ' '
  echo; echo -n "attn layers: "; awk '/^attention_layer_positions:/{f=1;next} f&&/^- /{printf "%s ",$2;next} f{exit}' $f.yaml; echo
done
