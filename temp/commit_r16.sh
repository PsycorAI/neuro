#!/bin/bash
# Commit the screen-era source flags (rounds 5-16), configs and docs. Logs are left out (live, tracked noise).
cd /home/glenn/projects/neuro
git add src/*.py configs/*.yaml docs/*.md docs/*.html
git commit -q -F - <<'EOF'
Screen flags rounds 5-16 (all default off, bit-identical) + v4 plan docs

Spiking stack (r5-r14): lif_levels/lif_signed/lif_learn_beta multi-level
signed LIF kernel (lif_exact_ml), pre_norm, mem_out_gate, canon_a, ff_conv,
swa read, late sparsity ramp, mqar_gap_max haystack curriculum,
data_independent_windows, attn_window, spike_rate_undiluted.

Interpretability round (r15): attn_ffn, SpikeGLU k-WTA FFN, top-k attention
(+ KV-cache path, _attn_weights audit helper), spk_layerscale.

v4 candidate round (r16, docs/v4_plan_review_2026_10_01.md):
- engram_v2_hash: hi  (fixes low-bit aliasing: ids x and x+K shared a bucket)
- attn_out_gate       (sigmoid gated attention output, removes sinks)
- attn_ffn_spikeglu_ramp_start/_end  (DSN k-ramp: dense, then sparsify late)
- attn_ffn_spikeglu_lif  (DSN-LIF: LIF-gated SpikeGLU, rate in sparsity loss)
- spk_out_norm: false (linear atoms)
- mqar_gap_start_frac / mqar_gap_weight (late-only haystack dose)
- attn_canon          (Canon-A taps in attention blocks)
Parity: temp/parity_r15_flags.py, temp/parity_r16_flags.py (flags off =
0.0 diff on 5 configs); CPU smoke temp/smoke_r16_cpu.py.

Docs: interpretable-spiking plan, interpretability code review, v4 recipe
research, v4 plan review, handoffs 09-29..10-01, to-do and journey updates.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
git log --oneline -1
git status --short | grep -v '^??' | grep -v ' logs/'
