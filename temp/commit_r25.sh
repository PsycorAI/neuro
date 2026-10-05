#!/bin/bash
# Commit the r25 build (Neuro 51, 2026-10-05).
set -e
cd /home/glenn/projects/neuro
git add src/hybrid_attn.py src/train_gpu.py src/nl_fact_aux.py configs/r25*.yaml configs/screen_p25*.yaml \
        docs/memory_beyond_window_2026_10_05.md docs/to-do.html logs/gap_queue.log
git add -f temp/mk_r25_configs.py temp/smoke_r25_cpu.py temp/smoke_r25_gpu.py temp/chunk_eval.py \
           temp/xchunk_haystack_eval.py temp/nl_fact_eval.py temp/run_r25_evals.sh temp/queue_5090_r25.sh \
           temp/queue_5080_r25.sh temp/queue_5080_r23.sh temp/run_5080_sched.sh temp/check_nl_fact_aux.py \
           temp/commit_r25.sh
git commit -q -F - <<'EOF'
r25: chunked attention + memory across chunks, cross-chunk MQAR, NL-fact aux, beyond-window evals

- attn_chunk: attention (and attn_canon) only inside aligned C-token chunks, memory over the whole sequence with full
  gradient; mqar_xchunk_frac: kv in chunk A, query in chunk B; nlfact_every (src/nl_fact_aux.py): NL fact episodes.
  All default off (CPU parity + compiled GPU smokes).
- evals: chunk_eval (per-chunk ppl, memory on/off), xchunk_haystack_eval, nl_fact_eval (run_r25_evals.sh)
- run_5080_sched.sh: UNTIL one-shot window mode
- docs/memory_beyond_window_2026_10_05.md: review, literature, ideas, decision rules

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
git log --oneline -1
