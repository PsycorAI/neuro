#!/bin/bash
# Commit all uncommitted work from Neuro 49-50 (Glenn 10-04: "commit all uncommitted work").
# Skips the 4 big profiler traces (*.trace.json, 270 MB) and lock files.
set -e
cd /home/glenn/projects/neuro
git add src scripts docs configs cloud
git add logs -- ':!logs/*.trace.json' ':!logs/*.lock'
# temp/ is gitignored; force-add the scripts the handoff/memory files point to
for f in mk_g3_configs.py mk_r19_configs.py mk_r19_seeds.py mk_r20_mqonce.py mk_r21_configs.py mk_r22_configs.py \
         mk_r23_configs.py mk_r24_configs.py mk_bench_h200.py mk_u16_corpus.py mk_probe_g4_r19d.py \
         smoke_r19_cpu.py smoke_r24_cpu.py run_gap_queue.sh run_5080_sched.sh queue_5090_r24.sh \
         haystack_recall_eval.py stream_eval.py run_eval_r9.sh check_demo_stream.py diag_spike_levels.py \
         parity_sparseact.py median_tps.py cmp_layouts.sh push_bench.sh; do
  [ -f temp/$f ] && git add -f temp/$f || echo "skip (missing): temp/$f"
done
git status --short | grep -v '^??' | wc -l
git reset -q logs/*.trace.json 2>/dev/null || true
git commit -q -F - <<'EOF'
Rounds 17-24 screens, G3 150M depth runs, H200 bench, cloud scripts, demo stage 1

- src: SparseAct relu2 + homeostatic MLP (r19), attn_window_fixed, gap-MQAR padding, eager-cumsum Inductor
  workaround, per-block forward_sequence compile fix, uint16 prelut corpus (all default off = byte-identical)
- configs: g3/r19-r24 screens, v4_cloud_bench
- cloud/: Vast H200 setup/bench/run scripts
- scripts/demo: brain-view demo server + page (stage 1)
- docs: handoffs 10-02..10-05, journey, to-do, demo design
- logs: screen + run logs (profiler traces excluded)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
git log --oneline -1
git status --short | wc -l
