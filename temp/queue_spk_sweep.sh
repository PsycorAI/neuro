#!/bin/bash
# queue_spk_sweep.sh -- 2026-10-06 10:15. 5090: r25t (12L, L8, 80M tok). 5080 (after p25r -> p25s): p25s2 (L8), p25s3
# (2x thresholds), p25s4 (ternary), then p25t (spike_code from step 0, ~1.6 h; skipped if it can't finish by 17:30).
cd /home/glenn/projects/neuro
PO=models/checkpoints/screen_p25o_7L_chunk512_x_H3add/step_0006104.pt
setsid -f bash temp/queue_spk_conv.sh 0 r25t_13L_spk_L8_80M_150m models/checkpoints/r25o_13L_H6only_150m/step_0012208.pt
(
  while pgrep -f "bash temp/queue_5080_p25r_s.sh" > /dev/null; do sleep 60; done
  bash temp/queue_spk_conv.sh 1 screen_p25s2_7L_spk_L8 $PO
  SCALE=2.0 bash temp/queue_spk_conv.sh 1 screen_p25s3_7L_spk_sc2 $PO
  bash temp/queue_spk_conv.sh 1 screen_p25s4_7L_spk_tern $PO
  if [ "$(date +%H%M)" -lt 1545 ]; then
    GPU=1 UNTIL="2026-10-06 18:00" bash temp/run_5080_sched.sh screen_p25t_7L_spk_scratch
    ck=$(ls models/checkpoints/screen_p25t_7L_spk_scratch/step_*.pt 2>/dev/null | tail -1)
    [ -f models/checkpoints/screen_p25t_7L_spk_scratch/DONE ] && GPU=1 bash temp/run_r25_evals.sh "screen_p25t_7L_spk_scratch=$ck" \
      2>&1 | sed "s/^/  r25-evals /" >> logs/gap_queue.log
    [ -n "$ck" ] && CUDA_VISIBLE_DEVICES=1 /home/glenn/projects/bdh/venv/bin/python temp/energy_eval.py "screen_p25t_7L_spk_scratch=$ck" \
      2>&1 | grep energy | sed "s/^/  energy /" >> logs/gap_queue.log
  fi
  echo "[s1 $(date '+%m-%d %H:%M:%S')] spk sweep (5080) DONE" >> logs/gap_queue.log
) > /dev/null 2>&1 &
disown
echo "[q0 $(date '+%m-%d %H:%M:%S')] r25s: 5.56x energy (src 1.39x), memory intact, chunk1 45.00 (+7.8% vs r25o). Queued r25t (L8, 80M) on 5090; 5080 sweep p25s2 L8 / p25s3 2x thresh / p25s4 ternary / p25t from-scratch after p25s" >> logs/gap_queue.log
